"""Self-play PPO training loop for the vanilla-authentic Minecraft PvP bot.

Launches the Rust simulation backend as a subprocess, connects over UDP
(binary datagrams - see env.py / sim/src/protocol.rs), and trains a single
shared policy that controls every player slot on both teams in every arena
(self-play falls out for free because observations are always
self/team-relative - see features.py). Supports 1v1 (the default) and NvN
team fights via --team-size.

Run with defaults (1v1):
    python train.py

Run a 2v2:
    python train.py --team-size 2 --fresh

Everything is uncoupled from real time: the sim never sleeps, and this
loop pulls state batches as fast as the sim + this process can produce
and consume them.

To stop pure self-play converging to an equilibrium only a copy of itself
can't punish, `--opponent-fraction` of the arenas instead pit the learner
against an opponent drawn from `opponents.py` - a rolling pool of frozen
policy snapshots plus a fixed scripted heuristic bot - with those slots
masked out of the PPO update. `win_vs_scripted` in the progress log is a
non-circular strength signal.
"""

import argparse
import json
import os
import random
import signal
import tempfile
import time

import numpy as np
import torch

import features
from device import resolve_device
from env import DEFAULT_SIM_BINARY, SelfPlayArenaEnv
from frame_stack import FrameStacker
from logging_setup import get_logger
from metrics import MetricsWriter
from opponents import OpponentPool, benchmark_slot_mask, opponent_slot_mask
from ppo_agent import ActorCritic, RolloutBuffer, ppo_update

log = get_logger(__name__)

# CLI flag -> (config section, key). `None` section means a top-level key.
_SIM_CONFIG_OVERRIDES = {
    "kit": (None, "kit"),
    "team_size": (None, "team_size"),
    "max_observed_enemies": (None, "max_observed_enemies"),
    "max_observed_teammates": (None, "max_observed_teammates"),
    "max_observed_projectiles": (None, "max_observed_projectiles"),
    "block_view_size": (None, "block_view_size"),
    "arena_radius": (None, "arena_radius"),
    "match_time": (None, "match_time_seconds"),
    "terrain_max_amplitude": (None, "terrain_max_amplitude"),
    "terrain_flat_only": (None, "terrain_flat_only"),
    "max_look_delta": (None, "max_look_delta"),
    "reward_per_hp_dealt": ("reward", "per_hp_dealt"),
    "reward_per_hp_taken": ("reward", "per_hp_taken"),
    "reward_win": ("reward", "win"),
    "reward_loss": ("reward", "loss"),
    "reward_sweep_penalty": ("reward", "sweep_penalty"),
    "reward_friendly_fire_penalty": ("reward", "friendly_fire_penalty"),
}

# Default `--num-arenas` is derived from the detected CPU count rather than
# a single hardcoded number, so it's reasonable on both a laptop and a big
# training box instead of being tuned to whatever one machine it was last
# benchmarked on. The ratio comes from benchmarking throughput (raw sim,
# sim + policy forward pass, and the full train loop) across a sweep of
# --num-arenas on a 12-thread machine: env-steps/sec rose steeply up to
# ~256 arenas, then plateaued and started giving a little back past ~384 as
# the sim's rayon pool and the policy batch began contending for the same
# cores. The ratio and cap below sit *inside* that plateau on purpose - they
# used to be 20/thread capped at 512, which cost little extra throughput but
# a lot of RAM on many-core boxes (the PPO rollout buffer scales linearly
# with num_arenas). Bump `--num-arenas` explicitly if you have the memory
# and want to push past the plateau. See README.md's Configuration section.
_ARENAS_PER_CPU = 12
# Keep a small/limited machine from getting an unhelpfully tiny default...
_MIN_NUM_ARENAS = 32
# ...and cap a very large one at the near-flat part of the throughput curve
# so a big box doesn't silently allocate a multi-GB rollout buffer.
_MAX_NUM_ARENAS = 256
# Used only if the CPU count genuinely can't be read (see
# `_detect_cpu_count`) - a modest value that should run fine on effectively
# any machine, not a performance target.
_FALLBACK_NUM_ARENAS = 64


def _detect_cpu_count() -> int | None:
    """The number of CPUs actually available to this process, or `None` if
    it can't be determined. Prefers `os.sched_getaffinity` over
    `os.cpu_count()` where available (Linux only) since it reflects
    cgroup/container/taskset CPU limits - the number of cores this process
    can actually schedule onto - rather than the host's raw core count,
    which would over-estimate inside a limited container.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()  # sched_getaffinity doesn't exist on macOS/Windows


# Rough per-(step, agent-slot) cost of the on-CPU PPO RolloutBuffer: the
# observation row plus the stored raw-continuous / binary / slot / logprob /
# reward / done / value columns, all float32 (~4 bytes). Used only for the
# startup RAM estimate and the `--max-ram` back-solve, so an over-estimate
# is the safe direction.
_ROLLOUT_BYTES_PER_SLOT_STEP = (256 + 20) * 4


def _parse_size(text: str) -> int:
    """`"3G"` / `"750M"` / `"512k"` / a bare byte count -> bytes."""
    t = text.strip().lower().rstrip("b")
    mult = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    if t and t[-1] in mult:
        return int(float(t[:-1]) * mult[t[-1]])
    return int(t)


def _rollout_buffer_bytes(num_arenas: int, rollout_len: int) -> int:
    return rollout_len * (2 * num_arenas) * _ROLLOUT_BYTES_PER_SLOT_STEP


def resolve_num_arenas(explicit: int | None, rollout_len: int, max_ram: str | None) -> int:
    """Returns `explicit` unchanged if the user passed `--num-arenas`,
    otherwise derives a default from the detected CPU count (falling back
    to `_FALLBACK_NUM_ARENAS` if that can't be read at all) - see the
    module-level comment above `_ARENAS_PER_CPU`. `--max-ram` (when set and
    `--num-arenas` is not) further caps the result so the estimated rollout
    buffer fits the budget.
    """
    def _log_estimate(n: int) -> int:
        log.info(
            "PPO rollout buffer ~%.0f MB (rollout_len=%d x %d agent-slots) - lower --num-arenas / "
            "--rollout-len, or set --max-ram, if this is too much",
            _rollout_buffer_bytes(n, rollout_len) / 1024**2,
            rollout_len,
            2 * n,
        )
        return n

    if explicit is not None:
        return _log_estimate(explicit)

    cpu_count = _detect_cpu_count()
    if not cpu_count:
        log.warning(
            "could not detect CPU count - falling back to --num-arenas=%d "
            "(pass --num-arenas explicitly to tune this for your machine)",
            _FALLBACK_NUM_ARENAS,
        )
        return _log_estimate(_FALLBACK_NUM_ARENAS)

    num_arenas = min(max(cpu_count * _ARENAS_PER_CPU, _MIN_NUM_ARENAS), _MAX_NUM_ARENAS)
    reason = f"{cpu_count} CPU thread(s) x ~{_ARENAS_PER_CPU}, clamped to [{_MIN_NUM_ARENAS}, {_MAX_NUM_ARENAS}]"

    if max_ram:
        budget = _parse_size(max_ram)
        fit = budget // (rollout_len * 2 * _ROLLOUT_BYTES_PER_SLOT_STEP)
        capped = max(_MIN_NUM_ARENAS, min(num_arenas, fit))
        if capped < num_arenas:
            reason = f"--max-ram {max_ram} fits ~{capped} arenas (was {num_arenas} from {reason})"
            num_arenas = capped

    log.info("auto-selected --num-arenas=%d (%s) - pass --num-arenas to override", num_arenas, reason)
    return _log_estimate(num_arenas)


def resolve_sim_config(args) -> str | None:
    """Merges `--sim-config <file>` (if any) with the individual
    `--arena-radius` / `--reward-*` / ... override flags and, if the result
    is non-empty, writes it to a temp JSON file whose path is returned for
    the sim's `--config`. Returns None if there is nothing to override."""
    config: dict = {}
    if args.sim_config:
        with open(args.sim_config) as f:
            config = json.load(f)

    for flag, (section, key) in _SIM_CONFIG_OVERRIDES.items():
        value = getattr(args, flag)
        if value is None or value is False:
            continue
        target = config.setdefault(section, {}) if section else config
        target[key] = value

    # These are real tri-states (unset / on / off), so they can't go through
    # the "skip falsy" loop above.
    if args.friendly_fire is not None:
        config["friendly_fire"] = args.friendly_fire
    if args.attribute_swapping is not None:
        config["attribute_swapping"] = args.attribute_swapping
    if args.natural_regen is not None:
        config["natural_regen"] = args.natural_regen
    if args.input_order is not None:
        config["input_order"] = args.input_order

    if not config:
        return None

    fd, path = tempfile.mkstemp(prefix="sim_config_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2)
    log.info("resolved sim config -> %s: %s", path, config)
    return path


def build_actions(act_out: dict, num_slots: int) -> np.ndarray:
    """Turns the policy output into a `[num_slots, ACTION_FLOATS_PER_SLOT]`
    float32 array, one `[move_x, move_z, yaw_delta, pitch_delta, jump, attack,
    sprint, use_item, sneak, held_slot]` row per slot - the exact flat wire
    form `env.step` sends to the Rust sim. All vectorized numpy over the whole
    batch.
    """
    squashed = act_out["squashed_cont"].cpu().numpy()
    binary = act_out["binary_action"].cpu().numpy()
    slot = act_out["slot_action"].cpu().numpy()

    max_look_delta = features.active_constants().max_look_delta
    rows = np.empty((num_slots, features.ACTION_FLOATS_PER_SLOT), dtype=np.float32)
    rows[:, 0] = squashed[:, 0]  # move_x
    rows[:, 1] = squashed[:, 1]  # move_z
    rows[:, 2] = squashed[:, 2] * max_look_delta  # yaw_delta
    rows[:, 3] = squashed[:, 3] * max_look_delta  # pitch_delta
    b = features.BINARY_ACTION_DIM
    rows[:, 4 : 4 + b] = binary  # jump, attack, sprint, use_item, sneak (0.0/1.0 samples)
    rows[:, 4 + b] = slot  # held-slot action: 0..8 select, 9..20 hotkey item id (a-9)
    return rows  # ndarray [num_slots, ACTION_FLOATS_PER_SLOT]; env.step consumes it without a copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-arenas",
        type=int,
        default=None,
        help="parallel arenas (2 agents each). Default: auto-detected from this machine's CPU count "
        f"(~{_ARENAS_PER_CPU} arenas/thread, clamped to [{_MIN_NUM_ARENAS}, {_MAX_NUM_ARENAS}]; falls "
        f"back to {_FALLBACK_NUM_ARENAS} if the CPU count can't be read) - pass an explicit value to "
        "override, e.g. after benchmarking your own machine (see README's Configuration section).",
    )
    parser.add_argument(
        "--max-ram",
        type=str,
        default=None,
        help="RAM budget for the PPO rollout buffer, e.g. '3G', '750M'. When set and --num-arenas is "
        "not, the auto-detected arena count is capped so the estimated buffer fits (never below "
        f"{_MIN_NUM_ARENAS} arenas). The buffer scales linearly with num_arenas x rollout_len.",
    )
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--sim-binary",
        type=str,
        default=DEFAULT_SIM_BINARY,
        help=f"path to the built Rust simulation binary (default: {DEFAULT_SIM_BINARY}, "
        "picking the .exe suffix automatically on Windows)",
    )
    parser.add_argument("--rollout-len", type=int, default=128, help="steps per agent before a PPO update")
    parser.add_argument("--total-updates", type=int, default=100_000)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate - the step size for each weight update (default 3e-4, the standard "
        "PPO value; too high diverges, too low crawls). Linearly annealed to 0 over the run unless "
        "--no-anneal-lr.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for future rewards (default 0.99). Lower = more myopic; at 0.99 a reward "
        "~100 steps out still carries ~37%% of its weight, which suits match lengths here.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE(lambda) bias/variance trade-off for advantage estimation (default 0.95). Toward 1.0 = "
        "lower bias / higher variance; toward 0 = the reverse. 0.95 is the usual PPO sweet spot.",
    )
    parser.add_argument(
        "--ppo-epochs",
        "--epochs",
        dest="ppo_epochs",
        type=int,
        default=2,
        help="how many optimization passes over each collected rollout (default 2). This is the "
        "single biggest lever on update wall-time (it scales linearly), and the sim is fast enough "
        "that collecting fresh experience is cheaper than squeezing every rollout dry - so 2 beats "
        "the classic 4 on wall-clock-to-skill here. Raise it (4-10) if you're data-limited rather "
        "than compute-limited, e.g. a much slower sim config or a tiny --num-arenas.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=4096,
        help="samples per gradient step (default 4096). The rollout (rollout_len * 2 * num_arenas "
        "samples) is shuffled and split into chunks of this size each epoch.",
    )
    parser.add_argument(
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO surrogate clip range (default 0.2, the canonical value) - caps how far one update "
        "can move the policy by clipping the probability ratio to [1-clip, 1+clip].",
    )
    parser.add_argument(
        "--value-coef",
        type=float,
        default=0.5,
        help="weight of the value-function loss in the total loss (default 0.5).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="global gradient-norm clip before each optimizer step (default 0.5) - guards against a "
        "single bad batch blowing up the weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for Python/NumPy/Torch RNGs so a run is reproducible (default: nondeterministic). "
        "Also used as the sim's base seed when --sim-seed isn't given.",
    )

    net = parser.add_argument_group("network architecture (saved into the checkpoint)")
    net.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="trunk width (default 256) - a balanced default: enough capacity for nuanced combat "
        "behaviour from the 24-dim observation, still tiny enough that CPU rollout collection stays "
        "latency-bound. Changing this makes existing checkpoints unresumable (pass --fresh).",
    )
    net.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="trunk depth in Linear+Tanh blocks (default 2) - the standard actor-critic depth; deeper "
        "rarely helps a low-dim control task and slows every rollout step.",
    )
    net.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="observations fed to the policy = the last N frames concatenated (default 1 = the "
        "memoryless MLP). N>1 gives the same trunk a short history for the latency-delayed, "
        "occlusion-limited view of other players, at N x the input width. Changing it makes existing "
        "checkpoints unresumable (pass --fresh). NOTE: the live inference bridges (azalea-bot, mod) "
        "don't stack frames yet, so an N>1 checkpoint can't be exported for live play.",
    )

    sim = parser.add_argument_group(
        "simulation config",
        "Written to a temp JSON file and passed to the sim's --config. "
        "Start from a full file with --sim-config and/or set individual knobs below.",
    )
    sim.add_argument("--sim-config", type=str, default=None, help="JSON file of SimConfig overrides")
    sim.add_argument("--sim-seed", type=int, default=None, help="sim base RNG seed (default: random)")
    sim.add_argument(
        "--kit",
        choices=["sword", "axe", "uhc"],
        default=None,
        help="combat kit (sim default: sword = diamond sword + full diamond armour, no shield). "
        "'axe' adds a shield + diamond axe + bow + crossbow + 6 arrows; 'uhc' is the full Ultra "
        "Hardcore kit (shield, Sharpness/Power/Piercing gear, golden apples & heads, placeable "
        "planks/cobweb/water/lava, and no natural regeneration).",
    )
    sim.add_argument(
        "--attribute-swapping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="model the vanilla MC-28289 attribute-swap bug: switching the held item on the same "
        "tick as an attack resolves the hit with the previous item's damage/speed/traits (sim "
        "default: on). --no-attribute-swapping disables it (the modern 26.2+ behaviour).",
    )
    sim.add_argument(
        "--input-order",
        choices=["legacy", "modern"],
        default=None,
        help="client input resolution order (sim default: legacy = 1.21.11 'hotbar->attack->use', "
        "so a same-tick swap+attack does an attribute swap and swapping is free). 'modern' = the "
        "26.2-pre-2 'attack->use->hotbar' order: attribute swapping is off and every hotbar swap "
        "costs a one-tick attack/use lockout.",
    )
    sim.add_argument(
        "--natural-regen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="slow passive HP regen when not recently hurt (sim default: off; always off for the uhc kit)",
    )
    sim.add_argument(
        "--team-size",
        type=int,
        default=None,
        help="players per team: 1 = 1v1 duel (default), 2 = 2v2, 3 = 3v3, ... Each arena holds "
        "2*team_size policy slots. Changing this changes the observation size, so a checkpoint "
        "trained at one team size can't be resumed at another (use --fresh).",
    )
    sim.add_argument(
        "--friendly-fire",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="whether melee/sweep can hurt a teammate (sim default: on, vanilla-authentic). "
        "--no-friendly-fire makes allies immune to each other.",
    )
    sim.add_argument(
        "--max-observed-enemies",
        type=int,
        default=None,
        help="how many nearest living enemies each observation describes in full (sim default 3)",
    )
    sim.add_argument(
        "--max-observed-teammates",
        type=int,
        default=None,
        help="how many nearest living teammates each observation describes in full (sim default 2)",
    )
    sim.add_argument(
        "--max-observed-projectiles",
        type=int,
        default=None,
        help="how many nearest in-flight arrows each observation describes (sim default 2)",
    )
    sim.add_argument(
        "--block-view-size",
        type=int,
        default=None,
        help="side length (block columns) of the yaw-rotated block-grid view in each observation "
        "- lets the policy see terrain, placed blocks, water and lava (sim default 5; 0 to omit). "
        "Larger costs O(n^2) observation floats.",
    )
    sim.add_argument("--arena-radius", type=float, default=None)
    sim.add_argument("--match-time", type=float, default=None, help="match_time_seconds")
    sim.add_argument("--terrain-max-amplitude", type=float, default=None)
    sim.add_argument("--terrain-flat-only", action="store_true", help="force every arena flat")
    sim.add_argument("--max-look-delta", type=float, default=None, help="per-step yaw/pitch clamp (radians)")
    sim.add_argument("--reward-per-hp-dealt", type=float, default=None)
    sim.add_argument("--reward-per-hp-taken", type=float, default=None)
    sim.add_argument("--reward-win", type=float, default=None)
    sim.add_argument("--reward-loss", type=float, default=None)
    sim.add_argument("--reward-sweep-penalty", type=float, default=None)
    sim.add_argument(
        "--reward-friendly-fire-penalty",
        type=float,
        default=None,
        help="reward subtracted per HP of damage dealt to a teammate (sim default 1.0)",
    )
    parser.add_argument(
        "--ent-coef",
        "--entropy-coef-start",
        dest="entropy_coef_start",
        type=float,
        default=0.01,
        help="entropy bonus coefficient at the start of training (default 0.01; higher = more "
        "exploration). Rewards the policy for keeping its action distribution spread out so it "
        "doesn't collapse onto one strategy before it has enough signal. Linearly annealed to "
        "--ent-coef-final over --total-updates.",
    )
    parser.add_argument(
        "--ent-coef-final",
        "--entropy-coef-end",
        dest="entropy_coef_end",
        type=float,
        default=0.001,
        help="entropy bonus coefficient by the final update - linearly annealed from "
        "--ent-coef over --total-updates. A fixed (non-annealed) coefficient "
        "leaves the entropy bonus permanently strong enough to keep rewarding "
        "randomness even after the policy has plenty of signal to commit to a more "
        "decisive strategy - in practice this looked like continuous actions (look "
        "direction especially) staying near pure noise and binary actions (jump/attack) "
        "hovering near a 50/50 coin flip every tick no matter how long training ran.",
    )
    league = parser.add_argument_group(
        "opponents / league",
        "Widen the self-play opponent beyond a live mirror of the current "
        "policy, so it doesn't converge to something only a copy of itself "
        "can't punish. The pool is persisted to <checkpoint-dir>/league/ and "
        "reloaded on resume (the scripted bot is always available); --fresh "
        "wipes it along with the checkpoints.",
    )
    league.add_argument(
        "--opponent-fraction",
        type=float,
        default=0.3,
        help="fraction of arenas whose team B is handed to an opponent (a frozen policy snapshot or "
        "the scripted bot) instead of the live policy; those slots are masked out of the PPO update. "
        "0 = pure self-play (the old behaviour).",
    )
    league.add_argument(
        "--scripted-opponent-prob",
        type=float,
        default=0.35,
        help="of the iterations that use an opponent, the probability it's the fixed scripted "
        "heuristic bot rather than a random pool snapshot (the rest fall back to self-play until the "
        "pool has its first snapshot).",
    )
    league.add_argument(
        "--opponent-pool-size",
        type=int,
        default=8,
        help="how many past policy snapshots the league keeps (oldest dropped). 0 disables snapshots "
        "(scripted-bot opponent only).",
    )
    league.add_argument(
        "--opponent-snapshot-every",
        type=int,
        default=25,
        help="PPO updates between freezing the current policy into the opponent pool.",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.0,
        help="std-dev of gaussian noise added to every observation feature during rollout collection "
        "(the same noised observation is what the policy acts on AND what's stored, so PPO stays "
        "consistent - it just trains against a noisier POMDP). A small value (~0.01) is cheap "
        "sim-to-real robustness against the live bot's imperfect state estimation; large values hurt.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="PPO updates between progress lines (and metrics rows). The rolling return / win-rate "
        "averages cover this window.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        nargs="?",
        const="AUTO",
        default=None,
        help="also write the progress metrics (returns, win rates, losses, entropy, KL, clip "
        "fraction, steps/sec) as CSV - one row per logged update, at the same cadence as the "
        "console line. Bare --metrics-csv writes <checkpoint-dir>/metrics.csv; pass a path to "
        "choose the file. Appended to (under the existing header) on resume.",
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        nargs="?",
        const="AUTO",
        default=None,
        help="also log the same metrics to a TensorBoard event dir (needs `pip install "
        "tensorboard`; a missing package is warned about, not fatal). Bare --tensorboard uses "
        "<checkpoint-dir>/tb; pass a path to choose the dir.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="PPO updates between checkpoints (default 200). Each save writes `latest.pt` (the "
        "auto-resume point) and a numbered `policy_update_<n>.pt` history file. A checkpoint is "
        "also written once on exit (Ctrl+C / crash / finish) so an interrupted run isn't lost.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="how many numbered policy_update_*.pt snapshots to keep on disk (default 3). Older ones "
        "are pruned right after each save so the checkpoint dir doesn't grow without bound over a long "
        "run. `latest.pt` is never pruned. Set to 0 to keep every snapshot forever.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto' (default) picks the fastest accelerator present: CUDA/ROCm, Apple MPS, Intel XPU, "
        "DirectML, then CPU. Or force one: cuda / cpu / mps / xpu / directml (see python/device.py).",
    )
    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="linearly decay the learning rate to 0 over --total-updates (on by default; "
        "--no-anneal-lr keeps it fixed). A decaying LR lets PPO keep making large early "
        "updates while settling into a stable policy later, which converges noticeably faster "
        "in practice than a fixed rate.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="intra-op CPU threads for torch DURING ROLLOUT COLLECTION (default 0 = auto: "
        "min(4, cpu_count // 3)). Each rollout step is one small forward pass, so this saturates "
        "fast - a handful of threads gives ~1.4x, a thread-per-core then regresses by fighting the "
        "Rust sim's own pool for cores. The PPO update wants far more - see --update-threads. Set 1 "
        "for the old strictly-single-thread rollout.",
    )
    parser.add_argument(
        "--update-threads",
        type=int,
        default=0,
        help="intra-op CPU threads for torch DURING THE PPO UPDATE + GAE (default 0 = auto: "
        "min(8, cpu_count - 2)). Unlike rollout, the update is a big batched matmul that the sim "
        "sits idle through, so it wants most of your cores - measured ~2.5x faster at 6 threads on "
        "a 12-core box (12 threads regresses from oversubscription). Ignored on GPU. Set 1 to "
        "disable and keep the old single-thread behaviour.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_policy",
        action="store_true",
        help="wrap the policy in torch.compile(mode='reduce-overhead'). Speeds up both the rollout "
        "forward pass and the PPO update's re-evaluation, but the first update pays a multi-second "
        "compile and it depends on a working torch 2.x compiler toolchain - off by default.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete EVERY checkpoint in --checkpoint-dir (latest.pt and all numbered snapshots) and "
        "start a fresh randomly-initialized model. By default, if latest.pt exists, training resumes "
        "from it automatically (model + optimizer state + update count) - so stopping training (Ctrl+C, "
        "a crash, a reboot) and running the same command again continues where it left off instead of "
        "silently restarting from scratch. Pass --fresh to wipe it and start completely over.",
    )
    args = parser.parse_args()
    args.num_arenas = resolve_num_arenas(args.num_arenas, args.rollout_len, args.max_ram)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        log.info("seeded Python/NumPy/Torch RNGs with --seed=%d", args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    cpu = _detect_cpu_count() or 4
    if args.torch_threads > 0:
        rollout_threads = args.torch_threads
    else:
        rollout_threads = min(4, max(1, cpu // 3))
    if args.update_threads > 0:
        update_threads = args.update_threads
    else:
        update_threads = min(8, max(1, cpu - 2))
    torch.set_num_threads(rollout_threads)
    device = resolve_device(args.device)
    # torch.set_num_threads is a no-op for GPU kernels; only flip it on CPU.
    cpu_threaded_update = device.type == "cpu" and update_threads != rollout_threads
    log.info(
        "device=%s rollout torch_threads=%d, PPO-update torch_threads=%d%s",
        device,
        rollout_threads,
        update_threads,
        "" if cpu_threaded_update else " (not switched: GPU or same count)",
    )

    sim_config_path = resolve_sim_config(args)

    csv_path = (
        os.path.join(args.checkpoint_dir, "metrics.csv") if args.metrics_csv == "AUTO" else args.metrics_csv
    )
    tb_dir = (
        os.path.join(args.checkpoint_dir, "tb") if args.tensorboard == "AUTO" else args.tensorboard
    )
    metrics = MetricsWriter(csv_path=csv_path, tensorboard_dir=tb_dir)

    env = None
    model = None
    pool = None
    optimizer = None
    last_update_completed = 0
    last_saved_update = 0
    try:
        env = SelfPlayArenaEnv(
            num_arenas=args.num_arenas,
            port=args.port,
            sim_binary=args.sim_binary,
            sim_config_path=sim_config_path,
            seed=args.sim_seed if args.sim_seed is not None else args.seed,
        )
        num_slots = env.num_slots
        # features.OBS_DIM is only final after env's handshake calls
        # features.configure() (it depends on the sim's team-observation
        # config), so read it here rather than importing it at module load.
        base_obs_dim = features.OBS_DIM
        frame_stack = max(1, args.frame_stack)
        # The width the policy actually sees: `--frame-stack` frames concatenated.
        obs_dim = base_obs_dim * frame_stack
        slot_dim = features.HOTBAR_ACTION_DIM
        log.info(
            "kit=%s, %dv%d, %d policy slots, obs_dim=%d (base %d x frame_stack %d), "
            "held-slot classes=%d (%d select + %d hotkey)",
            env.kit, env.team_size, env.team_size, num_slots, obs_dim, base_obs_dim, frame_stack,
            slot_dim, features.HOTBAR_SLOTS, features.ITEM_COUNT,
        )

        model = ActorCritic(
            obs_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, slot_dim=slot_dim
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        start_update = 1
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if args.fresh:
            removed = _wipe_checkpoints(args.checkpoint_dir)
            log.info("--fresh given - deleted %d checkpoint file(s) in %s, starting from scratch",
                     removed, args.checkpoint_dir)
        elif os.path.isfile(latest_path):
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            ckpt_arch = ckpt.get("arch") or {}
            requested_arch = {"hidden_size": args.hidden_size, "num_layers": args.num_layers}
            ckpt_slot_dim = ckpt_arch.get("slot_dim", slot_dim)
            if any(ckpt_arch.get(k) != v for k, v in requested_arch.items()) or ckpt_slot_dim != slot_dim:
                raise SystemExit(
                    f"checkpoint architecture {ckpt_arch} != requested "
                    f"hidden_size={args.hidden_size} num_layers={args.num_layers} slot_dim={slot_dim}; "
                    "pass matching --hidden-size/--num-layers (and kit) to resume, or --fresh to start over"
                )
            ckpt_frame_stack = ckpt.get("frame_stack", 1)
            if ckpt_frame_stack != frame_stack:
                raise SystemExit(
                    f"checkpoint was trained with --frame-stack {ckpt_frame_stack}, but this run asked "
                    f"for {frame_stack}; pass --frame-stack {ckpt_frame_stack} to resume, or --fresh to "
                    "start over"
                )
            ckpt_obs_dim = ckpt.get("obs_dim")
            if ckpt_obs_dim is not None and ckpt_obs_dim != obs_dim:
                raise SystemExit(
                    f"checkpoint was trained with obs_dim={ckpt_obs_dim}, but this code's observation "
                    f"space is now obs_dim={obs_dim} (the feature set in features.py or the team-fight "
                    "observation config changed since this checkpoint was saved); it can't be resumed "
                    "as-is - pass --fresh to start a new run with the current observation space"
                )
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                log.warning("checkpoint predates optimizer-state saving - Adam's momentum starts fresh")
            start_update = ckpt.get("update", 0) + 1
            log.info("resumed from %s at update=%d", latest_path, start_update)
        else:
            log.info("no checkpoint found at %s - starting fresh", latest_path)

        # Rollout collection (model.act on one tiny batch per env step) runs
        # on its own CPU copy of the policy, kept in sync with the real
        # (training) model after every PPO update. This trunk is small
        # (hidden_size~256, a couple of layers) and each rollout step is
        # already paced by a UDP round-trip to the Rust sim - on a CUDA
        # device that combination means `model.act` plus the `.cpu().numpy()`
        # in `build_actions` would force a host<->device sync every single
        # env step for a network far too small to benefit from the GPU at
        # that batch size. Collecting on CPU instead avoids all of that and
        # only pays for one bulk transfer per rollout (`buffer.to(device)`
        # below) right before the update, where the GPU's larger minibatches
        # actually do help. When `device` is already CPU this is just an
        # alias - no extra cost, no extra copies.
        collect_model = model if device.type == "cpu" else ActorCritic(
            obs_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, slot_dim=slot_dim
        )
        if collect_model is not model:
            collect_model.load_state_dict(model.state_dict())

        if args.compile_policy:
            # Compile the `forward` method in place (not the whole module) so
            # `act()` / `evaluate()` pick it up via `self.forward(...)` while
            # `model` stays a plain ActorCritic - the optimizer, state_dict
            # and checkpoint paths are all untouched.
            log.info("--compile: compiling the policy forward - the first update will be slow")
            model.forward = torch.compile(model.forward, mode="reduce-overhead")
            if collect_model is not model:
                collect_model.forward = torch.compile(collect_model.forward, mode="reduce-overhead")

        buffer = RolloutBuffer(args.rollout_len, num_slots, obs_dim, torch.device("cpu"))

        # Opponent league: for --opponent-fraction of the arenas, team B is
        # played by a frozen snapshot or the scripted bot instead of the live
        # policy. `learner_slots` marks the slots that still train.
        opp_rng = np.random.default_rng(args.seed)
        pool = OpponentPool(
            make_model=lambda: ActorCritic(
                obs_dim, hidden_size=args.hidden_size, num_layers=args.num_layers, slot_dim=slot_dim
            ),
            capacity=args.opponent_pool_size,
            snapshot_dir=os.path.join(args.checkpoint_dir, "league"),
        )
        if not args.fresh:
            pool.load()
        learner_slots = opponent_slot_mask(num_slots // env.players_per_arena, env.players_per_arena, args.opponent_fraction)
        bench_slots = benchmark_slot_mask(num_slots // env.players_per_arena, env.players_per_arena, args.opponent_fraction)
        opp_idx = np.nonzero(~learner_slots)[0]
        learner_mask_t = torch.as_tensor(learner_slots)
        n_opp_arenas = int(round((num_slots // env.players_per_arena) * args.opponent_fraction))
        if opp_idx.size:
            log.info(
                "opponent league: %d/%d arenas play the learner vs an opponent (%d opponent slots); "
                "pool size %d, snapshot every %d updates, scripted-opponent prob %.2f",
                n_opp_arenas, num_slots // env.players_per_arena, opp_idx.size,
                args.opponent_pool_size, args.opponent_snapshot_every, args.scripted_opponent_prob,
            )
        else:
            log.info("opponent league disabled (--opponent-fraction 0) - pure self-play")
        scripted_wins = 0
        scripted_matches = 0

        # `cur_obs_np` is always the latest single frame (the scripted
        # opponent reads fixed indices out of it); `obs` is what the policy
        # sees - identical to `cur_obs_np` unless --frame-stack > 1.
        stacker = FrameStacker(num_slots, base_obs_dim, frame_stack) if frame_stack > 1 else None
        cur_obs_np = env.reset()
        if args.obs_noise > 0.0:
            cur_obs_np = cur_obs_np + opp_rng.normal(0.0, args.obs_noise, cur_obs_np.shape).astype(np.float32)
        obs = torch.as_tensor(stacker.reset(cur_obs_np) if stacker else cur_obs_np)

        # Rolling win-rate / episode-return trackers purely for console logging.
        episode_return = np.zeros(num_slots, dtype=np.float32)
        recent_returns: list = []
        recent_wins = 0
        recent_matches = 0

        total_env_steps = 0
        start_time = time.time()

        for update in range(start_update, args.total_updates + 1):
            buffer.reset()

            # Pick this iteration's opponent (shared by every opponent arena).
            if opp_idx.size:
                opp_kind, opp_net = pool.choose(opp_rng, args.scripted_opponent_prob)
            else:
                opp_kind, opp_net = "none", None
            iter_learner = learner_slots if opp_kind != "none" else np.ones(num_slots, dtype=bool)
            iter_mask_t = learner_mask_t if opp_kind != "none" else None

            for _ in range(args.rollout_len):
                act_out = collect_model.act(obs)
                actions = build_actions(act_out, num_slots)

                if opp_kind == "scripted":
                    actions[opp_idx] = pool.scripted.actions(cur_obs_np[opp_idx])
                elif opp_kind == "net":
                    with torch.no_grad():
                        opp_out = opp_net.act(obs[opp_idx])
                    actions[opp_idx] = build_actions(opp_out, opp_idx.size)

                next_obs_np, rewards, dones, info = env.step(actions)
                total_env_steps += num_slots

                buffer.add(
                    obs=obs,
                    raw_cont=act_out["raw_cont"],
                    binary_action=act_out["binary_action"],
                    slot_action=act_out["slot_action"],
                    logprob=act_out["logprob"],
                    reward=rewards,
                    done=dones,
                    value=act_out["value"],
                )

                episode_return += rewards
                if dones.any():
                    finished = np.nonzero(dones)[0]
                    learner_finished = finished[iter_learner[finished]]
                    recent_returns.extend(episode_return[learner_finished].tolist())
                    episode_return[finished] = 0.0
                    recent_matches += learner_finished.size
                    recent_wins += int(np.count_nonzero(info["won"][learner_finished]))
                    if opp_kind == "scripted":
                        bench_finished = finished[bench_slots[finished]]
                        scripted_matches += bench_finished.size
                        scripted_wins += int(np.count_nonzero(info["won"][bench_finished]))

                if args.obs_noise > 0.0:
                    next_obs_np = next_obs_np + opp_rng.normal(
                        0.0, args.obs_noise, next_obs_np.shape
                    ).astype(np.float32)
                cur_obs_np = next_obs_np
                obs = torch.as_tensor(stacker.push(next_obs_np, dones) if stacker else next_obs_np)

            with torch.no_grad():
                *_, last_value = collect_model.forward(obs)

            # Linearly anneal the entropy bonus from --entropy-coef-start down to
            # --entropy-coef-end over the run, so exploration is strong early on
            # but the policy is allowed to commit to a decisive strategy later
            # instead of the entropy bonus permanently rewarding randomness.
            entropy_frac = min(update / args.total_updates, 1.0)
            entropy_coef = args.entropy_coef_start + (args.entropy_coef_end - args.entropy_coef_start) * entropy_frac

            # Linearly anneal the learning rate toward 0 over the run (see
            # --anneal-lr). Applied to every optimizer param group in place.
            lr_now = args.lr * (1.0 - entropy_frac) if args.anneal_lr else args.lr
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            # One bulk transfer of the whole filled rollout to the training
            # device (a no-op if it's already CPU) instead of the per-step
            # transfers `collect_model`/`buffer` above were built to avoid.
            device_buffer = buffer.to(device)
            last_value = last_value.to(device)

            # The rollout above is latency-bound and runs on `rollout_threads`
            # (1 by default). The GAE + PPO update is a big batched matmul the
            # sim sits idle through, so give it most of the cores, then switch
            # back for the next rollout. (No-op on GPU.)
            if cpu_threaded_update:
                torch.set_num_threads(update_threads)
            try:
                advantages, returns = device_buffer.compute_gae(last_value, args.gamma, args.gae_lambda)
                stats = ppo_update(
                    model,
                    optimizer,
                    device_buffer,
                    advantages,
                    returns,
                    epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    entropy_coef=entropy_coef,
                    clip_ratio=args.clip_ratio,
                    value_coef=args.value_coef,
                    max_grad_norm=args.max_grad_norm,
                    sample_mask=iter_mask_t,
                )
            finally:
                if cpu_threaded_update:
                    torch.set_num_threads(rollout_threads)

            if collect_model is not model:
                collect_model.load_state_dict(model.state_dict())

            if args.opponent_snapshot_every > 0 and update % args.opponent_snapshot_every == 0:
                pool.add_snapshot(model)
                log.info("added policy snapshot to opponent pool (update=%d, pool size=%d)", update, len(pool))

            if not np.isfinite(stats["policy_loss"]) or not np.isfinite(stats["value_loss"]):
                log.error(
                    "non-finite loss detected at update=%d (policy_loss=%s, value_loss=%s) - "
                    "training has likely diverged; stopping so you can inspect the last checkpoint",
                    update,
                    stats["policy_loss"],
                    stats["value_loss"],
                )
                break

            last_update_completed = update

            if update % args.log_every == 0:
                elapsed = time.time() - start_time
                sps = total_env_steps / elapsed
                avg_return = np.mean(recent_returns) if recent_returns else float("nan")
                win_rate = recent_wins / recent_matches if recent_matches else float("nan")
                vs_scripted = scripted_wins / scripted_matches if scripted_matches else float("nan")
                log.info(
                    "update=%d env_steps=%d steps/sec=%.0f avg_return=%.2f "
                    "win_rate=%.2f win_vs_scripted=%.2f (n=%d) "
                    "policy_loss=%.4f value_loss=%.4f entropy=%.4f approx_kl=%.4f clip_frac=%.2f "
                    "entropy_coef=%.5f lr=%.2e",
                    update,
                    total_env_steps,
                    sps,
                    avg_return,
                    win_rate,
                    vs_scripted,
                    scripted_matches,
                    stats["policy_loss"],
                    stats["value_loss"],
                    stats["entropy"],
                    stats["approx_kl"],
                    stats["clip_frac"],
                    entropy_coef,
                    lr_now,
                )
                if metrics.active:
                    metrics.log(update, {
                        "env_steps": total_env_steps,
                        "steps_per_sec": sps,
                        "avg_return": avg_return,
                        "win_rate": win_rate,
                        "win_vs_scripted": vs_scripted,
                        "vs_scripted_matches": scripted_matches,
                        "policy_loss": stats["policy_loss"],
                        "value_loss": stats["value_loss"],
                        "entropy": stats["entropy"],
                        "approx_kl": stats["approx_kl"],
                        "clip_frac": stats["clip_frac"],
                        "entropy_coef": entropy_coef,
                        "lr": lr_now,
                    })

                recent_returns.clear()
                recent_wins = 0
                recent_matches = 0
                scripted_wins = 0
                scripted_matches = 0

            if update % args.checkpoint_every == 0:
                _save_checkpoint(model, optimizer, update, args.checkpoint_dir, env, args.keep_checkpoints)
                last_saved_update = update

    except KeyboardInterrupt:
        log.warning("interrupted by user, shutting down cleanly")
    except (ConnectionError, FileNotFoundError) as e:
        # A user-fixable environment problem (sim binary missing, port taken,
        # sim died on startup) - the message says what to do; a Python
        # traceback would just be noise.
        log.error("could not start training: %s", e)
        raise
    except Exception:
        log.exception("training loop crashed with an unhandled exception")
        raise
    finally:
        # Persist whatever progress was made, on *any* exit path - a normal
        # finish, Ctrl+C, or a crash. Without this a Ctrl+C before the first
        # numbered snapshot would lose the run, and the README promises Ctrl+C
        # is always safe. Ignore SIGINT for this critical section so an
        # impatient second Ctrl+C can't skip the save or orphan the sim.
        try:
            prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            prev_sigint = None  # not the main thread (shouldn't happen here)
        try:
            if model is not None and last_update_completed > last_saved_update:
                log.info("saving checkpoint at update=%d before exit", last_update_completed)
                try:
                    _save_checkpoint(
                        model, optimizer, last_update_completed, args.checkpoint_dir, env,
                        args.keep_checkpoints,
                    )
                except Exception:
                    log.exception("could not save the exit checkpoint")
            if env is not None:
                env.close()
            metrics.close()
            if sim_config_path and os.path.isfile(sim_config_path):
                os.unlink(sim_config_path)
        finally:
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)


def _numbered_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """`(update, path)` for every `policy_update_<n>.pt` in the dir, sorted
    ascending by update number (unparseable names are skipped)."""
    import glob
    import re

    out = []
    for path in glob.glob(os.path.join(checkpoint_dir, "policy_update_*.pt")):
        m = re.search(r"policy_update_(\d+)\.pt$", os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def _wipe_checkpoints(checkpoint_dir: str) -> int:
    """Deletes `latest.pt` and every numbered snapshot in `checkpoint_dir`.
    Returns the number of files removed. Used by `--fresh`."""
    import glob

    removed = 0
    paths = [p for _, p in _numbered_checkpoints(checkpoint_dir)]
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.isfile(latest):
        paths.append(latest)
    # The opponent-league snapshots are part of the run state too.
    paths += glob.glob(os.path.join(checkpoint_dir, "league", "snapshot_*.pt"))
    for path in paths:
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            log.warning("could not delete %s: %s", path, e)
    return removed


def _save_checkpoint(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    checkpoint_dir: str,
    env: SelfPlayArenaEnv | None = None,
    keep_checkpoints: int = 3,
) -> None:
    """Writes `latest.pt` (the auto-resume point) and the numbered
    `policy_update_<update>.pt` history file, then prunes old snapshots.
    Called every `--checkpoint-every` updates and once more on any exit."""
    # The policy's real input width (== base feature width x --frame-stack).
    policy_obs_dim = model.trunk[0].in_features
    frame_stack = max(1, round(policy_obs_dim / features.OBS_DIM))
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "obs_dim": policy_obs_dim,
        "frame_stack": frame_stack,
        # Enough to rebuild the exact policy for export/inference without
        # re-deriving anything: trunk shape, and the sim-side normalization
        # constants this run trained against.
        "arch": {
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "slot_dim": model.slot_dim,
        },
        "sim_constants": features.active_constants().__dict__,
        "sim_config": getattr(env, "sim_config", None),
    }
    ckpt_path = os.path.join(checkpoint_dir, f"policy_update_{update}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(payload, ckpt_path)
    # Write `latest.pt` via a temp file + rename so it's never a half-written
    # file if the process dies mid-save (rename is atomic on the same fs).
    tmp_path = latest_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, latest_path)
    log.info("saved checkpoint: %s", ckpt_path)

    # Prune the oldest numbered snapshots so a long run doesn't slowly fill
    # the disk. `latest.pt` is separate and always kept.
    if keep_checkpoints > 0:
        snapshots = _numbered_checkpoints(checkpoint_dir)
        for _, old_path in snapshots[:-keep_checkpoints]:
            try:
                os.remove(old_path)
                log.info("pruned old checkpoint: %s", old_path)
            except OSError as e:
                log.warning("could not prune %s: %s", old_path, e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # A Ctrl+C that landed outside main()'s own handler (e.g. right at
        # startup, or a second one during shutdown) - main() has already run
        # its finally block, so just exit quietly without a traceback.
        raise SystemExit(130)
    except (ConnectionError, FileNotFoundError):
        # Already logged with a clear, actionable message by main().
        raise SystemExit(1)
