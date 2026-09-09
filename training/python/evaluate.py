"""Round-robin evaluation harness: rate a checkpoint against a ladder.

`win_vs_scripted` in the training log is the only strength signal that
isn't circular, and it's coarse (one opponent, noisy). This plays a
*candidate* checkpoint head-to-head against a ladder of past checkpoints
plus the fixed `ScriptedOpponent`, in the same Rust sim / kit / reward
config the candidate trained under, and fits a Bradley-Terry model to the
pairwise results to report a single Elo-scaled strength number per player.

Every pairing plays `--matches-per-pair` matches split evenly across the
two starting sides (spawns are randomised each match anyway, but swapping
A/B removes any residual first-slot bias). Matches run continuously across
`--num-arenas` parallel arenas - an arena picks up the next pending
pairing the instant its current match ends.

Usage:
    python evaluate.py                          # latest.pt vs every policy_update_*.pt + scripted
    python evaluate.py --candidate policy_update_800.pt
    python evaluate.py --matches-per-pair 400 --num-arenas 128
    python evaluate.py --sample                 # sample actions instead of the deterministic mean

Reads checkpoints from --checkpoint-dir (default ../checkpoints). The sim
config is taken from the candidate checkpoint so the ladder is judged
under the exact conditions it was trained for; override with --sim-config.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import torch

import features
from device import resolve_device
from env import DEFAULT_SIM_BINARY, SelfPlayArenaEnv
from frame_stack import FrameStacker
from logging_setup import get_logger
from opponents import ScriptedOpponent
from ppo_agent import ActorCritic
from train import _numbered_checkpoints, build_actions

log = get_logger(__name__)

_ELO_PER_LOGIT = 400.0 / np.log(10.0)


# --------------------------------------------------------------------------
# checkpoint discovery + loading
# --------------------------------------------------------------------------

def resolve_checkpoint(path_or_name: str, checkpoint_dir: str) -> str:
    """Accept either a full path or a bare filename inside `checkpoint_dir`."""
    if os.path.isfile(path_or_name):
        return os.path.abspath(path_or_name)
    cand = os.path.join(checkpoint_dir, path_or_name)
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    raise FileNotFoundError(f"checkpoint not found: {path_or_name} (also tried {cand})")


def discover_ladder(checkpoint_dir: str, candidate_path: str, max_ladder: int) -> list[str]:
    """Numbered `policy_update_*.pt` snapshots in `checkpoint_dir`, excluding
    the candidate itself, evenly subsampled to at most `max_ladder`."""
    numbered = [p for _, p in _numbered_checkpoints(checkpoint_dir)]
    numbered = [p for p in numbered if os.path.abspath(p) != os.path.abspath(candidate_path)]
    if max_ladder > 0 and len(numbered) > max_ladder:
        idx = np.linspace(0, len(numbered) - 1, max_ladder).round().astype(int)
        numbered = [numbered[i] for i in sorted(set(idx))]
    return numbered


def checkpoint_frame_stack(path: str) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("frame_stack", 1)


def load_policy(path: str, device: torch.device, frame_stack: int) -> ActorCritic:
    """Rebuild the exact `ActorCritic` a checkpoint was saved from, in eval
    mode. Raises if its observation space or frame-stack depth doesn't match
    the rest of the ladder (`frame_stack`)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", {})
    ckpt_fs = ckpt.get("frame_stack", 1)
    if ckpt_fs != frame_stack:
        raise ValueError(
            f"{os.path.basename(path)} trained with --frame-stack {ckpt_fs}, ladder is {frame_stack}"
        )
    base = ckpt.get("obs_dim", features.OBS_DIM) // ckpt_fs
    if base != features.OBS_DIM:
        raise ValueError(
            f"{os.path.basename(path)} trained with base obs_dim={base}, but this sim config "
            f"gives obs_dim={features.OBS_DIM} - it can't be judged on the same ladder"
        )
    model = ActorCritic(
        features.OBS_DIM * frame_stack,
        hidden_size=arch.get("hidden_size", 256),
        num_layers=arch.get("num_layers", 2),
        slot_dim=arch.get("slot_dim", features.HOTBAR_ACTION_DIM),
        lstm_hidden=arch.get("lstm_hidden", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# --------------------------------------------------------------------------
# action selection
# --------------------------------------------------------------------------

@torch.no_grad()
def policy_actions(model: ActorCritic, obs: torch.Tensor, sample: bool, hidden=None):
    """`([n, ACTION_FLOATS_PER_SLOT] actions, new_hidden)` for the given
    observation rows. `sample=False` takes the distribution's mode
    (tanh(mean), logit>0, argmax) for a crisp, reproducible strength
    measurement. `new_hidden` is `None` for the MLP head."""
    if sample:
        out = model.act(obs, hidden)
        new_hidden = out["hidden"]
    else:
        mean, _, binary_logits, slot_logits, _, new_hidden = model.forward(obs, hidden)
        out = {
            "squashed_cont": torch.tanh(mean),
            "binary_action": (binary_logits > 0).to(mean.dtype),
            "slot_action": slot_logits.argmax(dim=-1),
        }
    return build_actions(out, obs.shape[0]), new_hidden


# --------------------------------------------------------------------------
# Bradley-Terry rating
# --------------------------------------------------------------------------

def bradley_terry_elo(win_credit: np.ndarray, games: np.ndarray, prior: float = 1.0,
                      iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Fit Bradley-Terry strengths to a pairwise result matrix by the
    Zermelo/MM fixed-point iteration, then return them Elo-scaled (mean 0).

    `win_credit[i, j]` = i's score against j (1 per win, 0.5 per draw);
    `games[i, j]` = `games[j, i]` = total i-vs-j games. `prior` adds that
    many virtual even games against a phantom average-strength opponent so
    an unbeaten or winless player still gets a finite rating.
    """
    p = win_credit.shape[0]
    w = win_credit.sum(axis=1) + prior  # + half of `2*prior` virtual games won
    strength = np.ones(p)
    for _ in range(iters):
        denom = (games / (strength[:, None] + strength[None, :] + np.eye(p))).sum(axis=1)
        denom += 2.0 * prior / (strength + 1.0)
        nxt = w / denom
        nxt /= np.exp(np.mean(np.log(nxt)))  # anchor geometric mean to 1
        if np.max(np.abs(nxt - strength)) < tol:
            strength = nxt
            break
        strength = nxt
    return _ELO_PER_LOGIT * np.log(strength)


# --------------------------------------------------------------------------
# tournament
# --------------------------------------------------------------------------

def _build_queue(n_players: int, matches_per_pair: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """One `(team_A_player, team_B_player)` ticket per scheduled match, with
    each pair's matches split evenly between the two starting sides."""
    queue: list[tuple[int, int]] = []
    for i in range(n_players):
        for j in range(i + 1, n_players):
            for k in range(matches_per_pair):
                queue.append((i, j) if k % 2 == 0 else (j, i))
    rng.shuffle(queue)
    return queue


def run_tournament(env: SelfPlayArenaEnv, players: list[dict], matches_per_pair: int,
                   sample: bool, seed: int, frame_stack: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Play the full round-robin. Returns `(wins, draws)` integer matrices:
    `wins[i, j]` = clean wins of i over j, `draws[i, j] == draws[j, i]`."""
    n = len(players)
    team = env.team_size
    per_arena = env.players_per_arena
    rng = np.random.default_rng(seed)

    queue = _build_queue(n, matches_per_pair, rng)
    total = len(queue)
    wins = np.zeros((n, n))
    draws = np.zeros((n, n))
    scripted_bot = ScriptedOpponent()
    # Per-net-player carried LSTM state (`None` for MLP players), full slot
    # width and sliced per step by the slots that player currently controls;
    # a slot is zeroed the tick its match ends.
    hiddens = [
        p["model"].zero_hidden(env.num_slots) if p["kind"] == "net" else None
        for p in players
    ]

    # Per-arena current pairing; None once there's nothing left to play.
    pairing: list[tuple[int, int] | None] = [queue.pop() if queue else None for _ in range(env.num_arenas)]

    stacker = FrameStacker(env.num_slots, features.OBS_DIM, frame_stack) if frame_stack > 1 else None
    cur_obs_np = env.reset()
    obs = torch.as_tensor(stacker.reset(cur_obs_np) if stacker else cur_obs_np)
    done_matches = 0
    last_pct = -1
    while any(p is not None for p in pairing):
        actions = np.zeros((env.num_slots, features.ACTION_FLOATS_PER_SLOT), dtype=np.float32)

        # Gather, per player, every slot it controls this step, then run that
        # policy once on the whole batch.
        controller = np.full(env.num_slots, -1, dtype=int)
        for a, pair in enumerate(pairing):
            if pair is None:
                continue
            base = a * per_arena
            controller[base : base + team] = pair[0]
            controller[base + team : base + per_arena] = pair[1]
        for pid, player in enumerate(players):
            idx = np.nonzero(controller == pid)[0]
            if idx.size == 0:
                continue
            if player["kind"] == "scripted":
                actions[idx] = scripted_bot.actions(cur_obs_np[idx])
            else:
                h_in = None
                if hiddens[pid] is not None:
                    h_in = (hiddens[pid][0][:, idx], hiddens[pid][1][:, idx])
                acts, h_out = policy_actions(player["model"], obs[idx], sample, h_in)
                actions[idx] = acts
                if h_out is not None:
                    hiddens[pid][0][:, idx] = h_out[0]
                    hiddens[pid][1][:, idx] = h_out[1]

        next_obs, _, dones, info = env.step(actions)
        cur_obs_np = next_obs
        if dones.any():
            dmask = torch.as_tensor(np.asarray(dones, dtype=bool))
            for hp in hiddens:
                if hp is not None:
                    hp[0][:, dmask] = 0.0
                    hp[1][:, dmask] = 0.0

        for a, pair in enumerate(pairing):
            if pair is None:
                continue
            base = a * per_arena
            if not dones[base]:
                continue
            a_idx, b_idx = pair
            if info["won"][base]:
                wins[a_idx, b_idx] += 1
            elif info["lost"][base]:
                wins[b_idx, a_idx] += 1
            else:  # timeout with equal team HP
                draws[a_idx, b_idx] += 1
                draws[b_idx, a_idx] += 1
            done_matches += 1
            pairing[a] = queue.pop() if queue else None

        pct = int(100 * done_matches / total)
        if pct >= last_pct + 10:
            log.info("  %d%% (%d/%d matches)", pct, done_matches, total)
            last_pct = pct

        obs = torch.as_tensor(stacker.push(next_obs, dones) if stacker else next_obs)

    return wins, draws


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _sim_config_file(candidate_path: str, override: str | None,
                     match_time: float | None) -> tuple[str | None, bool]:
    """Path to the sim config to launch with, and whether it's a temp file
    we own (and must delete). Prefers --sim-config, else the config baked
    into the candidate checkpoint, else the sim's built-in defaults.
    `match_time`, if set, overrides `match_time_seconds` (a shorter match
    makes an eval sweep finish sooner)."""
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"--sim-config file not found: {override}")
        with open(override) as f:
            sim_config = json.load(f)
    else:
        ckpt = torch.load(candidate_path, map_location="cpu", weights_only=False)
        sim_config = ckpt.get("sim_config")
        if not sim_config:
            log.warning("candidate checkpoint has no sim_config - evaluating under the sim's default config")

    if not sim_config:
        return None, False
    if match_time is not None:
        sim_config = {**sim_config, "match_time_seconds": match_time}
    fd, path = tempfile.mkstemp(suffix=".json", prefix="eval-sim-config-")
    with os.fdopen(fd, "w") as f:
        json.dump(sim_config, f)
    return path, True


def _print_report(players: list[dict], wins: np.ndarray, draws: np.ndarray, elo: np.ndarray) -> None:
    n = len(players)
    games = wins + wins.T + draws
    credit = wins + 0.5 * draws  # score matrix
    order = np.argsort(-elo)
    name_w = max(len(p["label"]) for p in players)
    row = name_w + 42

    print("\n" + "=" * row)
    print(f"{'player':<{name_w}}   {'elo':>6}   {'W':>5} {'L':>5} {'D':>5}   {'score%':>7}")
    print("-" * row)
    for pid in order:
        g = games[pid].sum()
        w = int(wins[pid].sum())
        loss = int(wins[:, pid].sum())
        d = int(draws[pid].sum())
        score = 100.0 * credit[pid].sum() / g if g else float("nan")
        tag = "  <- candidate" if players[pid]["candidate"] else ""
        print(f"{players[pid]['label']:<{name_w}}   {int(round(elo[pid])):>6d}   "
              f"{w:>5} {loss:>5} {d:>5}   {score:7.1f}{tag}")
    print("=" * row)

    cand = next(i for i, p in enumerate(players) if p["candidate"])
    g = games[cand].sum()
    print(f"\ncandidate '{players[cand]['label']}': elo {int(round(elo[cand]))}, "
          f"{100.0 * credit[cand].sum() / g:.1f}% score vs the field ({int(g)} matches)")
    for j in order:
        if j == cand or not games[cand, j]:
            continue
        print(f"    vs {players[j]['label']:<26} {100.0 * credit[cand, j] / games[cand, j]:5.1f}%  ({int(games[cand, j])})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", default="../checkpoints")
    ap.add_argument("--candidate", default="latest.pt", help="checkpoint under test (path or name in --checkpoint-dir)")
    ap.add_argument("--sim-binary", default=DEFAULT_SIM_BINARY)
    ap.add_argument("--sim-config", default=None, help="override the sim config (default: the candidate's own)")
    ap.add_argument("--match-time", type=float, default=None,
                    help="override match_time_seconds (shorter = faster sweep; default: the training value)")
    ap.add_argument("--matches-per-pair", type=int, default=100)
    ap.add_argument("--max-ladder", type=int, default=8, help="cap on past-checkpoint opponents (0 = all); evenly subsampled")
    ap.add_argument("--num-arenas", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample actions instead of taking the deterministic mode")
    ap.add_argument("--no-scripted", action="store_true", help="leave the scripted bot out of the ladder")
    ap.add_argument("--device", default="cpu", help="torch device for the policies (default cpu - the nets are tiny)")
    args = ap.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    candidate_path = resolve_checkpoint(args.candidate, checkpoint_dir)
    ladder_paths = discover_ladder(checkpoint_dir, candidate_path, args.max_ladder)
    device = resolve_device(args.device)
    frame_stack = checkpoint_frame_stack(candidate_path)

    sim_config_path, owns_config = _sim_config_file(candidate_path, args.sim_config, args.match_time)

    env = None
    try:
        env = SelfPlayArenaEnv(
            num_arenas=args.num_arenas,
            sim_binary=args.sim_binary,
            sim_config_path=sim_config_path,
            seed=args.seed,
        )

        players: list[dict] = []
        players.append({"label": f"candidate:{os.path.basename(candidate_path)}", "kind": "net",
                        "model": load_policy(candidate_path, device, frame_stack), "candidate": True})
        for path in ladder_paths:
            try:
                players.append({"label": os.path.basename(path), "kind": "net",
                                "model": load_policy(path, device, frame_stack), "candidate": False})
            except ValueError as e:
                log.warning("skipping ladder checkpoint: %s", e)
        if not args.no_scripted:
            players.append({"label": "scripted", "kind": "scripted", "model": None, "candidate": False})

        if len(players) < 2:
            log.error("need at least one opponent - no usable ladder checkpoints and --no-scripted given")
            return 2

        n_pairs = len(players) * (len(players) - 1) // 2
        log.info("evaluating %d players (%d pairs x %d matches = %d) on %d arenas, %s actions, frame_stack=%d",
                 len(players), n_pairs, args.matches_per_pair, n_pairs * args.matches_per_pair,
                 args.num_arenas, "sampled" if args.sample else "deterministic", frame_stack)

        wins, draws = run_tournament(env, players, args.matches_per_pair, args.sample, args.seed, frame_stack)
        games = wins + wins.T + draws
        elo = bradley_terry_elo(wins + 0.5 * draws, games)
        _print_report(players, wins, draws, elo)
        return 0
    finally:
        if env is not None:
            env.close()
        if owns_config and sim_config_path and os.path.isfile(sim_config_path):
            os.unlink(sim_config_path)


if __name__ == "__main__":
    raise SystemExit(main())
