"""Fake-enemy aim check: does a checkpoint actually turn toward an enemy?

The sim's win rate can't tell a policy that faces and tracks its opponent
from one that circle-strafes and lucks into hits (a policy like that beat the
scripted bot ~86% of the time in the sim and still walked in circles live).
This puts a single enemy at a ring of known bearings in front of a stationary
bot and asks the deterministic policy what it wants to do, with no sim or
server involved:

  * **turn** - the yaw delta should have the sign of the turn that faces the
    enemy (`atan2(-rel_x, rel_z)`, the same convention as `ScriptedOpponent`);
    off-axis bearings only, where the answer is unambiguous;
  * **engage** - with the enemy dead ahead inside melee reach, it should
    attack; with it far away it should not need to.

Usage:
    python probe_aim.py                       # ../checkpoints/latest.pt
    python probe_aim.py --checkpoint policy_update_400.pt --sample
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

import features
from device import resolve_device
from evaluate import load_policy, policy_actions, resolve_checkpoint

# Bearings (degrees, 0 = dead ahead, +90 = to the bot's right) and distances
# (blocks) the probe places the enemy at.
BEARINGS = (0, 30, 60, 90, 135, 180, -135, -90, -60, -30)
DISTANCES = (2.0, 5.0)
# An off-axis bearing counts toward the turn check only above this many
# radians of required turn, so near-zero requirements can't fail on noise.
MIN_REQUIRED_TURN = 0.3
COLS = ("move_x", "move_z", "yaw_delta", "pitch_delta", "jump", "attack", "sprint")


def fake_observation(rel_x: float, rel_z: float, rel_y: float = 0.0) -> dict:
    """A healthy sword-kit bot on flat ground with one full-HP enemy at
    `(rel_x, rel_y, rel_z)` in the bot's own (yaw-rotated) frame."""
    sword = 1  # kit::Item::Sword
    return {
        "self_hp": 20.0, "self_vel_x": 0.0, "self_vel_y": 0.0, "self_vel_z": 0.0,
        "self_yaw": 0.0, "self_pitch": 0.0, "self_on_ground": True,
        "self_attack_cooldown": 1.0, "self_ping_ms": 50.0,
        "self_dist_from_center": 0.3, "self_held": sword, "self_slot": 0, "self_food": 20.0,
        "inventory": [1] + [0] * (features.ITEM_COUNT - 2),
        "hotbar": [sword] + [0] * (features.HOTBAR_SLOTS - 1),
        "time_left": 80.0,
        "enemies": [{"present": True, "hp": 20.0, "rel_x": rel_x, "rel_y": rel_y, "rel_z": rel_z,
                     "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0}],
        "enemies_alive": 1.0,
    }


def required_turn(rel_x: float, rel_z: float) -> float:
    """Yaw change (radians) that faces the enemy - +yaw turns toward -x."""
    return math.atan2(-rel_x, rel_z)


def probe(model, sample: bool = False) -> list[dict]:
    """One result row per (distance, bearing): the enemy's position, the turn
    needed to face it, and the policy's action."""
    cases, rows = [], []
    for dist in DISTANCES:
        for deg in BEARINGS:
            b = math.radians(deg)
            rel_x, rel_z = dist * math.sin(b), dist * math.cos(b)
            cases.append((dist, deg, rel_x, rel_z))
            rows.append(features.observation_to_row(fake_observation(rel_x, rel_z)))
    obs = torch.tensor(np.asarray(rows, dtype=np.float32))
    actions, _ = policy_actions(model, obs, sample)
    out = []
    for (dist, deg, rel_x, rel_z), act in zip(cases, actions):
        row = {"dist": dist, "bearing": deg, "needed_yaw": required_turn(rel_x, rel_z)}
        row.update({name: float(act[i]) for i, name in enumerate(COLS)})
        out.append(row)
    return out


def summarize(results: list[dict]) -> dict:
    off_axis = [r for r in results if abs(r["needed_yaw"]) >= MIN_REQUIRED_TURN]
    right = [r for r in off_axis if r["yaw_delta"] * r["needed_yaw"] > 0]
    ahead_close = [r for r in results if r["bearing"] == 0 and r["dist"] == min(DISTANCES)]
    return {
        "turn_correct": len(right),
        "turn_cases": len(off_axis),
        "attacks_ahead_close": all(r["attack"] > 0.5 for r in ahead_close),
        "mean_yaw": float(np.mean([r["yaw_delta"] for r in results])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="latest.pt", help="path or name in --checkpoint-dir")
    ap.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints"))
    ap.add_argument("--sample", action="store_true", help="sample actions instead of the deterministic mode")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    path = resolve_checkpoint(args.checkpoint, os.path.abspath(args.checkpoint_dir))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    features.configure(**ckpt.get("sim_constants", {}))
    frame_stack = ckpt.get("frame_stack", 1)
    if frame_stack != 1:
        raise SystemExit("probe_aim.py needs a frame_stack=1 checkpoint (it feeds one fake observation)")
    if ckpt.get("arch", {}).get("lstm_hidden", 0):
        raise SystemExit("probe_aim.py doesn't support --lstm checkpoints (no carried state across fake ticks)")
    model = load_policy(path, resolve_device(args.device), frame_stack)

    print(f"{os.path.basename(path)} (update {ckpt.get('update')}, max_look_delta "
          f"{features.active_constants().max_look_delta}) - {'sampled' if args.sample else 'deterministic'}")
    print(f"{'dist':>5} {'bearing':>8} {'needed':>8} | " + " ".join(f"{c:>10}" for c in COLS))
    results = probe(model, args.sample)
    for r in results:
        cols = " ".join(f"{r[c]:>+10.2f}" if c not in ("jump", "attack", "sprint") else f"{int(r[c] > 0.5):>10d}" for c in COLS)
        print(f"{r['dist']:>5.1f} {r['bearing']:>+8d} {r['needed_yaw']:>+8.2f} | {cols}")

    s = summarize(results)
    print()
    print(f"turns toward the enemy: {s['turn_correct']}/{s['turn_cases']} off-axis cases "
          f"({100 * s['turn_correct'] / max(s['turn_cases'], 1):.0f}%; chance is ~50%, a real tracker is ~100%)")
    print(f"attacks with the enemy dead ahead inside reach: {'yes' if s['attacks_ahead_close'] else 'NO'}")
    print(f"mean yaw over all bearings: {s['mean_yaw']:+.3f} rad/tick (far from 0 = a constant spin/circle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
