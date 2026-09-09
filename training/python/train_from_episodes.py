"""Offline fine-tune of the policy on episodes the Fabric mod recorded live.

`/fight train` in `mod/` writes one JSONL file per fight to
`<dataset dir>/<kit>/session-*.jsonl`, one line per game tick:
`{t, kit, target, obs, action}` (raw observation dict + the action the policy
took), then a trailing outcome record `{t, outcome, self_hp_end,
enemy_hp_end, reason}` with the real win/loss. This script closes the loop:

  1. rebuilds each tick's feature vector with `features.observation_to_row`
     (the exact decode the inference server uses),
  2. reconstructs a per-tick reward from the observation stream, mirroring
     `training/sim/src/arena.rs::reward` - damage dealt minus damage taken,
     plus a terminal win/loss bonus from the logged outcome record (or, for
     older datasets without one, guessed from the observation stream),
  3. computes GAE advantages using the current checkpoint's value head,
  4. runs a few epochs of advantage-weighted regression (AWR): weighted
     behavioural cloning toward the actions that did well, on real-server
     observations - a gentle domain-adaptation nudge,
  5. writes the updated weights back to `training/checkpoints/latest.pt`
     (the previous file is kept as `latest.pt.pre_finetune`).

Real-server data is scarce and off the sim's training distribution, so this
is a nudge, not a replacement for `./run.sh` self-play. `--dry-run` prints
the episode / return stats and touches nothing.

Usage:
    python train_from_episodes.py --episodes <dataset dir> [--dry-run]
    python train_from_episodes.py --episodes ../datasets --epochs 6 --beta 0.5
"""

import argparse
import glob
import json
import math
import os
import shutil

import numpy as np
import torch

import features
from logging_setup import get_logger
from ppo_agent import ActorCritic

log = get_logger(__name__)


def _find_sessions(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root]
    hits = sorted(glob.glob(os.path.join(root, "**", "session-*.jsonl"), recursive=True))
    hits += sorted(p for p in glob.glob(os.path.join(root, "*.jsonl")) if p not in hits)
    return hits


def _atanh_clip(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -0.999, 0.999))


def _load_session(path: str, scale: float):
    """One session file -> (obs[T,D], raw_cont[T,4], binary[T,5], slot[T],
    reward[T], done[T]) as numpy, or None if too short."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # The mod writes a trailing outcome record `{t, outcome, self_hp_end,
    # enemy_hp_end, reason}` (no `obs` key). Older datasets don't have it -
    # fall back to guessing the result from the observation stream.
    outcome_rec = None
    if rows and "obs" not in rows[-1]:
        outcome_rec = rows.pop()
    if len(rows) < 4:
        return None

    obs_vecs, raw_cont, binary, slot = [], [], [], []
    self_hp, enemy_hp, enemy_present = [], [], []
    for r in rows:
        o = r["obs"]
        obs_vecs.append(features.observation_to_row(o))
        a = r["action"]
        sq = [
            a.get("moveX", a.get("move_x", 0.0)),
            a.get("moveZ", a.get("move_z", 0.0)),
            a.get("yawDelta", a.get("yaw_delta", 0.0)) / scale,
            a.get("pitchDelta", a.get("pitch_delta", 0.0)) / scale,
        ]
        raw_cont.append(_atanh_clip(np.asarray(sq, dtype=np.float32)))
        binary.append([
            float(bool(a.get("jump"))), float(bool(a.get("attack"))),
            float(bool(a.get("sprint"))), float(bool(a.get("useItem", a.get("use_item")))),
            float(bool(a.get("sneak"))),
        ])
        slot.append(int(a.get("heldSlot", a.get("held_slot", 0)) or 0))

        self_hp.append(float(o.get("self_hp", 20.0)))
        enemies = o.get("enemies") or []
        e0 = enemies[0] if enemies else None
        present = bool(e0 and float(e0.get("present", 1.0)) > 0.5)
        enemy_present.append(present)
        enemy_hp.append(float(e0.get("hp", 0.0)) if present else math.nan)

    T = len(rows)
    reward = np.zeros(T, dtype=np.float32)
    for i in range(T - 1):
        taken = max(0.0, self_hp[i] - self_hp[i + 1])
        dealt = 0.0
        if enemy_present[i] and enemy_present[i + 1]:
            dealt = max(0.0, enemy_hp[i] - enemy_hp[i + 1])
        reward[i] = dealt * ARGS.per_hp_dealt - taken * ARGS.per_hp_taken

    done = np.zeros(T, dtype=np.float32)
    done[-1] = 1.0
    if outcome_rec is not None:
        # Real match result logged by the mod. Also fold in the final-tick HP
        # deltas (the per-tick loop above stops at T-1, so the last hit /
        # last hit taken would otherwise be dropped).
        she = outcome_rec.get("self_hp_end")
        if she is not None:
            reward[-1] -= max(0.0, self_hp[-1] - float(she)) * ARGS.per_hp_taken
        ehe = outcome_rec.get("enemy_hp_end")
        if ehe is not None and enemy_present[-1]:
            reward[-1] += max(0.0, enemy_hp[-1] - float(ehe)) * ARGS.per_hp_dealt
        oc = str(outcome_rec.get("outcome", "unknown"))
        if oc == "loss":
            reward[-1] -= ARGS.loss
        elif oc == "win":
            reward[-1] += ARGS.win
        # "unknown" (manual /fight stop, enemy ran off): no terminal bonus
    else:
        # legacy heuristic: enemy gone while we're alive = win; we're dead = loss.
        if self_hp[-1] <= 0.5:
            reward[-1] -= ARGS.loss
        elif enemy_present[0] and not enemy_present[-1]:
            reward[-1] += ARGS.win

    return (
        np.asarray(obs_vecs, dtype=np.float32),
        np.asarray(raw_cont, dtype=np.float32),
        np.asarray(binary, dtype=np.float32),
        np.asarray(slot, dtype=np.int64),
        reward,
        done,
    )


def _gae(reward, value, done, gamma, lam):
    T = len(reward)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        next_v = value[t + 1] if t + 1 < T else 0.0
        delta = reward[t] + gamma * next_v * nonterminal - value[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + value


def main():
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", required=True, help="dataset dir (or a single .jsonl file)")
    p.add_argument("--checkpoint", default="../checkpoints/latest.pt")
    p.add_argument("--out", default=None, help="where to write the updated checkpoint (default: overwrite --checkpoint)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=1.0, help="AWR temperature (higher = closer to plain BC)")
    p.add_argument("--weight-clip", type=float, default=20.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--per-hp-dealt", type=float, default=1.0)
    p.add_argument("--per-hp-taken", type=float, default=1.0)
    p.add_argument("--win", type=float, default=100.0)
    p.add_argument("--loss", type=float, default=100.0)
    p.add_argument("--dry-run", action="store_true", help="print stats, write nothing")
    ARGS = p.parse_args()

    if not os.path.isfile(ARGS.checkpoint):
        raise SystemExit(f"checkpoint not found: {ARGS.checkpoint} - train first with ./run.sh")
    sessions = _find_sessions(ARGS.episodes)
    if not sessions:
        raise SystemExit(f"no session-*.jsonl found under {ARGS.episodes} - record some with /fight train")

    ckpt = torch.load(ARGS.checkpoint, map_location="cpu", weights_only=False)
    features.configure(**ckpt.get("sim_constants", {}))
    scale = features.active_constants().max_look_delta
    arch = ckpt.get("arch", {})
    obs_dim = ckpt.get("obs_dim", features.OBS_DIM)
    model = ActorCritic(
        obs_dim,
        hidden_size=arch.get("hidden_size", 256),
        num_layers=arch.get("num_layers", 2),
        slot_dim=arch.get("slot_dim", features.HOTBAR_ACTION_DIM),
        lstm_hidden=arch.get("lstm_hidden", 0),
    )
    if model.recurrent:
        raise SystemExit(
            "this checkpoint was trained with --lstm; offline replay training here is stateless "
            "(it treats every logged tick independently) and can't fit a recurrent head. Use an "
            "MLP / frame-stacked checkpoint."
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_all, rc_all, bin_all, slot_all, adv_all, ret_all = [], [], [], [], [], []
    total_reward = 0.0
    kept = 0
    for path in sessions:
        loaded = _load_session(path, scale)
        if loaded is None:
            log.warning("skipping %s (too short)", os.path.basename(path))
            continue
        obs, rc, bn, sl, reward, done = loaded
        if obs.shape[1] != obs_dim:
            log.warning("skipping %s: obs dim %d != checkpoint %d (feature code changed?)",
                        os.path.basename(path), obs.shape[1], obs_dim)
            continue
        with torch.no_grad():
            _, _, _, _, value, _ = model.forward(torch.from_numpy(obs))
        adv, ret = _gae(reward, value.numpy(), done, ARGS.gamma, ARGS.gae_lambda)
        obs_all.append(obs); rc_all.append(rc); bin_all.append(bn); slot_all.append(sl)
        adv_all.append(adv); ret_all.append(ret)
        total_reward += float(reward.sum())
        kept += 1
        log.info("%-38s ticks=%4d  return=%8.2f", os.path.basename(path), len(reward), float(reward.sum()))

    if kept == 0:
        raise SystemExit("no usable sessions")

    obs = torch.from_numpy(np.concatenate(obs_all))
    raw_cont = torch.from_numpy(np.concatenate(rc_all))
    binary = torch.from_numpy(np.concatenate(bin_all))
    slot = torch.from_numpy(np.concatenate(slot_all)).clamp_(0, model.slot_dim - 1)
    adv = torch.from_numpy(np.concatenate(adv_all))
    ret = torch.from_numpy(np.concatenate(ret_all))
    N = obs.shape[0]

    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    weight = torch.exp((adv_n / ARGS.beta).clamp(-20.0, 20.0)).clamp(0.0, ARGS.weight_clip)

    log.info("episodes=%d  transitions=%d  mean episode return=%.2f  weight[min/mean/max]=%.2f/%.2f/%.2f",
             kept, N, total_reward / kept, float(weight.min()), float(weight.mean()), float(weight.max()))
    if ARGS.dry_run:
        log.info("--dry-run: not updating the checkpoint")
        return

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    for epoch in range(ARGS.epochs):
        perm = torch.randperm(N)
        pi_loss_sum = v_loss_sum = 0.0
        for start in range(0, N, ARGS.batch_size):
            idx = perm[start:start + ARGS.batch_size]
            logprob, _entropy, value, _ = model.evaluate(obs[idx], raw_cont[idx], binary[idx], slot[idx])
            pi_loss = -(weight[idx] * logprob).mean()
            v_loss = torch.nn.functional.mse_loss(value, ret[idx])
            loss = pi_loss + ARGS.value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pi_loss_sum += pi_loss.item() * len(idx)
            v_loss_sum += v_loss.item() * len(idx)
        log.info("epoch %d/%d  pi_loss=%.4f  value_loss=%.4f",
                 epoch + 1, ARGS.epochs, pi_loss_sum / N, v_loss_sum / N)

    out = ARGS.out or ARGS.checkpoint
    if out == ARGS.checkpoint and os.path.isfile(ARGS.checkpoint):
        backup = ARGS.checkpoint + ".pre_finetune"
        shutil.copy2(ARGS.checkpoint, backup)
        log.info("backed up previous checkpoint -> %s", backup)
    ckpt["model_state_dict"] = model.state_dict()
    ckpt["finetuned_from_episodes"] = ckpt.get("finetuned_from_episodes", 0) + kept
    tmp = out + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out)
    log.info("wrote updated checkpoint -> %s  (re-export with run_bot_mod.sh / export_model.py)", out)


if __name__ == "__main__":
    main()
