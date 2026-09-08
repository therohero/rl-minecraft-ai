"""Opponent pool + a scripted heuristic bot for self-play training.

Plain shared-policy self-play only ever faces a *live copy of itself*, so it
converges to whatever beats that exact mirror - frequently something a human
reads instantly (a fixed strafe rhythm, always-crit spacing, predictable
shield habits), because self-play never trains against being exploited.

This module widens the opponent distribution:

  * `OpponentPool` keeps a rolling set of frozen past snapshots of the
    policy (league-style), and
  * `ScriptedOpponent` is a fixed, never-trained heuristic bot.

Each PPO iteration `train.py` may hand one team in a fraction of the arenas
to an opponent drawn from here; those slots' transitions are masked out of
the PPO update (see `ppo_agent.ppo_update(sample_mask=...)`). Win-rate
against the scripted bot is a real, non-circular strength signal - unlike
self-play win-rate, which is ~50% by construction.

The scripted bot and the snapshot nets both consume the *processed*
observation matrix (`features.observation`/`wire_batch_to_obs` output) and
return raw action rows in `sim/src/protocol.rs::Action` wire order, exactly
what `env.step` expects.
"""

from __future__ import annotations

import numpy as np
import torch

import features


def enemy0_offset() -> int:
    """Index of the nearest-enemy block in a processed observation row. The
    self block expands `self_yaw`/`self_pitch` to sin/cos pairs (+2 over the
    wire width), then come the inventory-count and hotbar-layout blocks."""
    return (
        (features.OBS_SELF_FLOATS + 2)
        + features.OBS_INVENTORY_FLOATS
        + features.OBS_HOTBAR_FLOATS
    )


class ScriptedOpponent:
    """Face the nearest enemy, close the gap, strafe, and swing when it is in
    reach with the attack recharged. Deliberately mediocre and *fixed* - a
    yardstick, not a target. Stateless except for a step counter that drives
    the strafe oscillation.
    """

    # Processed-obs indices that do not move with the team-observation config
    # (they are inside the fixed leading self block).
    _PITCH_SIN, _PITCH_COS, _ATK_CD = 6, 7, 9
    # Cap the turn per tick to a human-ish rate so the bot can be out-strafed
    # - the point is a beatable, non-adaptive curriculum opponent, not an
    # aimbot yardstick.
    _MAX_TURN = 0.6

    def __init__(self) -> None:
        self._t = 0

    def actions(self, obs: np.ndarray) -> np.ndarray:
        """`obs`: `[n, obs_dim]` processed observations. Returns
        `[n, ACTION_FLOATS_PER_SLOT]` float32 action rows (look deltas already
        in radians and clamped, like `env.step` expects)."""
        obs = np.asarray(obs, dtype=np.float32)
        n = obs.shape[0]
        self._t += 1

        e = enemy0_offset()
        present = obs[:, e] > 0.5
        rel_x, rel_y, rel_z = obs[:, e + 2], obs[:, e + 3], obs[:, e + 4]
        horiz = np.hypot(rel_x, rel_z)
        self_pitch = np.arctan2(obs[:, self._PITCH_SIN], obs[:, self._PITCH_COS])
        atk_cd = obs[:, self._ATK_CD]
        max_look = float(features.active_constants().max_look_delta)

        # Turn to face: yaw 0 faces +z and +yaw turns toward -x, while rel_x
        # is the rightward offset - so the yaw change toward the enemy is
        # atan2(-rel_x, rel_z) (atan2 handles an enemy that's behind).
        want_yaw = np.arctan2(-rel_x, rel_z)
        want_pitch = np.arctan2(-rel_y, np.maximum(horiz, 0.1))
        turn_cap = min(self._MAX_TURN, max_look)
        yaw_delta = np.clip(want_yaw, -turn_cap, turn_cap)
        pitch_delta = np.clip(want_pitch - self_pitch, -turn_cap, turn_cap)

        # Range management: rush in, hold at the edge of reach, back off if
        # jammed right on top of the target.
        move_z = np.where(horiz > 2.6, 1.0, np.where(horiz < 1.7, -0.5, 0.15))
        strafe = 0.55 * np.sign(np.sin(0.06 * self._t + np.arange(n) * 2.399))
        move_x = np.where(horiz < 6.0, strafe, 0.0)

        attack = present & (horiz < 3.0) & (np.abs(rel_x) < 1.1) & (atk_cd > 0.85)
        sprint = present & (horiz > 3.5) & (move_z > 0.5)
        jump = (np.sin(0.9 * self._t + np.arange(n)) > 0.985) & present  # rare dodge hop

        # Wander when nothing is in view.
        no_enemy = ~present
        move_z = np.where(no_enemy, 0.25, move_z)
        move_x = np.where(no_enemy, 0.0, move_x)
        yaw_delta = np.where(no_enemy, 0.15, yaw_delta)
        pitch_delta = np.where(no_enemy, -self_pitch * 0.1, pitch_delta)

        rows = np.zeros((n, features.ACTION_FLOATS_PER_SLOT), dtype=np.float32)
        rows[:, 0] = move_x
        rows[:, 1] = move_z
        rows[:, 2] = yaw_delta
        rows[:, 3] = pitch_delta
        rows[:, 4] = jump
        rows[:, 5] = attack
        rows[:, 6] = sprint
        # use_item (7), sneak (8), held_slot (9) stay 0 - keep the default weapon.
        return rows


class OpponentPool:
    """A capped, rolling pool of frozen policy snapshots plus one shared
    `ScriptedOpponent`. Cheap: a snapshot is just CPU state-dict tensors, and
    only one extra `ActorCritic` is ever instantiated (reused across draws).
    """

    def __init__(self, make_model, capacity: int) -> None:
        """`make_model`: `() -> ActorCritic` (on CPU, eval-mode is set here)."""
        self._make_model = make_model
        self.capacity = max(int(capacity), 0)
        self._snapshots: list[dict] = []
        self._infer = None
        self.scripted = ScriptedOpponent()

    def __len__(self) -> int:
        return len(self._snapshots)

    def add_snapshot(self, model: torch.nn.Module) -> None:
        if self.capacity == 0:
            return
        sd = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        self._snapshots.append(sd)
        if len(self._snapshots) > self.capacity:
            self._snapshots.pop(0)

    def choose(self, rng: np.random.Generator, scripted_prob: float):
        """Pick this iteration's opponent.

        Returns `("scripted", None)`, `("net", ActorCritic)`, or - only when
        the pool is still empty and the scripted bot wasn't drawn -
        `("none", None)` (that iteration is plain self-play).
        """
        if rng.random() < scripted_prob:
            return ("scripted", None)
        if not self._snapshots:
            return ("none", None)
        sd = self._snapshots[int(rng.integers(len(self._snapshots)))]
        if self._infer is None:
            self._infer = self._make_model()
            self._infer.eval()
        self._infer.load_state_dict(sd)
        return ("net", self._infer)


def opponent_slot_mask(num_arenas: int, players_per_arena: int, opponent_fraction: float) -> np.ndarray:
    """Boolean `[num_slots]` mask, `True` for slots the *learner* controls.

    Team B (the second `players_per_arena // 2` slots) of the first
    `round(num_arenas * opponent_fraction)` arenas is handed to the opponent;
    every other slot stays with the learner.
    """
    num_slots = num_arenas * players_per_arena
    team_size = players_per_arena // 2
    learner = np.ones(num_slots, dtype=bool)
    n_opp = int(round(num_arenas * float(opponent_fraction)))
    for a in range(min(n_opp, num_arenas)):
        base = a * players_per_arena
        learner[base + team_size : base + players_per_arena] = False
    return learner


def benchmark_slot_mask(num_arenas: int, players_per_arena: int, opponent_fraction: float) -> np.ndarray:
    """Boolean `[num_slots]` mask, `True` for the *learner* team-A slots of the
    opponent arenas - the slots whose win/loss is a clean strength signal
    when the opponent is the scripted bot."""
    num_slots = num_arenas * players_per_arena
    team_size = players_per_arena // 2
    bench = np.zeros(num_slots, dtype=bool)
    n_opp = int(round(num_arenas * float(opponent_fraction)))
    for a in range(min(n_opp, num_arenas)):
        base = a * players_per_arena
        bench[base : base + team_size] = True
    return bench
