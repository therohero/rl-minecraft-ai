"""Observation frame stacking for a memoryless policy in a POMDP.

The sim gives each slot a latency-delayed, occlusion-limited view of the
other players, so a single frame doesn't carry enough to act optimally
(closing speed, whether an enemy is winding up, where an out-of-view
projectile went). Concatenating the last `n` observations is the cheapest
way to hand the same MLP trunk a short history - no recurrent state, no
change to the PPO update, just a wider input layer.

`FrameStacker` keeps an `[n, num_slots, base_dim]` ring (newest first) and
returns the flattened `[num_slots, n * base_dim]` stack. A slot's history
is zeroed the step its episode ends: the sim resets that arena in place,
so the observation that arrives alongside `done=True` already belongs to
the next episode and must not be stacked on top of the old one.
"""

from __future__ import annotations

import numpy as np


class FrameStacker:
    def __init__(self, num_slots: int, base_dim: int, n: int) -> None:
        if n < 1:
            raise ValueError(f"frame stack n must be >= 1, got {n}")
        self.num_slots = num_slots
        self.base_dim = base_dim
        self.n = n
        self._frames = np.zeros((n, num_slots, base_dim), dtype=np.float32)

    @property
    def stacked_dim(self) -> int:
        return self.n * self.base_dim

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Start fresh: history cleared, `obs` in the newest slot."""
        self._frames[:] = 0.0
        self._frames[0] = obs
        return self._stacked()

    def push(self, obs: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Shift `obs` in as the newest frame; clear history for any slot
        whose episode just ended (`dones[i]` true)."""
        self._frames[1:] = self._frames[:-1]
        self._frames[0] = obs
        if dones.any():
            self._frames[1:, np.asarray(dones, dtype=bool)] = 0.0
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        # [n, N, base] -> [N, n, base] -> [N, n*base], newest block first.
        return np.ascontiguousarray(self._frames.transpose(1, 0, 2).reshape(self.num_slots, -1))
