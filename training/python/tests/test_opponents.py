"""Tests for `opponents.py`: the scripted heuristic bot's action shape and
sign conventions, the snapshot pool, the slot masks, and `ppo_update`'s
`sample_mask` actually dropping the masked slots."""

import numpy as np
import torch

import features
from opponents import (
    OpponentPool,
    ScriptedOpponent,
    benchmark_slot_mask,
    enemy0_offset,
    opponent_slot_mask,
)
from ppo_agent import ActorCritic, RolloutBuffer, ppo_update


def _obs_with_enemy(rel_x, rel_y, rel_z, n=1):
    obs = np.zeros((n, features.OBS_DIM), dtype=np.float32)
    obs[:, 7] = 1.0  # cos(self_pitch) = 1 -> pitch 0
    obs[:, 9] = 1.0  # attack fully recharged
    e = enemy0_offset()
    obs[:, e] = 1.0  # enemy present
    obs[:, e + 2] = rel_x
    obs[:, e + 3] = rel_y
    obs[:, e + 4] = rel_z
    return obs


def test_scripted_action_row_shape_and_range():
    rows = ScriptedOpponent().actions(_obs_with_enemy(0.5, 0.0, 4.0, n=6))
    assert rows.shape == (6, features.ACTION_FLOATS_PER_SLOT)
    assert rows.dtype == np.float32
    max_look = features.active_constants().max_look_delta
    assert np.all(np.abs(rows[:, 2]) <= max_look + 1e-6)  # yaw_delta clamped
    assert np.all(np.abs(rows[:, 3]) <= max_look + 1e-6)  # pitch_delta clamped
    assert set(np.unique(rows[:, 4:7]).tolist()) <= {0.0, 1.0}  # jump/attack/sprint are flags


def test_scripted_turns_toward_the_enemy():
    bot = ScriptedOpponent()
    # enemy to the right (rel_x > 0) -> turn right = negative yaw_delta.
    right = bot.actions(_obs_with_enemy(3.0, 0.0, 1.0))
    assert right[0, 2] < 0
    # enemy to the left -> positive yaw_delta.
    left = bot.actions(_obs_with_enemy(-3.0, 0.0, 1.0))
    assert left[0, 2] > 0
    # enemy above (rel_y > 0) -> look up = negative pitch_delta.
    up = bot.actions(_obs_with_enemy(0.0, 3.0, 1.0))
    assert up[0, 3] < 0


def test_scripted_attacks_only_in_reach_and_aimed():
    bot = ScriptedOpponent()
    close = bot.actions(_obs_with_enemy(0.2, 0.0, 1.5))
    far = bot.actions(_obs_with_enemy(0.2, 0.0, 8.0))
    assert close[0, 5] == 1.0
    assert far[0, 5] == 0.0


def test_scripted_no_enemy_still_produces_finite_wander():
    rows = ScriptedOpponent().actions(np.zeros((4, features.OBS_DIM), dtype=np.float32))
    assert np.all(np.isfinite(rows))
    assert np.all(rows[:, 5] == 0.0)  # never attacks with no enemy


def test_slot_masks_partition_opponent_arenas():
    # 10 arenas, 1v1 (2 slots each), 30% -> 3 opponent arenas.
    learner = opponent_slot_mask(10, 2, 0.3)
    bench = benchmark_slot_mask(10, 2, 0.3)
    assert learner.shape == (20,)
    assert (~learner).sum() == 3  # one opponent (team-B) slot per opponent arena
    assert bench.sum() == 3  # one benchmark (team-A) slot per opponent arena
    # opponent arenas are the first 3: slots 1, 3, 5 are opponent; 0, 2, 4 bench.
    assert list(np.nonzero(~learner)[0]) == [1, 3, 5]
    assert list(np.nonzero(bench)[0]) == [0, 2, 4]
    # a 2v2: team B is the last 2 of each 4-slot arena.
    learner4 = opponent_slot_mask(4, 4, 0.5)  # 2 opponent arenas
    assert list(np.nonzero(~learner4)[0]) == [2, 3, 6, 7]


def test_pool_snapshot_and_choice():
    def make():
        return ActorCritic(obs_dim=features.OBS_DIM, hidden_size=8, num_layers=1)

    pool = OpponentPool(make_model=make, capacity=2)
    rng = np.random.default_rng(0)
    # empty pool, no scripted draw -> "none"
    assert pool.choose(rng, scripted_prob=0.0)[0] == "none"
    # always scripted
    assert pool.choose(rng, scripted_prob=1.0)[0] == "scripted"

    m = make()
    pool.add_snapshot(m)
    pool.add_snapshot(m)
    pool.add_snapshot(m)  # capacity 2 -> oldest dropped
    assert len(pool) == 2
    kind, net = pool.choose(rng, scripted_prob=0.0)
    assert kind == "net" and isinstance(net, ActorCritic)


def test_ppo_update_sample_mask_drops_masked_slots():
    torch.manual_seed(0)
    T, N, obs_dim = 4, 6, features.OBS_DIM
    model = ActorCritic(obs_dim=obs_dim, hidden_size=8, num_layers=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: weights can't move
    buf = RolloutBuffer(T, N, obs_dim, torch.device("cpu"))
    for _ in range(T):
        obs = torch.randn(N, obs_dim)
        out = model.act(obs)
        buf.add(
            obs=obs,
            raw_cont=out["raw_cont"],
            binary_action=out["binary_action"],
            slot_action=out["slot_action"],
            logprob=out["logprob"],
            reward=np.zeros(N, dtype=np.float32),
            done=np.zeros(N, dtype=bool),
            value=out["value"],
        )
    adv, ret = buf.compute_gae(torch.zeros(N), gamma=0.99, gae_lambda=0.95)

    mask = torch.tensor([True, True, False, False, True, False])
    stats_masked = ppo_update(model, opt, buf, adv, ret, epochs=1, minibatch_size=1024, sample_mask=mask)
    stats_full = ppo_update(model, opt, buf, adv, ret, epochs=1, minibatch_size=1024)
    # both finite, and the masked pass genuinely ran on fewer samples
    # (different loss than the full pass on the same buffer).
    assert np.isfinite(stats_masked["policy_loss"]) and np.isfinite(stats_full["policy_loss"])
    assert stats_masked["value_loss"] != stats_full["value_loss"]
