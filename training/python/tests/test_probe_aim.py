"""`probe_aim.py` - the fake-enemy aim check."""

import math

import numpy as np
import torch

import features
import probe_aim
from opponents import ScriptedOpponent
from ppo_agent import ActorCritic


def _tiny_model():
    torch.manual_seed(0)
    return ActorCritic(features.OBS_DIM, hidden_size=16, num_layers=1,
                       slot_dim=features.HOTBAR_ACTION_DIM, lstm_hidden=0).eval()


def test_probe_covers_every_bearing_and_distance():
    results = probe_aim.probe(_tiny_model())
    assert len(results) == len(probe_aim.DISTANCES) * len(probe_aim.BEARINGS)
    assert {"needed_yaw", "yaw_delta", "attack", "move_x"} <= set(results[0])
    s = probe_aim.summarize(results)
    assert 0 <= s["turn_correct"] <= s["turn_cases"] <= len(results)


def test_required_turn_uses_the_scripted_bots_yaw_convention():
    """An enemy to the right needs a negative yaw; the scripted bot (which
    faces its enemy by construction) must turn the same way."""
    bot = ScriptedOpponent()
    for rel_x, rel_z in ((4.0, 1.0), (-4.0, 1.0)):
        obs = np.asarray([features.observation_to_row(probe_aim.fake_observation(rel_x, rel_z))], dtype=np.float32)
        yaw = float(bot.actions(obs)[0, 2])
        want = probe_aim.required_turn(rel_x, rel_z)
        assert math.copysign(1, yaw) == math.copysign(1, want), (rel_x, yaw, want)


def test_summarize_counts_only_clear_off_axis_turns():
    rows = [
        {"dist": 2.0, "bearing": 0, "needed_yaw": 0.0, "yaw_delta": 0.5, "attack": 1.0},   # ahead: not a turn case
        {"dist": 2.0, "bearing": 90, "needed_yaw": -1.5, "yaw_delta": -0.1, "attack": 0.0},  # right way
        {"dist": 2.0, "bearing": -90, "needed_yaw": 1.5, "yaw_delta": -0.1, "attack": 0.0},  # wrong way
    ]
    s = probe_aim.summarize(rows)
    assert (s["turn_correct"], s["turn_cases"]) == (1, 2)
    assert s["attacks_ahead_close"] is True
