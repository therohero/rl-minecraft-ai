"""Tests for `ppo_agent.py`: the shared actor-critic's tensor shapes and
the GAE recurrence in `RolloutBuffer.compute_gae`."""

import torch

from features import BINARY_ACTION_DIM, CONTINUOUS_ACTION_DIM, HOTBAR_ACTION_DIM
from ppo_agent import ActorCritic, RolloutBuffer


CPU = torch.device("cpu")


def _model(obs_dim=24):
    torch.manual_seed(0)
    return ActorCritic(obs_dim=obs_dim, hidden_size=16, num_layers=2)


def test_forward_and_act_shapes():
    m = _model()
    obs = torch.zeros(5, 24)
    mean, log_std, bin_logits, slot_logits, value = m(obs)
    assert mean.shape == (5, CONTINUOUS_ACTION_DIM)
    assert log_std.shape == (5, CONTINUOUS_ACTION_DIM)
    assert bin_logits.shape == (5, BINARY_ACTION_DIM)
    assert slot_logits.shape == (5, HOTBAR_ACTION_DIM)
    assert value.shape == (5,)

    out = m.act(obs)
    assert out["squashed_cont"].shape == (5, CONTINUOUS_ACTION_DIM)
    assert out["binary_action"].shape == (5, BINARY_ACTION_DIM)
    assert out["slot_action"].shape == (5,)
    assert out["logprob"].shape == (5,)
    assert torch.all(out["squashed_cont"].abs() <= 1.0)
    assert set(out["binary_action"].unique().tolist()) <= {0.0, 1.0}
    assert torch.all(out["slot_action"] < HOTBAR_ACTION_DIM)


def test_log_std_is_clamped_to_non_positive():
    m = _model()
    with torch.no_grad():
        m.continuous_log_std.fill_(5.0)
    _, log_std, *_ = m(torch.zeros(1, 24))
    assert torch.all(log_std <= 0.0)
    with torch.no_grad():
        m.continuous_log_std.fill_(-9.0)
    _, log_std, *_ = m(torch.zeros(1, 24))
    assert torch.all(log_std >= -3.0)


def test_evaluate_matches_act_logprob_for_same_sample():
    m = _model()
    obs = torch.randn(8, 24)
    out = m.act(obs)
    logprob, entropy, value = m.evaluate(
        obs, out["raw_cont"], out["binary_action"], out["slot_action"]
    )
    torch.testing.assert_close(logprob, out["logprob"], rtol=1e-5, atol=1e-6)
    assert entropy.shape == (8,)
    assert torch.all(entropy > 0)  # continuous Gaussian entropy dominates


def test_invalid_num_layers_raises():
    import pytest

    with pytest.raises(ValueError):
        ActorCritic(obs_dim=24, num_layers=0)


def _buf(rewards, values, dones, last_value):
    T, N = len(rewards), 1
    b = RolloutBuffer(rollout_len=T, num_slots=N, obs_dim=1, device=CPU)
    for r, v, d in zip(rewards, values, dones):
        b.reward[b.ptr, 0] = r
        b.value[b.ptr, 0] = v
        b.done[b.ptr, 0] = d
        b.ptr += 1
    return b.compute_gae(torch.tensor([last_value]), gamma=1.0, gae_lambda=1.0)


def test_gae_with_zero_value_is_reward_to_go():
    # gamma = lambda = 1, all values 0, no dones -> advantage[t] = sum of
    # future rewards, returns == advantages.
    adv, ret = _buf([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], last_value=0.0)
    torch.testing.assert_close(adv.squeeze(1), torch.tensor([6.0, 5.0, 3.0]))
    torch.testing.assert_close(ret, adv)


def test_gae_done_flag_stops_bootstrap():
    # A done at t=0 must cut the return at t=0 off from t>=1.
    adv, _ = _buf([1.0, 100.0], [0.0, 0.0], [1.0, 0.0], last_value=0.0)
    assert adv[0, 0].item() == 1.0
    assert adv[1, 0].item() == 100.0


def test_gae_bootstraps_from_last_value():
    adv, _ = _buf([0.0], [0.0], [0.0], last_value=7.0)
    assert adv[0, 0].item() == 7.0


def test_buffer_to_same_device_is_identity():
    b = RolloutBuffer(rollout_len=2, num_slots=3, obs_dim=4, device=CPU)
    assert b.to(CPU) is b
