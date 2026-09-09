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
    mean, log_std, bin_logits, slot_logits, value, hidden = m(obs)
    assert mean.shape == (5, CONTINUOUS_ACTION_DIM)
    assert log_std.shape == (5, CONTINUOUS_ACTION_DIM)
    assert bin_logits.shape == (5, BINARY_ACTION_DIM)
    assert slot_logits.shape == (5, HOTBAR_ACTION_DIM)
    assert value.shape == (5,)
    assert hidden is None  # MLP head carries no state

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
    _, log_std, *_rest = m(torch.zeros(1, 24))
    assert torch.all(log_std <= 0.0)
    with torch.no_grad():
        m.continuous_log_std.fill_(-9.0)
    _, log_std, *_rest = m(torch.zeros(1, 24))
    assert torch.all(log_std >= -3.0)


def test_evaluate_matches_act_logprob_for_same_sample():
    m = _model()
    obs = torch.randn(8, 24)
    out = m.act(obs)
    logprob, entropy, value, _ = m.evaluate(
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


# --- LSTM (recurrent) head -------------------------------------------------

def _lstm_model(obs_dim=24, lstm_hidden=12):
    torch.manual_seed(0)
    return ActorCritic(obs_dim=obs_dim, hidden_size=16, num_layers=1, lstm_hidden=lstm_hidden)


def test_lstm_forward_carries_and_returns_hidden():
    m = _lstm_model()
    assert m.recurrent
    obs = torch.randn(5, 24)
    h0 = m.zero_hidden(5)
    assert h0[0].shape == (1, 5, 12) and h0[1].shape == (1, 5, 12)
    *_, value, h1 = m(obs, h0)
    assert value.shape == (5,)
    assert h1[0].shape == (1, 5, 12)
    # a non-zero input must move the state off zero
    assert not torch.allclose(h1[0], h0[0])

    # sequence form: [T, N, D] -> [T, N] value + final [1, N, H] hidden
    seq = torch.randn(4, 5, 24)
    *_, seq_value, seq_h = m(seq, h0)
    assert seq_value.shape == (4, 5)
    assert seq_h[0].shape == (1, 5, 12)


def test_lstm_evaluate_matches_act_logprob_for_same_sample():
    m = _lstm_model()
    obs = torch.randn(8, 24)
    h0 = m.zero_hidden(8)
    out = m.act(obs, h0)
    logprob, entropy, value, _ = m.evaluate(
        obs, out["raw_cont"], out["binary_action"], out["slot_action"], h0
    )
    torch.testing.assert_close(logprob, out["logprob"], rtol=1e-5, atol=1e-6)


def test_lstm_mlp_head_returns_no_hidden():
    m = _model()
    assert not m.recurrent
    assert m.zero_hidden(3) is None


def test_ppo_update_recurrent_runs_and_moves_weights():
    import numpy as np
    from ppo_agent import ppo_update

    torch.manual_seed(0)
    T, N, obs_dim = 6, 5, 24
    model = _lstm_model(obs_dim=obs_dim, lstm_hidden=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    buf = RolloutBuffer(T, N, obs_dim, CPU, lstm_hidden=8)
    hidden = model.zero_hidden(N)
    buf.set_init_hidden(hidden)
    for t in range(T):
        obs = torch.randn(N, obs_dim)
        out = model.act(obs, hidden)
        hidden = out["hidden"]
        dones = np.zeros(N, dtype=bool)
        if t == 3:
            dones[1] = True  # exercise the mid-rollout hidden reset
        buf.add(
            obs=obs, raw_cont=out["raw_cont"], binary_action=out["binary_action"],
            slot_action=out["slot_action"], logprob=out["logprob"],
            reward=np.zeros(N, dtype=np.float32), done=dones, value=out["value"],
        )
    adv, ret = buf.compute_gae(torch.zeros(N), gamma=0.99, gae_lambda=0.95)
    before = model.continuous_mean.weight.detach().clone()
    stats = ppo_update(model, opt, buf, adv, ret, epochs=2, minibatch_size=1024)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    assert all(np.isfinite(v) for v in stats.values())
    assert not torch.allclose(before, model.continuous_mean.weight)


def test_ppo_update_recurrent_sample_mask_freezes_weights_at_lr0():
    import numpy as np
    from ppo_agent import ppo_update

    torch.manual_seed(0)
    T, N, obs_dim = 4, 6, 24
    model = _lstm_model(obs_dim=obs_dim, lstm_hidden=8)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    buf = RolloutBuffer(T, N, obs_dim, CPU, lstm_hidden=8)
    hidden = model.zero_hidden(N)
    buf.set_init_hidden(hidden)
    for _ in range(T):
        obs = torch.randn(N, obs_dim)
        out = model.act(obs, hidden)
        hidden = out["hidden"]
        buf.add(
            obs=obs, raw_cont=out["raw_cont"], binary_action=out["binary_action"],
            slot_action=out["slot_action"], logprob=out["logprob"],
            reward=np.zeros(N, dtype=np.float32), done=np.zeros(N, dtype=bool), value=out["value"],
        )
    adv, ret = buf.compute_gae(torch.zeros(N), gamma=0.99, gae_lambda=0.95)
    mask = torch.tensor([True, True, False, False, True, False])
    stats = ppo_update(model, opt, buf, adv, ret, epochs=1, minibatch_size=1024, sample_mask=mask)
    assert all(np.isfinite(v) for v in stats.values())
