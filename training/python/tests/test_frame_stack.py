"""Tests for `frame_stack.FrameStacker` - ordering, reset, and the
episode-boundary history clear."""

import numpy as np

from frame_stack import FrameStacker


def test_stacked_dim_and_shape():
    fs = FrameStacker(num_slots=4, base_dim=3, n=2)
    assert fs.stacked_dim == 6
    out = fs.reset(np.ones((4, 3), dtype=np.float32))
    assert out.shape == (4, 6)


def test_reset_puts_obs_newest_and_zeros_history():
    fs = FrameStacker(num_slots=1, base_dim=2, n=3)
    out = fs.reset(np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.array_equal(out[0], [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])


def test_push_shifts_newest_first():
    fs = FrameStacker(num_slots=1, base_dim=1, n=3)
    fs.reset(np.array([[1.0]], dtype=np.float32))
    fs.push(np.array([[2.0]], dtype=np.float32), np.array([False]))
    out = fs.push(np.array([[3.0]], dtype=np.float32), np.array([False]))
    assert np.array_equal(out[0], [3.0, 2.0, 1.0])


def test_done_clears_that_slots_history_only():
    fs = FrameStacker(num_slots=2, base_dim=1, n=3)
    fs.reset(np.array([[1.0], [10.0]], dtype=np.float32))
    fs.push(np.array([[2.0], [20.0]], dtype=np.float32), np.array([False, False]))
    out = fs.push(np.array([[3.0], [30.0]], dtype=np.float32), np.array([True, False]))
    assert np.array_equal(out[0], [3.0, 0.0, 0.0])       # slot 0 reset
    assert np.array_equal(out[1], [30.0, 20.0, 10.0])    # slot 1 untouched


def test_n1_is_a_passthrough():
    fs = FrameStacker(num_slots=3, base_dim=4, n=1)
    obs = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(fs.reset(obs), obs)
    nxt = obs + 1
    assert np.array_equal(fs.push(nxt, np.array([False, True, False])), nxt)
