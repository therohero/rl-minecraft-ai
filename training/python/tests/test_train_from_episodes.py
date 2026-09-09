"""Tests for `train_from_episodes._load_session` - the trailing outcome
record the mod now logs (`{t, outcome, self_hp_end, enemy_hp_end}`) must
drive the terminal reward, and datasets without one must still fall back to
the old observation-stream heuristic."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import train_from_episodes as tfe

_ARGS = SimpleNamespace(per_hp_dealt=1.0, per_hp_taken=1.0, win=100.0, loss=100.0)


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    # _load_session only needs a stable feature-vector length here.
    monkeypatch.setattr(tfe.features, "observation_to_row",
                        lambda o: np.zeros(4, dtype=np.float32))
    monkeypatch.setattr(tfe, "ARGS", _ARGS, raising=False)


def _tick(t, self_hp, enemy_hp, present=True):
    enemies = [{"present": 1.0 if present else 0.0, "hp": enemy_hp}]
    return {"t": t, "kit": "sword", "target": "foe",
            "obs": {"self_hp": self_hp, "enemies": enemies}, "action": {}}


def _write(tmp_path, ticks, outcome_rec=None):
    p = tmp_path / "session-test.jsonl"
    with p.open("w") as f:
        for r in ticks:
            f.write(json.dumps(r) + "\n")
        if outcome_rec is not None:
            f.write(json.dumps(outcome_rec) + "\n")
    return str(p)


def test_explicit_win_outcome_applies_bonus(tmp_path):
    # enemy stays "present" the whole time -> the legacy heuristic would NOT
    # award a win, so a positive terminal reward proves the record won.
    ticks = [_tick(i, 20.0, 20.0) for i in range(5)]
    path = _write(tmp_path, ticks, {"t": 5, "outcome": "win",
                                    "self_hp_end": 20.0, "enemy_hp_end": 0.0})
    _, _, _, _, reward, done = tfe._load_session(path, scale=10.0)
    assert done[-1] == 1.0
    assert reward[-1] == pytest.approx(_ARGS.win + 20.0)  # bonus + final-tick kill


def test_explicit_loss_outcome_applies_penalty(tmp_path):
    ticks = [_tick(i, 20.0, 20.0) for i in range(5)]
    path = _write(tmp_path, ticks, {"t": 5, "outcome": "loss",
                                    "self_hp_end": 0.0, "enemy_hp_end": 20.0})
    _, _, _, _, reward, _ = tfe._load_session(path, scale=10.0)
    assert reward[-1] == pytest.approx(-_ARGS.loss - 20.0)  # penalty + final-tick damage taken


def test_unknown_outcome_no_terminal_bonus(tmp_path):
    ticks = [_tick(i, 20.0, 20.0) for i in range(5)]
    path = _write(tmp_path, ticks, {"t": 5, "outcome": "unknown",
                                    "self_hp_end": 20.0, "enemy_hp_end": 20.0})
    _, _, _, _, reward, _ = tfe._load_session(path, scale=10.0)
    assert reward[-1] == pytest.approx(0.0)


def test_legacy_session_without_outcome_uses_heuristic(tmp_path):
    # enemy present at t0, gone at the end, we're alive -> heuristic win.
    ticks = [_tick(i, 20.0, 20.0) for i in range(4)]
    ticks.append(_tick(4, 20.0, 0.0, present=False))
    path = _write(tmp_path, ticks)  # no outcome record
    _, _, _, _, reward, _ = tfe._load_session(path, scale=10.0)
    assert reward[-1] == pytest.approx(_ARGS.win)
