"""Unit tests for `train.TerrainCurriculum` - the amplitude schedule. The
sim-relaunch wiring is exercised end-to-end by `smoke_train.py`.
"""

from train import TerrainCurriculum


def _curr(updates=100, start=0.0, stages=5, target=3.0):
    return TerrainCurriculum({"terrain_max_amplitude": target}, updates, start, stages)


def test_disabled_when_updates_zero_or_start_at_target():
    assert not _curr(updates=0).active
    assert not _curr(start=3.0).active
    c = _curr(updates=0)
    assert c.amplitude_for(1) == 3.0 == c.amplitude_for(999)


def test_ramp_starts_flat_and_ends_at_target():
    c = _curr(updates=100, start=0.0, stages=5, target=3.0)
    assert c.amplitude_for(1) == c._FLAT_THRESHOLD  # 0 is floored (avoids div-by-0)
    assert c.amplitude_for(100) == 3.0
    assert c.amplitude_for(500) == 3.0  # past the ramp
    # monotonic non-decreasing across the ramp
    amps = [c.amplitude_for(u) for u in range(1, 101)]
    assert amps == sorted(amps)
    assert set(amps) == {c._FLAT_THRESHOLD, 0.75, 1.5, 2.25, 3.0}  # 5 discrete steps


def test_config_for_sets_flat_only_at_the_low_end():
    c = _curr(updates=100, start=0.0, target=3.0)
    cfg0, amp0 = c.config_for(1)
    assert amp0 == c._FLAT_THRESHOLD and cfg0["terrain_flat_only"] is True
    cfg_end, amp_end = c.config_for(100)
    assert amp_end == 3.0 and "terrain_flat_only" not in cfg_end
    assert cfg_end["terrain_max_amplitude"] == 3.0


def test_base_config_is_preserved():
    c = TerrainCurriculum({"terrain_max_amplitude": 4.0, "kit": "uhc"}, 50, 1.0, 3)
    cfg, _ = c.config_for(25)
    assert cfg["kit"] == "uhc"
    assert c.target == 4.0
