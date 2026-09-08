"""Tests for `features.py` - the observation/action feature engineering
shared by training (`env.py`) and live inference
(`azalea-bot/inference_server.py`).

The load-bearing one is `dict_and_wire_paths_agree`: `observation_to_row`
(the per-dict path the live bridge uses) and `wire_batch_to_obs` (the
vectorized path training uses) are two hand-written decoders of the same
wire layout, and the module docstring's "Never duplicate this logic" is
only safe if they stay byte-for-byte equivalent.
"""

import math

import numpy as np
import pytest

import features
from features import SimConstants


@pytest.fixture(autouse=True)
def _reset_active_constants():
    """`features.configure()` mutates module-level globals; restore the
    defaults after every test."""
    yield
    features.configure(**SimConstants().__dict__)


def _self_dict_from_raw(raw):
    """The self-block wire columns (order = `wire_batch_to_obs`) as the dict
    keys `observation_to_row` reads."""
    keys = [
        "self_hp", "self_vel_x", "self_vel_y", "self_vel_z",
        "self_yaw", "self_pitch", "self_on_ground", "self_attack_cooldown",
        "self_ping_ms", "self_shield", "self_dist_from_center",
        "self_ground_height", "self_slope_forward", "self_slope_right",
        "self_hurt", "self_held", "self_absorption", "self_eating",
        "self_bow_draw", "self_burning", "self_shield_disabled",
        "self_arrows", "self_slot", "self_swap_lockout", "self_food",
        "self_sneaking", "self_mining",
    ]
    assert len(keys) == features.OBS_SELF_SCALARS
    d = {k: float(v) for k, v in zip(keys, raw)}
    d["self_on_ground"] = bool(raw[6])
    return d


_OTHER_KEYS = [
    "present", "hp", "rel_x", "rel_y", "rel_z", "vel_x", "vel_y", "vel_z",
    "ground_height", "blocking", "eating", "held_ranged", "sneaking",
]
_PROJ_KEYS = ["present", "rel_x", "rel_y", "rel_z", "vel_x", "vel_y", "vel_z"]
_COL_KEYS = ["top_rel", "water", "lava", "cobweb"]


def _build_raw_and_obs(consts: SimConstants):
    """A single wire row of distinct values plus the equivalent observation
    dict. `present` columns are pinned to exactly 1.0 (the dict path
    hard-codes 1.0 for a present row, so any other value would be a
    spurious mismatch)."""
    features.configure(**consts.__dict__)
    c = features.active_constants()
    n_enemy, n_team = c.max_observed_enemies, c.max_observed_teammates
    n_proj = c.max_observed_projectiles
    ncols = c.block_view_size**2

    raw = (np.arange(1, features.WIRE_FLOATS_PER_SLOT + 1, dtype=np.float32) * 0.013)
    obs = {}
    p = 0

    def take(n):
        nonlocal p
        chunk = raw[p : p + n]
        p += n
        return chunk

    self_raw = take(features.OBS_SELF_FLOATS)
    self_raw[6] = 1.0  # on_ground is 0/1
    obs.update(_self_dict_from_raw(self_raw))
    obs["self_effects"] = list(self_raw[features.OBS_SELF_SCALARS : features.OBS_SELF_FLOATS])

    obs["inventory"] = list(take(features.OBS_INVENTORY_FLOATS))
    obs["hotbar"] = list(take(features.OBS_HOTBAR_FLOATS))

    def other_rows(count):
        rows = []
        for _ in range(count):
            r = take(features.OBS_OTHER_FLOATS)
            r[0] = 1.0  # present
            rows.append({k: float(v) for k, v in zip(_OTHER_KEYS, r)})
        return rows

    obs["enemies"] = other_rows(n_enemy)
    obs["teammates"] = other_rows(n_team)

    projs = []
    for _ in range(n_proj):
        r = take(features.OBS_PROJ_FLOATS)
        r[0] = 1.0
        projs.append({k: float(v) for k, v in zip(_PROJ_KEYS, r)})
    obs["projectiles"] = projs

    cols = []
    for _ in range(ncols):
        r = take(features.OBS_BLOCKCOL_FLOATS)
        cols.append({k: float(v) for k, v in zip(_COL_KEYS, r)})
    obs["block_view"] = cols

    g = take(features.OBS_GLOBAL_FLOATS)
    obs["time_left"] = float(g[0])
    obs["enemies_alive"] = float(g[1])
    obs["teammates_alive"] = float(g[2])

    # Trailing event floats are not part of a row; leave them in `raw` so
    # its width matches the wire.
    take(features.OBS_EVENT_FLOATS)
    assert p == features.WIRE_FLOATS_PER_SLOT
    return raw, obs


@pytest.mark.parametrize(
    "consts",
    [
        SimConstants(),
        SimConstants(max_observed_enemies=1, max_observed_teammates=0,
                     max_observed_projectiles=0, block_view_size=3),
        SimConstants(max_observed_enemies=3, max_observed_teammates=2,
                     max_observed_projectiles=2, block_view_size=7,
                     max_hp=30.0, arena_radius=20.0, max_ping_ms=250.0),
    ],
)
def test_dict_and_wire_paths_agree(consts):
    raw, obs = _build_raw_and_obs(consts)

    from_dict = np.asarray(features.observation_to_row(obs), dtype=np.float32)
    from_wire = features.wire_batch_to_obs(raw.reshape(1, -1))[0]

    assert from_dict.shape == from_wire.shape == (features.obs_dim(),)
    np.testing.assert_allclose(from_dict, from_wire, rtol=0, atol=1e-6)


def test_obs_row_width_matches_obs_dim():
    _, obs = _build_raw_and_obs(SimConstants())
    assert len(features.observation_to_row(obs)) == features.obs_dim()


def test_wire_width_is_row_blocks_plus_events():
    c = features.active_constants()
    n_other = c.max_observed_enemies + c.max_observed_teammates
    ncols = c.block_view_size**2
    expected = (
        features.OBS_SELF_FLOATS
        + features.OBS_INVENTORY_FLOATS
        + features.OBS_HOTBAR_FLOATS
        + features.OBS_OTHER_FLOATS * n_other
        + features.OBS_PROJ_FLOATS * c.max_observed_projectiles
        + features.OBS_BLOCKCOL_FLOATS * ncols
        + features.OBS_GLOBAL_FLOATS
        + features.OBS_EVENT_FLOATS
    )
    assert features.wire_floats_per_slot() == expected
    # The policy input drops the 7 event floats but gains the yaw/pitch
    # sin/cos expansion (+2).
    assert features.obs_dim() == expected - features.OBS_EVENT_FLOATS + 2


def test_configure_recomputes_layout_and_wire_indices():
    features.configure(max_observed_enemies=5, block_view_size=9, max_ping_ms=1234.0)
    c = features.active_constants()
    assert c.max_observed_enemies == 5
    assert c.block_view_size == 9
    assert c.max_ping_ms == 1234.0
    assert features.OBS_DIM == features.obs_dim()
    assert features.WIRE_DONE == features.wire_floats_per_slot() - 1
    assert features.WIRE_REWARD == features.obs_dim() - 2  # events start right after the row


def test_configure_ignores_unknown_keys_and_coerces_ints():
    before = features.active_constants()
    features.configure(nonsense_key=999, max_observed_enemies=2.0)
    after = features.active_constants()
    assert not hasattr(after, "nonsense_key")
    assert after.max_observed_enemies == 2 and isinstance(after.max_observed_enemies, int)
    assert after.max_hp == before.max_hp


def test_ping_is_normalized_by_max_ping_ms():
    features.configure(max_ping_ms=200.0)
    _, obs = _build_raw_and_obs(features.active_constants())
    obs["self_ping_ms"] = 100.0
    row = features.observation_to_row(obs)
    assert row[10] == pytest.approx(0.5)


def test_short_lists_zero_pad_and_absent_rows_are_zero():
    features.configure(max_observed_enemies=3, max_observed_teammates=0,
                       max_observed_projectiles=0, block_view_size=1)
    obs = {
        "self_hp": 20.0, "self_vel_x": 0.0, "self_vel_y": 0.0, "self_vel_z": 0.0,
        "self_yaw": 0.0, "self_pitch": 0.0, "self_on_ground": True,
        "self_attack_cooldown": 0.0, "self_dist_from_center": 0.0, "time_left": 90.0,
        "enemies": [{"present": True, "hp": 20.0, "rel_x": 1.0}],  # 1 of 3
        "inventory": [2],  # 1 of 11
        "hotbar": [1, 2],  # 2 of 9
    }
    row = features.observation_to_row(obs)
    assert len(row) == features.obs_dim()
    # enemy slot 1 and 2 are absent -> their 13-wide blocks are all zero.
    base = features.OBS_SELF_FLOATS + 2 + features.OBS_INVENTORY_FLOATS + features.OBS_HOTBAR_FLOATS
    absent = row[base + features.OBS_OTHER_FLOATS : base + 3 * features.OBS_OTHER_FLOATS]
    assert all(v == 0.0 for v in absent)


def test_present_false_row_is_all_zero():
    from features import _other_block_row, _proj_block_row, OBS_OTHER_FLOATS, OBS_PROJ_FLOATS

    assert _other_block_row({"present": False, "hp": 5.0}) == [0.0] * OBS_OTHER_FLOATS
    assert _proj_block_row({"present": False, "rel_x": 5.0}) == [0.0] * OBS_PROJ_FLOATS


def test_action_vector_to_dict_scales_look_and_coerces_flags():
    features.configure(max_look_delta=3.0)
    out = features.action_vector_to_dict(
        np.array([0.5, -0.25, 1.0, -1.0], dtype=np.float32),
        jump=1, attack=0, sprint=True, use_item=False, held_slot=4,
    )
    assert out["move_x"] == pytest.approx(0.5)
    assert out["move_z"] == pytest.approx(-0.25)
    assert out["yaw_delta"] == pytest.approx(3.0)
    assert out["pitch_delta"] == pytest.approx(-3.0)
    assert out["jump"] is True and out["attack"] is False and out["sprint"] is True
    assert out["held_slot"] == 4 and isinstance(out["held_slot"], int)


def test_yaw_pitch_expand_to_unit_sin_cos_pairs():
    _, obs = _build_raw_and_obs(SimConstants())
    obs["self_yaw"] = 1.1
    obs["self_pitch"] = -0.4
    row = features.observation_to_row(obs)
    assert row[4] == pytest.approx(math.sin(1.1))
    assert row[5] == pytest.approx(math.cos(1.1))
    assert row[6] == pytest.approx(math.sin(-0.4))
    assert row[7] == pytest.approx(math.cos(-0.4))
    assert row[4] ** 2 + row[5] ** 2 == pytest.approx(1.0)
