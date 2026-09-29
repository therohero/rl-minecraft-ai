"""Shared observation/action feature engineering.

Imported by BOTH the training environment (env.py) and the live-inference
bridge (azalea-bot/inference_server.py) so the exact feature vector the
policy trained on is the one it sees against a real Minecraft bot. Never
duplicate this logic.

The normalization constants AND the observation *shape* (nearby
enemies / teammates / projectiles, and the size of the block-grid view)
come from the Rust sim's `Hello` handshake. `configure(...)` recomputes
`OBS_DIM` and the wire-column indices from them, so read `obs_dim()` /
`WIRE_*` rather than caching the module defaults.
"""

import math
from dataclasses import dataclass

import numpy as np

# Continuous actions: move_x, move_z, yaw_delta, pitch_delta
CONTINUOUS_ACTION_DIM = 4
# Binary actions: jump, attack, sprint, use_item, sneak
BINARY_ACTION_DIM = 5
# Item types (must match kit.rs) - width of the `inventory` observation block.
ITEM_COUNT = 17
# Status-effect types (must match effects.rs::EFFECT_COUNT) - width of the
# `self_effects` observation block.
EFFECT_COUNT = 9
# Physical hotbar slots (must match kit.rs::HOTBAR_SLOTS) - width of the
# `hotbar` observation block.
HOTBAR_SLOTS = 9
# Held-slot categorical action head width (must match kit.rs::HOTBAR_ACTION_DIM):
# 0..HOTBAR_SLOTS selects a physical slot, HOTBAR_SLOTS..this hotkeys an
# `Item` into the selected slot (id = action - HOTBAR_SLOTS).
HOTBAR_ACTION_DIM = HOTBAR_SLOTS + ITEM_COUNT
# One action row on the wire: 4 continuous + 5 binary + 1 held-slot index.
ACTION_FLOATS_PER_SLOT = 10

# --- wire observation layout (must match sim/src/protocol.rs, WIRE_VERSION 9) ---
# v9 adds the `self_effects` block (one float per effects.rs::Effect,
# amplifier + 1 while active, else 0) right after self_mining, and widens the
# `inventory` block by the 5 new splash-potion items. v8 added self_mining.
OBS_SELF_SCALARS = 27  # the fixed self scalars before the effects block
OBS_EFFECT_FLOATS = EFFECT_COUNT
OBS_SELF_FLOATS = OBS_SELF_SCALARS + OBS_EFFECT_FLOATS
OBS_INVENTORY_FLOATS = ITEM_COUNT - 1
OBS_HOTBAR_FLOATS = HOTBAR_SLOTS
OBS_OTHER_FLOATS = 13
OBS_PROJ_FLOATS = 7
OBS_BLOCKCOL_FLOATS = 4  # [top_rel, water, lava, cobweb]
OBS_GLOBAL_FLOATS = 3
OBS_EVENT_FLOATS = 7

_ARROW_NORM = 16.0
_COUNT_NORM = 16.0  # inventory counts are clamped to this then normalized
_FOOD_NORM = 20.0  # self_food is 0..20
_EFFECT_NORM = 4.0  # self_effects carry `amplifier + 1` (1..~3)


@dataclass(frozen=True)
class SimConstants:
    """Sim-side constants the feature code needs. Defaults mirror
    `SimConfig::default()` in sim/src/config.rs."""

    max_hp: float = 20.0
    match_time_seconds: float = 90.0
    arena_radius: float = 12.0
    terrain_max_amplitude: float = 3.0
    max_look_delta: float = 0.2
    max_ping_ms: float = 100.0
    max_observed_enemies: int = 3
    max_observed_teammates: int = 2
    max_observed_projectiles: int = 2
    block_view_size: int = 5


_ACTIVE = SimConstants()


def _n_other() -> int:
    return _ACTIVE.max_observed_enemies + _ACTIVE.max_observed_teammates


def _n_cols() -> int:
    return _ACTIVE.block_view_size * _ACTIVE.block_view_size


def wire_floats_per_slot() -> int:
    return (
        OBS_SELF_FLOATS
        + OBS_INVENTORY_FLOATS
        + OBS_HOTBAR_FLOATS
        + OBS_OTHER_FLOATS * _n_other()
        + OBS_PROJ_FLOATS * _ACTIVE.max_observed_projectiles
        + OBS_BLOCKCOL_FLOATS * _n_cols()
        + OBS_GLOBAL_FLOATS
        + OBS_EVENT_FLOATS
    )


def obs_dim() -> int:
    """Policy input width: raw self block with yaw/pitch each expanded to a
    sin/cos pair (+2), then inventory / other-player / projectile /
    block-column blocks and the 3 arena-global scalars. Event fields drop."""
    return (
        (OBS_SELF_FLOATS + 2)
        + OBS_INVENTORY_FLOATS
        + OBS_HOTBAR_FLOATS
        + OBS_OTHER_FLOATS * _n_other()
        + OBS_PROJ_FLOATS * _ACTIVE.max_observed_projectiles
        + OBS_BLOCKCOL_FLOATS * _n_cols()
        + OBS_GLOBAL_FLOATS
    )


def _events_base() -> int:
    return (
        OBS_SELF_FLOATS
        + OBS_INVENTORY_FLOATS
        + OBS_HOTBAR_FLOATS
        + OBS_OTHER_FLOATS * _n_other()
        + OBS_PROJ_FLOATS * _ACTIVE.max_observed_projectiles
        + OBS_BLOCKCOL_FLOATS * _n_cols()
        + OBS_GLOBAL_FLOATS
    )


WIRE_FLOATS_PER_SLOT = wire_floats_per_slot()
OBS_DIM = obs_dim()
WIRE_REWARD = _events_base() + 0
WIRE_DAMAGE_DEALT = _events_base() + 1
WIRE_DAMAGE_TAKEN = _events_base() + 2
WIRE_SWEPT = _events_base() + 3
WIRE_WON = _events_base() + 4
WIRE_LOST = _events_base() + 5
WIRE_DONE = _events_base() + 6


def configure(**overrides) -> SimConstants:
    """Sets the active sim constants and recomputes the observation layout.
    Unknown keys are ignored."""
    global _ACTIVE, WIRE_FLOATS_PER_SLOT, OBS_DIM
    global WIRE_REWARD, WIRE_DAMAGE_DEALT, WIRE_DAMAGE_TAKEN, WIRE_SWEPT, WIRE_WON, WIRE_LOST, WIRE_DONE
    known = {k: overrides[k] for k in SimConstants.__dataclass_fields__ if k in overrides}
    for k in (
        "max_observed_enemies",
        "max_observed_teammates",
        "max_observed_projectiles",
        "block_view_size",
    ):
        if k in known:
            known[k] = int(known[k])
    _ACTIVE = SimConstants(**{**_ACTIVE.__dict__, **known})

    WIRE_FLOATS_PER_SLOT = wire_floats_per_slot()
    OBS_DIM = obs_dim()
    e = _events_base()
    WIRE_REWARD, WIRE_DAMAGE_DEALT, WIRE_DAMAGE_TAKEN = e + 0, e + 1, e + 2
    WIRE_SWEPT, WIRE_WON, WIRE_LOST, WIRE_DONE = e + 3, e + 4, e + 5, e + 6
    return _ACTIVE


def active_constants() -> SimConstants:
    return _ACTIVE


MAX_HP = _ACTIVE.max_hp
MATCH_TIME_SECONDS = _ACTIVE.match_time_seconds
ARENA_RADIUS = _ACTIVE.arena_radius
TERRAIN_MAX_AMPLITUDE = _ACTIVE.terrain_max_amplitude
MAX_LOOK_DELTA = _ACTIVE.max_look_delta
MAX_PING_MS = _ACTIVE.max_ping_ms


def _other_block_row(o: dict) -> list:
    c = _ACTIVE
    if not o or not o.get("present", True):
        return [0.0] * OBS_OTHER_FLOATS
    return [
        1.0,
        float(o.get("hp", 0.0)) / c.max_hp,
        float(o.get("rel_x", 0.0)),
        float(o.get("rel_y", 0.0)),
        float(o.get("rel_z", 0.0)),
        float(o.get("vel_x", 0.0)),
        float(o.get("vel_y", 0.0)),
        float(o.get("vel_z", 0.0)),
        float(o.get("ground_height", 0.0)) / c.terrain_max_amplitude,
        float(o.get("blocking", 0.0)),
        float(o.get("eating", 0.0)),
        float(o.get("held_ranged", 0.0)),
        float(o.get("sneaking", 0.0)),
    ]


def _proj_block_row(p: dict) -> list:
    if not p or not p.get("present", True):
        return [0.0] * OBS_PROJ_FLOATS
    return [
        1.0,
        float(p.get("rel_x", 0.0)),
        float(p.get("rel_y", 0.0)),
        float(p.get("rel_z", 0.0)),
        float(p.get("vel_x", 0.0)),
        float(p.get("vel_y", 0.0)),
        float(p.get("vel_z", 0.0)),
    ]


def _blockcol_row(col) -> list:
    """One block-grid column: `[top_rel, water, lava, cobweb]`, either a
    dict or a 4-list. `top_rel` is already normalized by the sim."""
    if col is None:
        return [0.0] * OBS_BLOCKCOL_FLOATS
    if isinstance(col, dict):
        return [
            float(col.get("top_rel", 0.0)),
            float(col.get("water", 0.0)),
            float(col.get("lava", 0.0)),
            float(col.get("cobweb", 0.0)),
        ]
    return [float(x) for x in list(col)[:OBS_BLOCKCOL_FLOATS]] + [0.0] * (
        OBS_BLOCKCOL_FLOATS - len(col)
    )


def _effects_row(eff) -> list:
    """The `self_effects` block: one float per effect, `amplifier + 1` while
    active (else 0), normalized."""
    e = list(eff or [])
    e = e[:OBS_EFFECT_FLOATS] + [0.0] * (OBS_EFFECT_FLOATS - len(e))
    return [float(x) / _EFFECT_NORM for x in e]


def _inventory_row(inv) -> list:
    inv = list(inv or [])
    inv = inv[:OBS_INVENTORY_FLOATS] + [0.0] * (OBS_INVENTORY_FLOATS - len(inv))
    return [min(float(x), _COUNT_NORM) / _COUNT_NORM for x in inv]


def _hotbar_row(hotbar) -> list:
    """The kit's slot->item layout, item ids normalized by ITEM_COUNT-1."""
    hb = list(hotbar or [])
    hb = hb[:OBS_HOTBAR_FLOATS] + [0.0] * (OBS_HOTBAR_FLOATS - len(hb))
    return [float(x) / (ITEM_COUNT - 1) for x in hb]


def observation_to_row(obs: dict) -> list:
    """One raw observation dict -> the fixed-size feature list the policy
    consumes, in `wire_batch_to_obs` order. `obs` carries the `self_*`
    scalars, an `inventory` list, and `enemies` / `teammates` /
    `projectiles` / `block_view` lists (nearest / row-major first; short
    lists zero-padded). Missing optional fields default to 0."""
    c = _ACTIVE
    yaw = obs["self_yaw"]
    pitch = obs["self_pitch"]
    row = [
        obs["self_hp"] / c.max_hp,
        obs["self_vel_x"],
        obs["self_vel_y"],
        obs["self_vel_z"],
        math.sin(yaw),
        math.cos(yaw),
        math.sin(pitch),
        math.cos(pitch),
        1.0 if obs["self_on_ground"] else 0.0,
        obs["self_attack_cooldown"],
        obs.get("self_ping_ms", 0.0) / c.max_ping_ms,
        obs.get("self_shield", 0.0),
        obs["self_dist_from_center"],
        obs.get("self_ground_height", 0.0) / c.terrain_max_amplitude,
        obs.get("self_slope_forward", 0.0) / c.terrain_max_amplitude,
        obs.get("self_slope_right", 0.0) / c.terrain_max_amplitude,
        obs.get("self_hurt", 0.0),
        float(obs.get("self_held", 0.0)) / (ITEM_COUNT - 1),
        obs.get("self_absorption", 0.0) / c.max_hp,
        obs.get("self_eating", 0.0),
        obs.get("self_bow_draw", 0.0),
        obs.get("self_burning", 0.0),
        obs.get("self_shield_disabled", 0.0),
        obs.get("self_arrows", 0.0) / _ARROW_NORM,
        float(obs.get("self_slot", 0.0)) / (HOTBAR_SLOTS - 1),
        obs.get("self_swap_lockout", 0.0),
        float(obs.get("self_food", 20.0)) / _FOOD_NORM,
        obs.get("self_sneaking", 0.0),
        obs.get("self_mining", 0.0),
    ]
    row += _effects_row(obs.get("self_effects"))
    row += _inventory_row(obs.get("inventory"))
    row += _hotbar_row(obs.get("hotbar"))

    def slots(key, count, fn):
        items = list(obs.get(key, []))[:count]
        items += [None] * (count - len(items))
        out = []
        for it in items:
            out.extend(fn(it))
        return out

    row += slots("enemies", c.max_observed_enemies, _other_block_row)
    row += slots("teammates", c.max_observed_teammates, _other_block_row)
    row += slots("projectiles", c.max_observed_projectiles, _proj_block_row)
    row += slots("block_view", _n_cols(), _blockcol_row)
    row += [
        obs["time_left"] / c.match_time_seconds,
        float(obs.get("enemies_alive", 0.0)),
        float(obs.get("teammates_alive", 0.0)),
    ]
    return row


def wire_batch_to_obs(raw: np.ndarray) -> np.ndarray:
    """Vectorized decode of a whole batch of raw wire rows into the
    `[N, obs_dim()]` float32 feature matrix (field order =
    `Observation::write_wire`)."""
    c = _ACTIVE
    n = raw.shape[0]
    out = np.empty((n, obs_dim()), dtype=np.float32)

    yaw = raw[:, 4]
    pitch = raw[:, 5]
    out[:, 0] = raw[:, 0] / c.max_hp
    out[:, 1:4] = raw[:, 1:4]
    out[:, 4] = np.sin(yaw)
    out[:, 5] = np.cos(yaw)
    out[:, 6] = np.sin(pitch)
    out[:, 7] = np.cos(pitch)
    out[:, 8] = raw[:, 6]
    out[:, 9] = raw[:, 7]
    out[:, 10] = raw[:, 8] / c.max_ping_ms
    out[:, 11] = raw[:, 9]
    out[:, 12] = raw[:, 10]
    out[:, 13] = raw[:, 11] / c.terrain_max_amplitude
    out[:, 14] = raw[:, 12] / c.terrain_max_amplitude
    out[:, 15] = raw[:, 13] / c.terrain_max_amplitude
    out[:, 16] = raw[:, 14]  # hurt
    out[:, 17] = raw[:, 15] / (ITEM_COUNT - 1)  # held
    out[:, 18] = raw[:, 16] / c.max_hp  # absorption
    out[:, 19] = raw[:, 17]  # eating
    out[:, 20] = raw[:, 18]  # bow_draw
    out[:, 21] = raw[:, 19]  # burning
    out[:, 22] = raw[:, 20]  # shield_disabled
    out[:, 23] = raw[:, 21] / _ARROW_NORM  # arrows
    out[:, 24] = raw[:, 22] / (HOTBAR_SLOTS - 1)  # self_slot
    out[:, 25] = raw[:, 23]  # self_swap_lockout
    out[:, 26] = raw[:, 24] / _FOOD_NORM  # self_food
    out[:, 27] = raw[:, 25]  # self_sneaking
    out[:, 28] = raw[:, 26]  # self_mining
    # self_effects block: passes through, normalized.
    out[:, 29 : 29 + OBS_EFFECT_FLOATS] = (
        raw[:, OBS_SELF_SCALARS : OBS_SELF_SCALARS + OBS_EFFECT_FLOATS] / _EFFECT_NORM
    )
    w = 29 + OBS_EFFECT_FLOATS
    r = OBS_SELF_FLOATS

    # inventory
    out[:, w : w + OBS_INVENTORY_FLOATS] = np.minimum(
        raw[:, r : r + OBS_INVENTORY_FLOATS], _COUNT_NORM
    ) / _COUNT_NORM
    w += OBS_INVENTORY_FLOATS
    r += OBS_INVENTORY_FLOATS

    # hotbar slot->item layout (ids normalized like self_held)
    out[:, w : w + OBS_HOTBAR_FLOATS] = raw[:, r : r + OBS_HOTBAR_FLOATS] / (ITEM_COUNT - 1)
    w += OBS_HOTBAR_FLOATS
    r += OBS_HOTBAR_FLOATS

    for _ in range(_n_other()):
        out[:, w + 0] = raw[:, r + 0]  # present
        out[:, w + 1] = raw[:, r + 1] / c.max_hp
        out[:, w + 2 : w + 8] = raw[:, r + 2 : r + 8]
        out[:, w + 8] = raw[:, r + 8] / c.terrain_max_amplitude
        out[:, w + 9 : w + 13] = raw[:, r + 9 : r + 13]  # blocking, eating, held_ranged, sneaking
        w += OBS_OTHER_FLOATS
        r += OBS_OTHER_FLOATS

    for _ in range(c.max_observed_projectiles):
        out[:, w : w + OBS_PROJ_FLOATS] = raw[:, r : r + OBS_PROJ_FLOATS]
        w += OBS_PROJ_FLOATS
        r += OBS_PROJ_FLOATS

    # block-grid columns pass straight through (top_rel already normalized).
    ncols = _n_cols()
    if ncols:
        span = OBS_BLOCKCOL_FLOATS * ncols
        out[:, w : w + span] = raw[:, r : r + span]
        w += span
        r += span

    out[:, w + 0] = raw[:, r + 0] / c.match_time_seconds
    out[:, w + 1] = raw[:, r + 1]
    out[:, w + 2] = raw[:, r + 2]
    return out


def observation_to_vector(obs: dict) -> np.ndarray:
    return np.array(observation_to_row(obs), dtype=np.float32)


def action_vector_to_dict(
    raw_continuous: np.ndarray,
    jump: bool,
    attack: bool,
    sprint: bool,
    use_item: bool = False,
    held_slot: int = 0,
) -> dict:
    """Maps a policy's raw outputs to the action dict the sim / live bridge
    expects. `raw_continuous` is already squashed to [-1, 1] in the order
    [move_x, move_z, yaw_delta, pitch_delta]."""
    move_x, move_z, yaw_delta, pitch_delta = raw_continuous
    return {
        "move_x": float(move_x),
        "move_z": float(move_z),
        "yaw_delta": float(yaw_delta) * _ACTIVE.max_look_delta,
        "pitch_delta": float(pitch_delta) * _ACTIVE.max_look_delta,
        "jump": bool(jump),
        "attack": bool(attack),
        "sprint": bool(sprint),
        "use_item": bool(use_item),
        "held_slot": int(held_slot),
    }
