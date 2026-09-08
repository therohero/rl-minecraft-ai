//! Wire protocol between the Rust simulation and the Python trainer.
//!
//! Transport: length-framed binary datagrams over UDP on localhost (see
//! `server.rs`). Action/state batches are flat little-endian `f32` arrays
//! the Python side reads straight into a numpy buffer with no per-element
//! parsing. The exchange is strictly request/response with a 32-bit
//! sequence number; a retransmitted request replays the cached reply.
//!
//! Datagram layout: a 10-byte header then a payload fragment.
//!   byte 0      message type (`MSG_*`)
//!   byte 1      wire version (`WIRE_VERSION`)
//!   bytes 2..6  sequence number, u32 LE
//!   bytes 6..8  fragment index,  u16 LE
//!   bytes 8..10 fragment count,  u16 LE
//!
//! One round-trip steps every arena:
//!   Python -> Rust: `MSG_ACTION` = `num_slots * ACTION_FLOATS_PER_SLOT` f32
//!   Rust -> Python: `MSG_STATE`  = `num_slots * obs_floats_per_slot`   f32
//! `num_slots = num_arenas * players_per_arena` (`= 2 * team_size`), and
//! `obs_floats_per_slot` depends on the kit's observation config - both are
//! sent in the `Hello`.

use crate::config::{cfg, SimConfig};
use crate::effects::EFFECT_COUNT;
use crate::kit::{HOTBAR_ACTION_DIM, HOTBAR_SLOTS, ITEM_COUNT};
use serde::{Deserialize, Serialize};

/// Action for a single player for a single step.
///
/// Wire form (10 f32, this exact order):
///   `[move_x, move_z, yaw_delta, pitch_delta, jump, attack, sprint,
///     use_item, sneak, held_slot]`
/// The five flags are 0.0/1.0. `held_slot` is rounded and clamped to
/// `0..HOTBAR_ACTION_DIM`: `0..9` selects a physical hotbar slot (key 1-9),
/// `9..21` hotkeys `kit::Item` id `(held_slot - 9)` into the selected slot
/// (vanilla number-key swap; `9` = clear the slot to an empty hand). What a
/// selected slot holds comes from the kit's `Loadout::hotbar` layout.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(from = "[f32; 10]")]
pub struct Action {
    pub move_x: f32,
    pub move_z: f32,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub jump: bool,
    pub attack: bool,
    pub sprint: bool,
    /// Raise the shield / draw the bow / eat / place - dispatched by the
    /// held item (see `arena::apply_input`).
    pub use_item: bool,
    /// Crouch: ~0.3x move speed, a 1.5-block hitbox, and no walking off the
    /// edge of a block / the platform.
    pub sneak: bool,
    /// Held-slot action this step (`0..HOTBAR_ACTION_DIM`): select a
    /// physical slot, or hotkey an item into the selected slot. Any hotbar
    /// op on the same tick as `attack` triggers attribute swapping under
    /// the `legacy` input order, or a one-tick attack/use lockout under
    /// `modern`.
    pub slot: usize,
}

impl Action {
    #[allow(dead_code)] // used by arena.rs tests
    pub const NOOP: Action = Action {
        move_x: 0.0,
        move_z: 0.0,
        yaw_delta: 0.0,
        pitch_delta: 0.0,
        jump: false,
        attack: false,
        sprint: false,
        use_item: false,
        sneak: false,
        slot: 0,
    };
}

impl From<[f32; 10]> for Action {
    fn from(a: [f32; 10]) -> Self {
        Action::from_wire(&a)
    }
}

/// Bumped whenever the on-wire layout changes incompatibly.
/// v3: kits - `held_slot` action, kit/consumable/projectile observation.
/// v4: block grid - `inventory` counts and the `block_view` column grid
///     replace the old effect-field observation.
/// v5: physical hotbar - `held_slot` action is a slot (`0..HOTBAR_SLOTS`),
///     not an `Item`; observation gains `self_slot`, `self_swap_lockout`
///     and the `hotbar` slot->item layout block.
/// v6: sneak + hunger - new `sneak` action flag (10 action floats now);
///     observation gains `self_food`, `self_sneaking` and per-other
///     `sneaking`.
/// v8: mining - observation gains `self_mining` (0..1 break progress on the
///     block the `uhc` pickaxe is aimed at; 0 when not mining). One extra
///     self float. No new action - mining reuses `attack`.
/// v7: vanilla-fidelity pass - **no field added or removed**, but the
///     *meaning* of several changes: the world is now a voxel grid so
///     `self_ground_height` / `self_slope_*` / per-other `ground_height` /
///     `block_view.top_rel` are integer block tops (and include placed
///     blocks); `self_hurt` counts down over 1.0 s (vanilla 20-tick
///     i-frames), not 0.5 s; other players / arrows are observed with the
///     observer's ping of latency; and the `yaw_delta` / `pitch_delta`
///     action head is rescaled by the larger `max_look_delta` (3.0). A v6
///     policy would decode a v7 stream without erroring and silently
///     regress - hence the bump.
/// v9: splash potions - 5 new `kit::Item`s (widening `inventory` and the
///     `held_slot` hotkey action) and a new `self_effects` block (one float
///     per `effects::Effect`, `amplifier + 1` while active, else 0).
pub const WIRE_VERSION: u8 = 9;

pub const MSG_HELLO_REQ: u8 = 1;
pub const MSG_HELLO_RESP: u8 = 2;
pub const MSG_ACTION: u8 = 3;
pub const MSG_STATE: u8 = 4;

pub const HEADER_LEN: usize = 10;
pub const MAX_PAYLOAD: usize = 60_000;

pub const ACTION_FLOATS_PER_SLOT: usize = 10;

/// Fixed leading "self" fields of a wire observation row (the 27 scalars in
/// `write_wire` plus the `self_effects` block).
pub const OBS_SELF_FLOATS: usize = 27 + OBS_EFFECT_FLOATS;
/// One float per `effects::Effect`: `amplifier + 1` while active, else 0.
pub const OBS_EFFECT_FLOATS: usize = EFFECT_COUNT;
/// Per-item inventory counts (all `kit::Item`s except `Empty`).
pub const OBS_INVENTORY_FLOATS: usize = ITEM_COUNT - 1;
/// The `Item` id sitting in each physical hotbar slot.
pub const OBS_HOTBAR_FLOATS: usize = HOTBAR_SLOTS;
/// Floats in one enemy / teammate block.
pub const OBS_OTHER_FLOATS: usize = 13;
/// Floats in one in-flight projectile block.
pub const OBS_PROJ_FLOATS: usize = 7;
/// Floats in one block-grid column: `[top_rel_height, water, lava, cobweb]`.
pub const OBS_BLOCKCOL_FLOATS: usize = 4;
/// Fixed trailing fields: 3 arena-global scalars + 7 event fields.
pub const OBS_GLOBAL_FLOATS: usize = 3;
pub const OBS_EVENT_FLOATS: usize = 7;

/// Width of one observation row on the wire for the given config.
pub fn obs_floats_per_slot(config: &SimConfig) -> usize {
    OBS_SELF_FLOATS
        + OBS_INVENTORY_FLOATS
        + OBS_HOTBAR_FLOATS
        + OBS_OTHER_FLOATS * (config.max_observed_enemies + config.max_observed_teammates)
        + OBS_PROJ_FLOATS * config.max_observed_projectiles
        + OBS_BLOCKCOL_FLOATS * config.block_view_size * config.block_view_size
        + OBS_GLOBAL_FLOATS
        + OBS_EVENT_FLOATS
}

impl Action {
    pub fn from_wire(f: &[f32]) -> Self {
        let slot = if f.len() > 9 {
            (f[9].round() as i64).clamp(0, HOTBAR_ACTION_DIM as i64 - 1) as usize
        } else {
            0
        };
        Action {
            move_x: f[0],
            move_z: f[1],
            yaw_delta: f[2],
            pitch_delta: f[3],
            jump: f[4] != 0.0,
            attack: f[5] != 0.0,
            sprint: f[6] != 0.0,
            use_item: f[7] != 0.0,
            sneak: f.get(8).copied().unwrap_or(0.0) != 0.0,
            slot,
        }
    }
}

/// One observed nearby player (enemy or teammate), observer-relative.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct OtherPlayer {
    pub present: bool,
    /// HP + absorption.
    pub hp: f32,
    pub rel_x: f32,
    pub rel_y: f32,
    pub rel_z: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub vel_z: f32,
    pub ground_height: f32,
    /// 1.0 = actively blocking; below 1.0 = a shield still being raised.
    pub blocking: f32,
    pub eating: f32,
    /// 1.0 if this player is holding a bow or crossbow.
    pub held_ranged: f32,
    /// 1.0 if this player is crouched (shorter hitbox, edge-guarding).
    pub sneaking: f32,
}

/// One observed in-flight arrow / bolt, observer-relative.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ProjectileObs {
    pub present: bool,
    pub rel_x: f32,
    pub rel_y: f32,
    pub rel_z: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub vel_z: f32,
}

/// One column of the yaw-rotated block-grid view around the observer.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct BlockColumnObs {
    /// Surface height (terrain or top solid block) minus the observer's y.
    pub top_rel: f32,
    pub water: f32,
    pub lava: f32,
    pub cobweb: f32,
}

/// Observation for a single player, expressed relative to that player.
#[derive(Debug, Clone, Serialize)]
pub struct Observation {
    pub self_hp: f32,
    pub self_vel_x: f32,
    pub self_vel_y: f32,
    pub self_vel_z: f32,
    pub self_yaw: f32,
    pub self_pitch: f32,
    pub self_on_ground: bool,
    /// Vanilla "attack strength scale" for the *held* weapon (0 = just
    /// swung, 1 = recharged).
    pub self_attack_cooldown: f32,
    pub self_ping_ms: f32,
    /// Shield-raised fraction (0 = down, 1 = blocking). 0 if no shield / disabled.
    pub self_shield: f32,
    pub self_dist_from_center: f32,
    pub self_ground_height: f32,
    pub self_slope_forward: f32,
    pub self_slope_right: f32,
    /// Remaining hurt-invulnerability fraction (0..1).
    pub self_hurt: f32,
    /// Held `kit::Item` index.
    pub self_held: f32,
    /// Food level 0..20.
    pub self_food: f32,
    /// 1.0 if crouched.
    pub self_sneaking: f32,
    /// Absorption ("extra hearts") on top of HP.
    pub self_absorption: f32,
    /// Eating progress 0..1.
    pub self_eating: f32,
    /// Bow draw 0..1.
    pub self_bow_draw: f32,
    /// 1.0 if on fire.
    pub self_burning: f32,
    /// Remaining shield-disable fraction 0..1 (axe hit).
    pub self_shield_disabled: f32,
    /// Loose arrows left.
    pub self_arrows: f32,
    /// Currently selected physical hotbar slot (`0..HOTBAR_SLOTS`).
    pub self_slot: f32,
    /// Remaining post-swap attack/use lockout fraction 0..1. Always 0 under
    /// the `legacy` input order.
    pub self_swap_lockout: f32,
    /// Break progress 0..1 on the block the pickaxe is mining (0 when not
    /// mining). `uhc` only - always 0 for the other kits.
    pub self_mining: f32,

    /// One float per `effects::Effect` in discriminant order: `amplifier + 1`
    /// while that effect is active, else 0. All zero unless a config hands
    /// out splash potions.
    pub self_effects: Vec<f32>,

    /// Per-item counts held (all `kit::Item`s except `Empty`, in order),
    /// so the policy can learn which items it can actually switch to.
    pub inventory: Vec<f32>,
    /// The `kit::Item` id in each physical hotbar slot (the kit's layout).
    pub hotbar: Vec<f32>,

    pub enemies: Vec<OtherPlayer>,
    pub teammates: Vec<OtherPlayer>,
    pub projectiles: Vec<ProjectileObs>,
    /// The yaw-rotated `block_view_size` x `block_view_size` column grid
    /// around the player (row-major, front-left first). Empty when
    /// `block_view_size == 0`.
    pub block_view: Vec<BlockColumnObs>,

    pub time_left: f32,
    pub enemies_alive: f32,
    pub teammates_alive: f32,

    pub reward: f32,
    pub damage_dealt: f32,
    pub damage_taken: f32,
    pub swept: bool,
    pub won: bool,
    pub lost: bool,
    pub done: bool,
}

impl Observation {
    /// Appends this observation's wire row to `out`. Field order is
    /// load-bearing: `features.py` must match it exactly.
    pub fn write_wire(&self, out: &mut Vec<f32>) {
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        out.extend_from_slice(&[
            self.self_hp,
            self.self_vel_x,
            self.self_vel_y,
            self.self_vel_z,
            self.self_yaw,
            self.self_pitch,
            b(self.self_on_ground),
            self.self_attack_cooldown,
            self.self_ping_ms,
            self.self_shield,
            self.self_dist_from_center,
            self.self_ground_height,
            self.self_slope_forward,
            self.self_slope_right,
            self.self_hurt,
            self.self_held,
            self.self_absorption,
            self.self_eating,
            self.self_bow_draw,
            self.self_burning,
            self.self_shield_disabled,
            self.self_arrows,
            self.self_slot,
            self.self_swap_lockout,
            self.self_food,
            self.self_sneaking,
            self.self_mining,
        ]);
        debug_assert_eq!(self.self_effects.len(), OBS_EFFECT_FLOATS);
        out.extend_from_slice(&self.self_effects);
        debug_assert_eq!(self.inventory.len(), OBS_INVENTORY_FLOATS);
        out.extend_from_slice(&self.inventory);
        debug_assert_eq!(self.hotbar.len(), OBS_HOTBAR_FLOATS);
        out.extend_from_slice(&self.hotbar);
        debug_assert_eq!(self.enemies.len(), cfg().max_observed_enemies);
        debug_assert_eq!(self.teammates.len(), cfg().max_observed_teammates);
        for o in self.enemies.iter().chain(self.teammates.iter()) {
            out.extend_from_slice(&[
                b(o.present),
                o.hp,
                o.rel_x,
                o.rel_y,
                o.rel_z,
                o.vel_x,
                o.vel_y,
                o.vel_z,
                o.ground_height,
                o.blocking,
                o.eating,
                o.held_ranged,
                o.sneaking,
            ]);
        }
        debug_assert_eq!(self.projectiles.len(), cfg().max_observed_projectiles);
        for p in &self.projectiles {
            out.extend_from_slice(&[
                b(p.present),
                p.rel_x,
                p.rel_y,
                p.rel_z,
                p.vel_x,
                p.vel_y,
                p.vel_z,
            ]);
        }
        debug_assert_eq!(
            self.block_view.len(),
            cfg().block_view_size * cfg().block_view_size
        );
        for col in &self.block_view {
            out.extend_from_slice(&[col.top_rel, col.water, col.lava, col.cobweb]);
        }
        out.extend_from_slice(&[
            self.time_left,
            self.enemies_alive,
            self.teammates_alive,
            self.reward,
            self.damage_dealt,
            self.damage_taken,
            b(self.swept),
            b(self.won),
            b(self.lost),
            b(self.done),
        ]);
    }
}

/// Handshake sent once by Rust right after the first client hello.
#[derive(Debug, Clone, Serialize)]
pub struct Hello {
    pub num_arenas: usize,
    pub players_per_arena: usize,
    pub team_size: usize,
    pub kit: String,
    pub attribute_swapping: bool,
    pub natural_regen: bool,
    pub friendly_fire: bool,
    pub max_observed_enemies: usize,
    pub max_observed_teammates: usize,
    pub max_observed_projectiles: usize,
    pub block_view_size: usize,
    /// Width of the `inventory` observation block (`kit::ITEM_COUNT`).
    pub item_count: usize,
    /// Physical hotbar slots = the `hotbar` observation block width
    /// (`kit::HOTBAR_SLOTS`).
    pub hotbar_slots: usize,
    /// Held-slot categorical action head width (`kit::HOTBAR_ACTION_DIM` =
    /// `HOTBAR_SLOTS` selects + `ITEM_COUNT` hotkeys).
    pub hotbar_action_dim: usize,
    /// Width of the `self_effects` observation block (`effects::EFFECT_COUNT`).
    pub effect_count: usize,
    pub obs_floats_per_slot: usize,
    pub action_floats_per_slot: usize,
    pub tick_dt: f32,
    pub max_hp: f32,
    pub arena_radius: f32,
    pub match_time_seconds: f32,
    pub terrain_max_amplitude: f32,
    pub max_look_delta: f32,
    pub max_ping_ms: f32,
    pub config: SimConfig,
}
