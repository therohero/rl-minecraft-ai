//! One player entity and its whole per-tick self-update: input handling
//! (look, hotbar, item use), movement + physics integration, the hunger /
//! saturation / exhaustion economy, environmental effects (cobweb / water /
//! lava), consumables, and damage application. `Arena::step` drives these in
//! order; melee attacks and projectiles live in `combat.rs` / `projectile.rs`.

use std::collections::VecDeque;

use crate::blocks::{BlockWorld, Cell};
use crate::collision;
use crate::combat::{self, Armor, TargetState};
use crate::config::{cfg, InputOrder};
use crate::kit::{Item, Loadout, HOTBAR_SLOTS, ITEM_COUNT};
use crate::physics::*;
use crate::projectile::PendingShot;
use crate::protocol::Action;
use crate::terrain::Terrain;

/// How many past ticks of public state each player / projectile retains, so
/// an observer with a high ping sees others where they *were*, not where they
/// are (vanilla is client-authoritative for your own movement - only your
/// view of everyone else lags).
pub(crate) const MAX_DELAY_TICKS: usize = 8;

/// The slice of a player's state another player can observe. Snapshotted
/// every tick into `Player::history` so a laggy observer reads a stale copy.
#[derive(Clone, Copy)]
pub(crate) struct PublicSnapshot {
    pub pos: Vec3,
    pub vel: Vec3,
    /// Hitbox height at that tick (standing vs. crouched) - a laggy attacker
    /// raycasts against this stale box, not the target's current one.
    pub height: f32,
    pub hp: f32,
    pub absorption: f32,
    /// 1.0 = shield actively blocking, else the raise fraction.
    pub shield: f32,
    pub eating: f32,
    pub held_ranged: f32,
    pub sneaking: f32,
}

/// Look `ping_ms` of latency back into a per-tick history ring and return the
/// index to read (`0` = current). Clamped to what the ring actually holds.
pub(crate) fn delay_back(len: usize, ping_ms: f32) -> usize {
    let ticks = ((ping_ms / 1000.0) / DT).round() as usize;
    ticks.min(MAX_DELAY_TICKS).min(len.saturating_sub(1))
}
/// A "fully charged" attack timer value, larger than any weapon recharge.
pub(crate) const FULLY_CHARGED: f32 = 10.0;
/// Vanilla food bar maximum, and the saturation a player spawns with.
pub(crate) const MAX_FOOD: f32 = 20.0;
pub(crate) const START_SATURATION: f32 = 5.0;

#[derive(Clone)]
pub struct Player {
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub hp: f32,
    /// Absorption ("extra hearts") on top of `hp`, soaked before HP.
    pub absorption: f32,
    pub on_ground: bool,
    /// 0 = team A, 1 = team B.
    pub team: u8,

    // --- inventory / kit ---
    /// The kit's slot->item layout (index 0 = key 1).
    pub hotbar: [Item; HOTBAR_SLOTS],
    /// Currently selected physical hotbar slot.
    pub slot: usize,
    /// `hotbar[slot]` - what's actually in hand.
    pub held: Item,
    pub prev_held: Item,
    /// The held slot changed on the same tick as this step's action (feeds
    /// attribute swapping - see `cfg().attribute_swapping`).
    pub held_changed_this_step: bool,
    pub counts: [u32; ITEM_COUNT],
    pub arrows: u32,
    pub has_shield: bool,
    pub crossbow_loaded: bool,
    pub sharpness: u32,
    pub power: u32,
    pub piercing: u32,
    /// Pickaxe Efficiency level (`uhc` = 3) - speeds up mining placed blocks.
    pub efficiency: u32,
    pub armor: Armor,

    // --- mining (uhc pickaxe) ---
    /// The placed-block cell the player is currently breaking, if any. Clears
    /// when they stop holding `attack`, look away, or switch off the pickaxe.
    pub(crate) mining_cell: Option<Cell>,
    /// Progress on `mining_cell`, 0..1. Reaches 1.0 -> the block breaks.
    pub mining_progress: f32,

    // --- combat timers ---
    pub time_since_last_attack: f32,
    pub time_since_damage: f32,
    pub sprinting: bool,
    pub sneaking: bool,
    /// `jump` held this tick - `integrate` reads it for swim-up.
    jump_held: bool,
    pub shield_time: f32,
    pub shield_disabled_time: f32,
    pub hurt_time_left: f32,
    pub last_damage: f32,
    pub fall_distance: f32,
    /// Horizontal distance moved last tick (vanilla `walkDist` delta) - the
    /// sweep-attack "nearly stationary" check reads this, not current speed.
    pub dist_moved_last_tick: f32,
    pub burn_time_left: f32,
    burn_accum: f32,

    // --- hunger ---
    pub food: f32,
    pub saturation: f32,
    exhaustion: f32,
    regen_accum: f32,
    starve_accum: f32,

    // --- item-use state ---
    pub bow_draw: f32,
    pub crossbow_load: f32,
    pub eat_progress: f32,
    pub place_cooldown: f32,
    /// Remaining post-swap attack/use lockout (seconds). Only set under the
    /// `modern` input order; see `InputOrder`.
    pub swap_lockout: f32,
    pub regen_time_left: f32,
    regen_rate: f32,
    prev_use_item: bool,
    pub(crate) pending_shot: Option<PendingShot>,
    pub(crate) pending_place: Option<Item>,
    /// A splash potion the policy released this tick, thrown by `Arena::step`.
    pub(crate) pending_throw: Option<Item>,

    /// Active potion / status effects (see `effects.rs`).
    pub effects: crate::effects::StatusEffects,

    /// This match's baseline connection latency (ms), rolled once per match.
    pub(crate) base_ping_ms: f32,
    /// `base_ping_ms` plus this tick's small jitter - what the observation
    /// reports and what other observers use to age their view of this player.
    pub ping_ms: f32,
    /// Ring of the last `MAX_DELAY_TICKS + 1` public snapshots (newest last).
    pub(crate) history: VecDeque<PublicSnapshot>,
}

impl Player {
    /// A fresh player for `team`, with placeholder state - `respawn` fills in
    /// the position, orientation and kit for each match.
    pub(crate) fn new(team: u8) -> Player {
        Player {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            hp: combat::MAX_HP,
            absorption: 0.0,
            on_ground: true,
            team,
            hotbar: [Item::Empty; HOTBAR_SLOTS],
            slot: 0,
            held: Item::Sword,
            prev_held: Item::Sword,
            held_changed_this_step: false,
            counts: [0; ITEM_COUNT],
            arrows: 0,
            has_shield: false,
            crossbow_loaded: false,
            sharpness: 0,
            power: 0,
            piercing: 0,
            efficiency: 0,
            armor: Armor { points: 20.0, toughness: 8.0, protection_epf: 14.0 },
            mining_cell: None,
            mining_progress: 0.0,
            time_since_last_attack: FULLY_CHARGED,
            time_since_damage: FULLY_CHARGED,
            sprinting: false,
            sneaking: false,
            jump_held: false,
            shield_time: 0.0,
            shield_disabled_time: 0.0,
            hurt_time_left: 0.0,
            last_damage: 0.0,
            fall_distance: 0.0,
            dist_moved_last_tick: 0.0,
            burn_time_left: 0.0,
            burn_accum: 0.0,
            food: MAX_FOOD,
            saturation: START_SATURATION,
            exhaustion: 0.0,
            regen_accum: 0.0,
            starve_accum: 0.0,
            bow_draw: 0.0,
            crossbow_load: 0.0,
            eat_progress: 0.0,
            place_cooldown: 0.0,
            swap_lockout: 0.0,
            regen_time_left: 0.0,
            regen_rate: 0.0,
            prev_use_item: false,
            pending_shot: None,
            pending_place: None,
            pending_throw: None,
            effects: crate::effects::StatusEffects::default(),
            base_ping_ms: cfg().min_ping_ms,
            ping_ms: cfg().min_ping_ms,
            history: VecDeque::new(),
        }
    }

    /// Reset every per-match field: drop the player at `pos` facing `yaw`
    /// with a full HP bar and the kit `l`.
    pub(crate) fn respawn(&mut self, pos: Vec3, yaw: f32, l: &Loadout) {
        self.pos = pos;
        self.yaw = yaw;
        self.pitch = 0.0;
        self.vel = Vec3::ZERO;
        self.hp = combat::MAX_HP;
        self.absorption = 0.0;
        self.on_ground = true;
        self.time_since_last_attack = FULLY_CHARGED;
        self.time_since_damage = FULLY_CHARGED;
        self.sprinting = false;
        self.sneaking = false;
        self.jump_held = false;
        self.food = MAX_FOOD;
        self.saturation = START_SATURATION;
        self.exhaustion = 0.0;
        self.regen_accum = 0.0;
        self.starve_accum = 0.0;
        self.dist_moved_last_tick = 0.0;
        self.shield_time = 0.0;
        self.shield_disabled_time = 0.0;
        self.hurt_time_left = 0.0;
        self.last_damage = 0.0;
        self.fall_distance = 0.0;
        self.burn_time_left = 0.0;
        self.burn_accum = 0.0;
        self.bow_draw = 0.0;
        self.crossbow_load = 0.0;
        self.eat_progress = 0.0;
        self.place_cooldown = 0.0;
        self.swap_lockout = 0.0;
        self.regen_time_left = 0.0;
        self.regen_rate = 0.0;
        self.prev_use_item = false;
        self.pending_shot = None;
        self.pending_place = None;
        self.pending_throw = None;
        self.effects.clear();
        self.history.clear();
        // Kit-derived state.
        self.hotbar = l.hotbar;
        self.slot = l.default_slot;
        self.held = l.default_held;
        self.prev_held = l.default_held;
        self.held_changed_this_step = false;
        self.counts = l.counts;
        self.arrows = l.arrows;
        self.has_shield = l.has_shield;
        self.crossbow_loaded = l.crossbow_preloaded;
        self.sharpness = l.sharpness;
        self.power = l.power;
        self.piercing = l.piercing;
        self.efficiency = l.efficiency;
        self.mining_cell = None;
        self.mining_progress = 0.0;
        self.armor = Armor {
            points: l.armor_points,
            toughness: l.armor_toughness,
            protection_epf: l.protection_epf,
        };
    }

    pub(crate) fn alive(&self) -> bool {
        self.hp > 0.0
    }

    pub(crate) fn shield_available(&self) -> bool {
        self.has_shield && self.shield_disabled_time <= 0.0
    }

    /// Shield actually blocking right now (in hand long enough, not disabled).
    pub(crate) fn shield_up(&self) -> bool {
        self.shield_available() && self.shield_time >= cfg().combat.shield_raise_seconds
    }

    pub(crate) fn shield_fraction(&self) -> f32 {
        if !self.shield_available() {
            return 0.0;
        }
        (self.shield_time / cfg().combat.shield_raise_seconds).clamp(0.0, 1.0)
    }

    pub(crate) fn hurt_fraction(&self) -> f32 {
        let w = cfg().combat.hurt_invulnerability_seconds;
        if w > 0.0 {
            (self.hurt_time_left / w).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub(crate) fn eating(&self) -> bool {
        self.eat_progress > 0.0
    }

    /// Eating progress 0..1 (0 for a non-food held item).
    pub(crate) fn eat_fraction(&self) -> f32 {
        (self.eat_progress / eat_seconds(self.held).max(1e-3)).clamp(0.0, 1.0)
    }

    pub(crate) fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos())
    }

    pub(crate) fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    #[cfg(test)]
    pub(crate) fn horizontal_speed(&self) -> f32 {
        (self.vel.x * self.vel.x + self.vel.z * self.vel.z).sqrt()
    }

    pub(crate) fn hitbox_height(&self) -> f32 {
        if self.sneaking {
            cfg().combat.sneak_hitbox_height
        } else {
            PLAYER_HEIGHT
        }
    }

    pub(crate) fn held_weapon_recharge(&self) -> f32 {
        combat::weapon_for(self.held, self.held, self.sharpness).recharge_seconds
    }

    /// The read-only slice of state a melee/projectile hit resolves against.
    pub(crate) fn target_state(&self) -> TargetState {
        TargetState {
            pos: self.pos,
            yaw: self.yaw,
            shield_up: self.shield_up(),
            hurt_time_left: self.hurt_time_left,
            last_damage: self.last_damage,
            armor: self.armor,
        }
    }

    /// Apply `applied` damage (post-armor, post-shield, post-i-frames):
    /// absorption first, then HP. `refresh` opens (resets) the 20-tick
    /// hurt-invulnerability window - vanilla only does this on a full hit,
    /// not on a partial hit landed during the window's first half.
    pub(crate) fn apply_hit(&mut self, applied: f32, new_last: f32, refresh: bool) {
        let mut dmg = applied;
        if self.absorption > 0.0 {
            let a = dmg.min(self.absorption);
            self.absorption -= a;
            dmg -= a;
        }
        self.hp = (self.hp - dmg).max(0.0);
        if refresh {
            self.hurt_time_left = cfg().combat.hurt_invulnerability_seconds;
        }
        self.last_damage = new_last;
        self.time_since_damage = 0.0;
    }

    /// Run `raw_after_armor` through the i-frame / last-damage rule and apply
    /// what survives. Returns the HP actually lost.
    pub(crate) fn take_damage(&mut self, raw_after_armor: f32) -> f32 {
        let (applied, new_last, refresh) =
            combat::apply_iframes(raw_after_armor, self.hurt_time_left, self.last_damage);
        if applied > 0.0 {
            self.apply_hit(applied, new_last, refresh);
        }
        applied
    }
}

/// Vanilla exhaustion: at `exhaustion_per_unit` accumulated, spend 1
/// saturation, or 1 food once saturation is gone.
pub(crate) fn add_exhaustion(p: &mut Player, amount: f32) {
    let c = &cfg().combat;
    p.exhaustion += amount;
    while p.exhaustion >= c.exhaustion_per_unit {
        p.exhaustion -= c.exhaustion_per_unit;
        if p.saturation > 0.0 {
            p.saturation = (p.saturation - 1.0).max(0.0);
        } else {
            p.food = (p.food - 1.0).max(0.0);
        }
    }
}

fn eat_seconds(item: Item) -> f32 {
    match item {
        Item::GoldenApple => cfg().combat.golden_apple_eat_seconds,
        Item::GoldenHead => cfg().combat.golden_head_eat_seconds,
        _ => f32::INFINITY,
    }
}

/// Golden apple / head finishes. Vanilla-like: **no instant heal** - the HP
/// comes back over time via Regeneration - plus an immediate absorption
/// bump and the food/saturation any apple restores. The golden head's
/// Regen ticks at twice the apple's rate (`golden_head_regen_rate`).
fn finish_eating(p: &mut Player) {
    let c = &cfg().combat;
    let slot = p.held.index();
    if p.counts[slot] == 0 {
        p.eat_progress = 0.0;
        return;
    }
    p.counts[slot] -= 1;
    let (absorb, regen_s, regen_r, food, sat) = match p.held {
        Item::GoldenHead => (
            c.golden_head_absorption,
            c.golden_head_regen_seconds,
            c.golden_head_regen_rate,
            c.golden_head_food,
            c.golden_head_saturation,
        ),
        _ => (
            c.golden_apple_absorption,
            c.golden_apple_regen_seconds,
            c.golden_apple_regen_rate,
            c.golden_apple_food,
            c.golden_apple_saturation,
        ),
    };
    p.absorption = p.absorption.max(absorb);
    p.regen_time_left = p.regen_time_left.max(regen_s);
    p.regen_rate = p.regen_rate.max(regen_r);
    p.food = (p.food + food).min(MAX_FOOD);
    p.saturation = (p.saturation + sat).min(p.food);
    p.regen_accum = 0.0;
    p.eat_progress = 0.0;
    if p.counts[slot] == 0 {
        // Out of this food - the client would auto-switch to the next slot;
        // fall back to the sword slot if the kit has one, else an empty slot.
        p.slot = fallback_slot(p);
        p.held = p.hotbar[p.slot];
        p.prev_held = p.held;
    }
}

/// Where the selected slot lands when the held item is used up: the sword
/// slot if the kit still has a sword, otherwise the first empty slot,
/// otherwise slot 0.
fn fallback_slot(p: &Player) -> usize {
    if p.counts[Item::Sword.index()] > 0 {
        if let Some(i) = p.hotbar.iter().position(|&it| it == Item::Sword) {
            return i;
        }
    }
    p.hotbar.iter().position(|&it| it == Item::Empty).unwrap_or(0)
}

/// One tick of input: look, hotbar select / hotkey, item use (shield / bow /
/// crossbow / eat / place), then movement intent (walk / sprint / sneak /
/// jump). Queued shots / placements land in `pending_shot` / `pending_place`
/// for `Arena::step` to act on.
pub(crate) fn apply_input(p: &mut Player, action: &Action) {
    let c = &cfg().combat;
    let max_look = cfg().max_look_delta;
    p.yaw += clamp(action.yaw_delta, -max_look, max_look);
    p.pitch = clamp(p.pitch + clamp(action.pitch_delta, -max_look, max_look), -MAX_PITCH, MAX_PITCH);

    // Decay per-step timers.
    if p.shield_disabled_time > 0.0 {
        p.shield_disabled_time = (p.shield_disabled_time - DT).max(0.0);
    }
    if p.place_cooldown > 0.0 {
        p.place_cooldown = (p.place_cooldown - DT).max(0.0);
    }
    if p.swap_lockout > 0.0 {
        p.swap_lockout = (p.swap_lockout - DT).max(0.0);
    }

    apply_hotbar_action(p, action.slot);

    // --- what does "use item" do, given the held item? ---
    // A fresh slot switch interrupts item use for that tick (the client
    // released the button on the swap); it has to rise again next tick. The
    // modern input order additionally locks out use while `swap_lockout`
    // runs (and attack too, gated in `step`).
    let use_now = action.use_item && p.swap_lockout <= 0.0 && !p.held_changed_this_step;
    let rising = use_now && !p.prev_use_item;
    let mut shielding = false;
    p.pending_shot = None;
    p.pending_place = None;
    p.pending_throw = None;

    if p.held.is_ranged() {
        p.eat_progress = 0.0;
        match p.held {
            Item::Bow => {
                if use_now {
                    p.bow_draw = (p.bow_draw + DT).min(c.bow_max_draw_seconds);
                } else {
                    // Release: fire if charged past the minimum and we have
                    // ammo. Vanilla `BowItem::getPowerForTime`: with the draw
                    // time `t` in seconds, `charge = (t^2 + 2t) / 3`, clamped
                    // to 1 - a convex ramp, not the old linear one.
                    let t = p.bow_draw;
                    let charge = ((t * t + 2.0 * t) / 3.0).min(1.0);
                    if charge >= c.bow_min_draw_fraction && p.arrows > 0 {
                        // Vanilla arrow damage is `ceil(speed * base)` at
                        // impact; Power adds `level*0.5 + 0.5` to `base`.
                        let base = c.arrow_damage_per_speed
                            + if p.power > 0 {
                                p.power as f32 * c.power_per_level + 0.5
                            } else {
                                0.0
                            };
                        p.pending_shot = Some(PendingShot {
                            speed: c.arrow_max_speed * charge,
                            base_damage: base,
                            piercing: 0,
                            crossbow: false,
                        });
                    }
                    p.bow_draw = 0.0;
                }
                p.crossbow_load = 0.0;
            }
            Item::Crossbow => {
                p.bow_draw = 0.0;
                if p.crossbow_loaded {
                    if use_now {
                        p.pending_shot = Some(PendingShot {
                            speed: c.crossbow_arrow_speed,
                            base_damage: c.arrow_damage_per_speed,
                            piercing: (c.piercing_per_level * p.piercing) as i32,
                            crossbow: true,
                        });
                        p.crossbow_load = 0.0;
                    }
                } else if use_now && p.arrows > 0 {
                    p.crossbow_load += DT;
                    if p.crossbow_load >= c.crossbow_load_seconds {
                        p.crossbow_loaded = true;
                        p.arrows -= 1;
                        p.crossbow_load = 0.0;
                    }
                } else {
                    p.crossbow_load = 0.0;
                }
            }
            _ => {}
        }
    } else if p.held.is_food() {
        p.bow_draw = 0.0;
        p.crossbow_load = 0.0;
        if use_now && p.counts[p.held.index()] > 0 {
            p.eat_progress += DT;
            if p.eat_progress >= eat_seconds(p.held) {
                finish_eating(p);
            }
        } else {
            p.eat_progress = 0.0;
        }
    } else if p.held.is_placeable() {
        p.bow_draw = 0.0;
        p.crossbow_load = 0.0;
        p.eat_progress = 0.0;
        if rising {
            p.pending_place = Some(p.held);
        }
    } else if p.held.is_splash_potion() {
        p.bow_draw = 0.0;
        p.crossbow_load = 0.0;
        p.eat_progress = 0.0;
        if rising && p.counts[p.held.index()] > 0 {
            p.pending_throw = Some(p.held);
        }
    } else {
        // Melee weapon / empty hand -> use the off-hand shield if we have one.
        p.bow_draw = 0.0;
        p.crossbow_load = 0.0;
        p.eat_progress = 0.0;
        if use_now && p.shield_available() {
            p.shield_time += DT;
            shielding = p.shield_up();
        } else {
            p.shield_time = 0.0;
        }
    }
    p.prev_use_item = use_now;

    // --- movement ---
    p.jump_held = action.jump;
    p.sneaking = action.sneak;

    let mut move_x = clamp(action.move_x, -1.0, 1.0);
    let mut move_z = clamp(action.move_z, -1.0, 1.0);
    let input_len = (move_x * move_x + move_z * move_z).sqrt();
    if input_len > 1.0 {
        move_x /= input_len;
        move_z /= input_len;
    }
    // No sprinting while shielding, eating, drawing a bow, crouched, or too
    // hungry to start (vanilla: food <= 6).
    let busy = shielding || p.eating() || p.bow_draw > 0.0 || p.sneaking;
    p.sprinting = action.sprint
        && !busy
        && move_z > cfg().sprint_forward_threshold
        && p.food > c.min_food_to_sprint;

    let mut accel = if p.on_ground {
        if p.sprinting {
            SPRINT_ACCEL_PER_TICK
        } else {
            WALK_ACCEL_PER_TICK
        }
    } else {
        AIR_ACCEL_PER_TICK
    };
    if shielding {
        accel *= c.shield_move_multiplier;
    } else if p.eating() || p.bow_draw > 0.0 {
        accel *= 0.3; // vanilla slows you while using an item
    } else if p.sneaking {
        accel *= c.sneak_speed_multiplier;
    }
    // Speed / Slowness scale the movement attribute (vanilla): +20% / -15%
    // per level, so the drag-equilibrium top speed shifts with it.
    accel *= p.effects.move_multiplier();
    let fwd = p.forward();
    let right = p.right();
    p.vel.x += (fwd.x * move_z + right.x * move_x) * accel;
    p.vel.z += (fwd.z * move_z + right.z * move_x) * accel;

    if p.on_ground && action.jump {
        p.vel.y = JUMP_VELOCITY_PER_TICK;
        p.on_ground = false;
        add_exhaustion(p, if p.sprinting { c.sprint_jump_exhaustion } else { c.jump_exhaustion });
        if p.sprinting {
            p.vel.x += fwd.x * SPRINT_JUMP_BOOST;
            p.vel.z += fwd.z * SPRINT_JUMP_BOOST;
        }
    }

    p.time_since_last_attack += DT;
}

/// The held-slot action: `0..HOTBAR_SLOTS` selects a physical slot;
/// `HOTBAR_SLOTS..` hotkeys `Item` id `(slot - HOTBAR_SLOTS)` into the
/// selected slot (one hotbar op per tick). A hotkey'd item only needs to be
/// owned (`counts > 0`); a displaced item stays in `counts` off-hotbar, so
/// the policy can swap it back and still sees it in the `inventory` block.
fn apply_hotbar_action(p: &mut Player, slot_action: usize) {
    let c = &cfg().combat;
    let modern = cfg().input_order == InputOrder::Modern;
    p.held_changed_this_step = false;

    let hotbar_changed = if slot_action < HOTBAR_SLOTS {
        // Select a physical slot.
        let changed = slot_action != p.slot;
        p.slot = slot_action;
        changed
    } else {
        // Hotkey an item into the selected slot.
        let item = Item::from_index(slot_action - HOTBAR_SLOTS);
        let owned = item == Item::Empty || p.counts[item.index()] > 0;
        if owned && p.hotbar[p.slot] != item {
            if let Some(k) = p.hotbar.iter().position(|&it| it == item) {
                p.hotbar[k] = p.hotbar[p.slot]; // swap the two slots
            }
            p.hotbar[p.slot] = item;
            true
        } else {
            false
        }
    };
    if !hotbar_changed {
        return;
    }

    p.prev_held = p.held;
    p.held = p.hotbar[p.slot];
    p.held_changed_this_step = true;
    // A hotbar op cancels whatever was being used: the shield drops, the bow
    // draw / crossbow load / eating all reset, and the use input must rise
    // again before anything new starts.
    p.shield_time = 0.0;
    p.bow_draw = 0.0;
    p.crossbow_load = 0.0;
    p.eat_progress = 0.0;
    p.prev_use_item = false;
    if modern {
        // 26.2-pre-2 input order: the swap resolves after attack/use, so
        // nothing lands on the swap tick - model it as a lockout.
        p.swap_lockout = c.swap_lockout_seconds;
    }
}

/// Integrate one tick of physics: gravity or buoyant swim, then a vanilla
/// per-axis AABB move against the voxel world (`collision::move_with_collision`,
/// with the 0.6 step-up and the sneak edge back-off), `on_ground` / velocity /
/// fall-damage derivation, the arena-rim clamp, and the sprint hunger cost.
/// Returns fall damage to apply this tick (post-armor). `scratch` is a reused
/// collision-candidate buffer.
pub(crate) fn integrate(
    p: &mut Player,
    terrain: &Terrain,
    world: &BlockWorld,
    contact: crate::blocks::Contact,
    scratch: &mut Vec<Aabb>,
) -> f32 {
    let c = &cfg().combat;
    let (prev_x, prev_z) = (p.pos.x, p.pos.z);

    // A cobweb / water zeroes fall distance every tick you're in it - that
    // (plus the harder vertical crush in `apply_block_effects`) is why they
    // negate fall damage. The caller already sampled the pre-move contact
    // (also used for `submerged`), so reuse it rather than scan again.
    let ct = contact;
    let submerged = ct.water;
    let cushioned = ct.cobweb || ct.water;
    if cushioned {
        p.fall_distance = 0.0;
    }

    if submerged {
        // Vanilla water physics: buoyant fall, ~0.8 drag on every axis, and
        // holding jump swims you up.
        p.vel.x *= c.swim_drag;
        p.vel.z *= c.swim_drag;
        p.vel.y = (p.vel.y - c.swim_gravity) * c.swim_drag;
        if p.jump_held {
            p.vel.y += c.swim_up_impulse;
        }
    } else {
        p.vel.y = ((p.vel.y - GRAVITY_PER_TICK) * Y_DRAG).max(TERMINAL_VELOCITY_PER_TICK);
    }

    let mut delta = Vec3::new(p.vel.x, p.vel.y, p.vel.z);

    // Vanilla "sneak doesn't walk off ledges": a grounded, crouching player's
    // horizontal delta is shrunk until a foot still has solid support.
    if p.sneaking && p.on_ground {
        let (mut dx, mut dz) = (delta.x, delta.z);
        collision::back_off_from_edge(p.pos, &mut dx, &mut dz, world, terrain);
        if dx != delta.x {
            p.vel.x = 0.0;
        }
        if dz != delta.z {
            p.vel.z = 0.0;
        }
        delta.x = dx;
        delta.z = dz;
    }

    let res = collision::move_with_collision(
        &mut p.pos,
        delta,
        PLAYER_HEIGHT,
        p.on_ground,
        MAX_UP_STEP,
        world,
        terrain,
        scratch,
    );

    p.on_ground = res.hit_y_neg && delta.y <= 0.0;
    if res.hit_y_neg || res.hit_y_pos {
        p.vel.y = 0.0;
    }
    if res.hit_x && !res.stepped_up {
        p.vel.x = 0.0;
    }
    if res.hit_z && !res.stepped_up {
        p.vel.z = 0.0;
    }

    // Vanilla `checkFallDamage`: accumulate the *applied* downward move, reset
    // only on landing. `ceil(d - 3)`, bypassing armor points (EPF still applies).
    let mut fall_after_armor = 0.0;
    if p.on_ground {
        if !cushioned && !submerged && p.fall_distance > FALL_DAMAGE_SAFE_DISTANCE {
            let raw = (p.fall_distance - FALL_DAMAGE_SAFE_DISTANCE).ceil();
            if raw > 0.0 {
                fall_after_armor = combat::apply_epf_only(raw, p.armor);
            }
        }
        p.fall_distance = 0.0;
    } else if res.applied.y < 0.0 {
        p.fall_distance -= res.applied.y;
    }

    if !submerged {
        let drag = if p.on_ground { GROUND_DRAG } else { AIR_DRAG };
        p.vel.x *= drag;
        p.vel.z *= drag;
    }

    // Circular platform rim: hard clamp, then de-penetrate in case the clamp
    // shoved the player into a block near the edge.
    let r = (p.pos.x * p.pos.x + p.pos.z * p.pos.z).sqrt();
    let arena_radius = cfg().arena_radius;
    if r > arena_radius {
        let scale = arena_radius / r;
        p.pos.x *= scale;
        p.pos.z *= scale;
        p.vel.x = 0.0;
        p.vel.z = 0.0;
        collision::push_out_of_solids(&mut p.pos, &mut p.vel, PLAYER_HEIGHT, world, terrain);
    }

    // Sprinting costs hunger by the distance actually covered (vanilla
    // 0.1 exhaustion/block). `dist_moved_last_tick` feeds the sweep check.
    let moved = ((p.pos.x - prev_x).powi(2) + (p.pos.z - prev_z).powi(2)).sqrt();
    p.dist_moved_last_tick = moved;
    if p.sprinting {
        add_exhaustion(p, c.sprint_exhaustion_per_block * moved);
    }

    fall_after_armor
}

/// Cobweb slow / lava & burn damage / water fire-out, the golden-apple Regen
/// tick, and vanilla hunger-gated natural regen (or starvation at food 0).
/// Returns HP lost this tick.
pub(crate) fn apply_block_effects(p: &mut Player, world: &BlockWorld, terrain: &Terrain) -> f32 {
    let c = &cfg().combat;
    let ct = world.player_contact(p.pos);
    let (in_cobweb, in_water, in_lava) = (ct.cobweb, ct.water, ct.lava);

    // Vanilla flowing-fluid push: every fluid cell the hitbox overlaps adds a
    // shove toward its lower-level neighbours; the sum is normalized and
    // scaled (water 0.014 / lava 0.007 per tick).
    if in_water || in_lava {
        let push = world.flow_push(p.pos, terrain);
        let mag = (push.x * push.x + push.z * push.z).sqrt();
        if mag > 1e-6 {
            let scale = if in_water { c.water_push_per_tick } else { c.lava_push_per_tick } / mag;
            p.vel.x += push.x * scale;
            p.vel.z += push.z * scale;
        }
    }

    if in_cobweb {
        p.vel.x *= c.cobweb_velocity_multiplier;
        p.vel.z *= c.cobweb_velocity_multiplier;
        // Vanilla crushes vertical harder than horizontal in a web.
        p.vel.y *= c.cobweb_fall_multiplier;
        p.fall_distance = 0.0;
    }
    if in_water {
        // Horizontal / vertical water drag + buoyancy is handled in
        // `integrate` (swim physics); here we just kill fire.
        p.burn_time_left = 0.0;
        p.burn_accum = 0.0;
        p.fall_distance = 0.0;
    }
    // Fire Resistance: no burn is set and no fire/lava damage lands (the
    // lava velocity slow is physical, so it still applies).
    let fire_immune = p.effects.fire_immune();
    if in_lava {
        if !fire_immune {
            p.burn_time_left = c.lava_burn_seconds;
        }
        p.vel.x *= 0.5;
        p.vel.z *= 0.5;
    }

    // Accumulate lava + fire damage and apply whole HP through i-frames.
    let mut hp_before_pool = 0.0_f32;
    if in_lava && !fire_immune {
        hp_before_pool += c.lava_damage_rate * DT;
    }
    if p.burn_time_left > 0.0 {
        p.burn_time_left = (p.burn_time_left - DT).max(0.0);
        if !fire_immune {
            hp_before_pool += c.lava_burn_rate * DT;
        }
    }
    p.burn_accum += hp_before_pool;
    let mut lost = 0.0;
    if p.burn_accum >= 1.0 {
        let whole = p.burn_accum.floor();
        p.burn_accum -= whole;
        lost = p.take_damage(whole);
    }

    // Regeneration (golden apple / head).
    if p.regen_time_left > 0.0 {
        p.regen_time_left -= DT;
        p.hp = (p.hp + p.regen_rate * DT).min(combat::MAX_HP);
    }

    // Potion effects: tick timers; Regeneration heals, Poison chips away
    // (bypasses armour and i-frames, but never reduces below 1 HP).
    let (potion_heal, potion_poison) = p.effects.tick();
    if potion_heal > 0.0 {
        p.hp = (p.hp + potion_heal).min(combat::MAX_HP);
    }
    if potion_poison > 0.0 && p.hp > 1.0 {
        let d = potion_poison.min(p.hp - 1.0);
        p.hp -= d;
        lost += d;
    }

    // Vanilla hunger-based natural regen: when the `naturalRegeneration`
    // gamerule is on and the player is well-fed, heal 1 HP at a time - fast
    // while saturation remains, slowly on food alone - each HP spending
    // `regen_exhaustion`. At food 0, starve for 1 HP instead.
    if p.hp > 0.0 {
        if p.food <= 0.0 {
            p.starve_accum += DT / c.starve_damage_seconds.max(1e-3);
            if p.starve_accum >= 1.0 {
                p.starve_accum -= 1.0;
                lost += p.take_damage(1.0);
            }
        } else if cfg().natural_regen && p.hp < combat::MAX_HP && p.food >= c.min_food_to_regen {
            let per_hp = if p.saturation > 0.0 {
                c.saturated_regen_seconds
            } else {
                c.unsaturated_regen_seconds
            };
            p.regen_accum += DT / per_hp.max(1e-3);
            if p.regen_accum >= 1.0 {
                p.regen_accum -= 1.0;
                p.hp = (p.hp + 1.0).min(combat::MAX_HP);
                add_exhaustion(p, c.regen_exhaustion);
            }
        }
    }
    p.time_since_damage += DT;

    lost
}

/// Push two players apart if their (vertically overlapping) hitboxes
/// intersect horizontally - vanilla's soft entity push.
pub(crate) fn resolve_player_collision(a: &mut Player, b: &mut Player) {
    let vertically_overlapping =
        a.pos.y < b.pos.y + PLAYER_HEIGHT && b.pos.y < a.pos.y + PLAYER_HEIGHT;
    if !vertically_overlapping {
        return;
    }
    let dx = b.pos.x - a.pos.x;
    let dz = b.pos.z - a.pos.z;
    let dist = (dx * dx + dz * dz).sqrt();
    let min_dist = PLAYER_RADIUS * 2.0;
    if dist < min_dist && dist > 1e-4 {
        let overlap = (min_dist - dist) * 0.5;
        let nx = dx / dist;
        let nz = dz / dist;
        a.pos.x -= nx * overlap;
        a.pos.z -= nz * overlap;
        b.pos.x += nx * overlap;
        b.pos.z += nz * overlap;
    }
}

impl Player {
    /// This tick's public state, for another player's (possibly stale) view.
    pub(crate) fn snapshot(&self) -> PublicSnapshot {
        PublicSnapshot {
            pos: self.pos,
            vel: self.vel,
            height: self.hitbox_height(),
            hp: self.hp,
            absorption: self.absorption,
            shield: if self.shield_up() { 1.0 } else { self.shield_fraction() },
            eating: if self.eating() { 1.0 } else { 0.0 },
            held_ranged: if self.held.is_ranged() { 1.0 } else { 0.0 },
            sneaking: if self.sneaking { 1.0 } else { 0.0 },
        }
    }

    /// Append this tick's snapshot, trimming the ring to `MAX_DELAY_TICKS + 1`.
    pub(crate) fn push_snapshot(&mut self) {
        self.history.push_back(self.snapshot());
        while self.history.len() > MAX_DELAY_TICKS + 1 {
            self.history.pop_front();
        }
    }

    /// This player as seen by an observer whose latency is `observer_ping_ms`
    /// (vanilla: you see others where they were one round-trip ago). Falls
    /// back to the live snapshot before the ring has filled.
    pub(crate) fn delayed_view(&self, observer_ping_ms: f32) -> PublicSnapshot {
        if self.history.is_empty() {
            return self.snapshot();
        }
        let back = delay_back(self.history.len(), observer_ping_ms);
        self.history[self.history.len() - 1 - back]
    }

    /// The hitbox an attacker with `observer_ping_ms` of latency raycasts
    /// against: vanilla melee hit detection is client-side, so the swing
    /// tests against where the target *was* on the attacker's screen one
    /// round-trip ago, not its current server position.
    pub(crate) fn delayed_hitbox(&self, observer_ping_ms: f32) -> Aabb {
        let s = self.delayed_view(observer_ping_ms);
        Aabb::player_box(s.pos, s.height)
    }
}
