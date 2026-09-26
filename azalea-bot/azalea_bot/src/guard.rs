//! Client-side legality guard - a "personal anticheat" between the policy
//! and the wire.
//!
//! The trained policy learned inside a sim that is vanilla-*shaped* but not
//! vanilla-*exact*, and nothing stops it from emitting an action a real
//! vanilla client physically could not produce: a 170 deg/tick aim snap,
//! an attack on someone six blocks away or through a wall, a machine-gun
//! click rate, sprinting on an empty hunger bar, a jump in mid-air. Modern
//! server anticheats (GrimAC, Vulcan, Themis, NCP, ...) flag exactly those,
//! and the bot gets kicked.
//!
//! This module rewrites every outgoing action so it stays inside what a
//! legit vanilla client can do:
//!
//!   * **rotation** is rate-limited, low-pass smoothed, given a small
//!     per-tick "hand tremor" so successive deltas are never identical, and
//!     snapped to the vanilla mouse-sensitivity grid (the "GCD" every
//!     rotation-analysis check locks onto). `azalea` already snaps the
//!     absolute rotation it sends to the same 0.15 deg grid; the guard
//!     mirrors that so its own aim math and the wire never drift apart,
//!   * an **attack** is only sent when there is actually a player hitbox
//!     under the crosshair, within `reach`, with a clear line of sight,
//!     *after* the aim has settled (no "spun 40 deg and hit the same
//!     tick"), and no faster than a **randomised** human click cadence,
//!   * **sprint** is dropped when hunger is too low, while an item is being
//!     used, while sneaking, or when not moving forward,
//!   * **jump** is dropped unless the client is actually on the ground,
//!   * **sneak** can't be toggled faster than `min_sneak_hold_ticks`,
//!   * **use-item** never coexists with an attack on the same tick.
//!
//! It is geometry, rate limiting and humanisation only - it never invents
//! inputs. Every knob has an env-var override (see [`GuardConfig::from_env`]);
//! set `AZALEA_GUARD_DISABLE=1` to pass the raw policy action straight
//! through.
//!
//! Intended for bots you are authorized to run (your own test servers,
//! research, CTF-style events) - it keeps an honest RL policy from *looking*
//! like a cheat, it is not a tool for hiding one.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use azalea::block::BlockState;
use azalea::ecs::prelude::Entity;
use azalea::entity::{Physics, Position};
use azalea::physics::collision::BlockWithShape;
use azalea::world::Instance;
use azalea::{BlockPos, SprintDirection, WalkDirection};
use log::info;

use crate::{discretize_walk_direction, nearest_player_entities, sprint_direction_for, Action, State};

/// Vanilla eye height (standing) - the crosshair ray starts here.
const EYE_HEIGHT: f64 = 1.62;
/// Vanilla player hitbox: 0.6 wide, 1.8 tall.
const PLAYER_RADIUS: f64 = 0.3;
const PLAYER_HEIGHT: f64 = 1.8;
/// Vanilla max pitch magnitude (~89 deg - a hair inside the hard ±90 clamp).
const MAX_PITCH: f64 = 1.5533;
/// The rotation quantum (degrees) `azalea::entity::LookDirection::update`
/// snaps every sent rotation delta to: `sensitivity * 0.15` with azalea's
/// hard-coded `sensitivity = 1.0`, which is exactly vanilla's **default
/// 50%** in-game sensitivity. The guard mirrors this so its tracked aim
/// matches what actually goes on the wire; changing it here without
/// patching azalea would only make the two disagree.
const ROTATION_GCD_DEG: f64 = 0.15;

/// Tunable limits for the guard. Defaults are deliberately lenient on the
/// geometry (they only clip the *clearly* inhuman tail) but firmly human on
/// the timing / cadence knobs, which is what modern rotation- and
/// click-analysis anticheats actually score.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuardConfig {
    /// Max yaw change per tick (radians). Vanilla has no limit; humans
    /// sustain ~40-60 deg/tick and burst higher.
    pub max_yaw_rate: f64,
    /// Max pitch change per tick (radians).
    pub max_pitch_rate: f64,
    /// Low-pass factor for rotation, `0.0..=1.0`. `1.0` = no smoothing
    /// (just the rate cap); lower = softer, more human acceleration onto a
    /// target at the cost of tracking lag. Default is < 1.0 so the aim is
    /// never a perfectly linear ramp (a "cinematic" / constant-delta flag).
    pub rotation_smoothing: f64,
    /// Std-dev (degrees) of the per-tick gaussian "hand tremor" mixed into
    /// the rotation *before* it is snapped to the grid, so two successive
    /// deltas are essentially never bit-identical. ~0.4 deg is well inside
    /// human aim noise and invisible in play. `0.0` disables it.
    pub aim_jitter_deg: f64,
    /// If the yaw actually turned more than this in a tick, hold the attack
    /// for one tick so the aim settles first (anticheats flag a big turn
    /// landing a hit on the same tick).
    pub aim_settle_deg: f64,
    /// Melee reach, eye to hitbox surface (blocks). Vanilla survival: 3.0.
    pub reach: f64,
    /// Hitbox inflation for the crosshair test (blocks) - vanilla inflates
    /// the attack pick AABB by 0.1.
    pub hitbox_expansion: f64,
    /// Require an unobstructed line of sight to the target before attacking.
    pub require_line_of_sight: bool,
    /// Max attacks per second (long-run average). Enforced as a minimum gap
    /// between clicks of `1000 / max_cps` ms, jittered by `click_jitter_ms`.
    pub max_cps: f64,
    /// Std-dev (ms) of the random jitter on that minimum click gap, so the
    /// click stream isn't a metronome (auto-clicker "consistency" checks
    /// score inter-click standard deviation / kurtosis).
    pub click_jitter_ms: f64,
    /// Below this food level, sprint is forced off (vanilla: 6).
    pub min_food_to_sprint: f64,
    /// Minimum ticks a sneak state is held before it may flip again.
    pub min_sneak_hold_ticks: u32,
    /// Pass the raw policy action through untouched.
    pub disabled: bool,
}

impl Default for GuardConfig {
    fn default() -> Self {
        GuardConfig {
            max_yaw_rate: 80f64.to_radians(),
            max_pitch_rate: 60f64.to_radians(),
            rotation_smoothing: 0.7,
            aim_jitter_deg: 0.4,
            aim_settle_deg: 50.0,
            reach: 3.0,
            hitbox_expansion: 0.1,
            require_line_of_sight: true,
            max_cps: 12.0,
            click_jitter_ms: 25.0,
            min_food_to_sprint: 6.0,
            min_sneak_hold_ticks: 3,
            disabled: false,
        }
    }
}

impl GuardConfig {
    /// Apply `AZALEA_GUARD_*` env-var overrides on top of the defaults.
    pub fn from_env() -> Self {
        let mut c = GuardConfig::default();
        if let Some(v) = env_f64("AZALEA_GUARD_MAX_YAW_DEG") {
            c.max_yaw_rate = v.to_radians();
        }
        if let Some(v) = env_f64("AZALEA_GUARD_MAX_PITCH_DEG") {
            c.max_pitch_rate = v.to_radians();
        }
        if let Some(v) = env_f64("AZALEA_GUARD_SMOOTHING") {
            c.rotation_smoothing = v.clamp(0.05, 1.0);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_AIM_JITTER_DEG") {
            c.aim_jitter_deg = v.max(0.0);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_AIM_SETTLE_DEG") {
            c.aim_settle_deg = v.max(0.0);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_REACH") {
            c.reach = v;
        }
        if let Some(v) = env_f64("AZALEA_GUARD_HITBOX_EXPANSION") {
            c.hitbox_expansion = v.max(0.0);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_MAX_CPS") {
            c.max_cps = v.max(0.1);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_CLICK_JITTER_MS") {
            c.click_jitter_ms = v.max(0.0);
        }
        if let Some(v) = env_f64("AZALEA_GUARD_MIN_FOOD_SPRINT") {
            c.min_food_to_sprint = v;
        }
        if let Some(v) = env_f64("AZALEA_GUARD_MIN_SNEAK_HOLD_TICKS") {
            c.min_sneak_hold_ticks = v.max(0.0) as u32;
        }
        if let Some(v) = env_bool("AZALEA_GUARD_LOS") {
            c.require_line_of_sight = v;
        }
        if env_bool("AZALEA_GUARD_DISABLE").unwrap_or(false) {
            c.disabled = true;
        }
        c
    }
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.trim().parse().ok()
}
fn env_bool(key: &str) -> Option<bool> {
    match std::env::var(key).ok()?.trim() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// A fully-sanitized action, ready to hand straight to Azalea's client API
/// with no further decisions.
pub(crate) struct SafeAction {
    pub walk: WalkDirection,
    pub sprint: Option<SprintDirection>,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub jump: bool,
    /// The exact entity to swing at this tick, or `None` for no attack.
    pub attack: Option<Entity>,
    pub use_item: bool,
    pub sneak: bool,
    pub held_slot: i64,
}

/// Per-connection guard state (rotation low-pass memory, click history,
/// sneak-hold timer, suppression counters for the periodic report).
pub(crate) struct Guard {
    cfg: GuardConfig,
    smoothed_yaw_delta: f64,
    smoothed_pitch_delta: f64,
    /// Actual applied yaw turn last tick (deg), for the aim-settle gate.
    last_yaw_turn_deg: f64,
    clicks: VecDeque<Instant>,
    /// Earliest instant the next click is allowed (randomised each click).
    next_click_at: Instant,
    /// Current committed sneak state and how many ticks it's been held.
    sneak_state: bool,
    sneak_held_ticks: u32,
    suppressed_attacks: u64,
    suppressed_sprints: u64,
    suppressed_jumps: u64,
    clamped_rotations: u64,
    last_report: Instant,
}

impl Guard {
    pub fn new(cfg: GuardConfig) -> Self {
        let now = Instant::now();
        Guard {
            cfg,
            smoothed_yaw_delta: 0.0,
            smoothed_pitch_delta: 0.0,
            last_yaw_turn_deg: 0.0,
            clicks: VecDeque::with_capacity(32),
            next_click_at: now,
            sneak_state: false,
            sneak_held_ticks: u32::MAX / 2,
            suppressed_attacks: 0,
            suppressed_sprints: 0,
            suppressed_jumps: 0,
            clamped_rotations: 0,
            last_report: now,
        }
    }

    /// Reset the rotation low-pass after a respawn (the server snaps our
    /// orientation, so carried-over deltas would be meaningless).
    pub fn on_respawn(&mut self) {
        self.smoothed_yaw_delta = 0.0;
        self.smoothed_pitch_delta = 0.0;
        self.last_yaw_turn_deg = 0.0;
    }

    /// Turn one raw policy [`Action`] into a [`SafeAction`], updating the
    /// bot's tracked look direction in `state.look` as a side effect.
    pub fn sanitize(&mut self, bot: &azalea::Client, state: &State, raw: &Action) -> SafeAction {
        let (yaw, pitch) = self.resolve_look(state, raw);

        let walk = discretize_walk_direction(raw.move_x, raw.move_z);
        let using_item = raw.use_item && !raw.attack;
        let sprint = self.resolve_sprint(bot, raw, walk, using_item);
        let jump = self.resolve_jump(bot, raw);
        let sneak = self.resolve_sneak(raw);

        let attack = if raw.attack {
            self.resolve_attack(bot, state, yaw, pitch)
        } else {
            None
        };
        // Vanilla never attacks and uses an item on the same tick; an
        // attack always wins (it drops any raised guard anyway).
        let use_item = raw.use_item && attack.is_none();

        self.maybe_report();

        SafeAction {
            walk,
            sprint,
            yaw_deg: yaw.to_degrees() as f32,
            pitch_deg: pitch.to_degrees() as f32,
            jump,
            attack,
            use_item,
            sneak,
            held_slot: raw.held_slot,
        }
    }

    /// Integrate the policy's look deltas: low-pass, rate-limit, add a small
    /// gaussian tremor, then snap the absolute result to the vanilla
    /// sensitivity grid - the same grid `azalea` sends on. Writes the
    /// snapped `(yaw, pitch)` (radians) back to `state.look` and returns it.
    fn resolve_look(&mut self, state: &State, raw: &Action) -> (f64, f64) {
        let (mut yaw, mut pitch) = *state.look.lock().unwrap();

        if self.cfg.disabled {
            yaw += raw.yaw_delta;
            pitch = (pitch + raw.pitch_delta).clamp(-MAX_PITCH, MAX_PITCH);
            *state.look.lock().unwrap() = (yaw, pitch);
            self.last_yaw_turn_deg = raw.yaw_delta.to_degrees().abs();
            return (yaw, pitch);
        }

        let a = self.cfg.rotation_smoothing;
        self.smoothed_yaw_delta += a * (raw.yaw_delta - self.smoothed_yaw_delta);
        self.smoothed_pitch_delta += a * (raw.pitch_delta - self.smoothed_pitch_delta);

        let mut dy = self
            .smoothed_yaw_delta
            .clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate);
        let mut dp = self
            .smoothed_pitch_delta
            .clamp(-self.cfg.max_pitch_rate, self.cfg.max_pitch_rate);
        if dy != self.smoothed_yaw_delta || dp != self.smoothed_pitch_delta {
            self.clamped_rotations += 1;
        }
        // Bleed the clamped-off part out of the low-pass memory so it does
        // not accumulate into a lasting turn-rate bias.
        self.smoothed_yaw_delta = dy;
        self.smoothed_pitch_delta = dp;

        // Sub-degree gaussian tremor (radians), added before the grid snap
        // so it actually perturbs which grid cell we land in.
        if self.cfg.aim_jitter_deg > 0.0 {
            let j = self.cfg.aim_jitter_deg.to_radians();
            dy += gaussian() * j;
            dp += gaussian() * j;
        }

        let yaw_before = yaw;
        yaw = snap_to_grid_rad(yaw + dy);
        pitch = snap_to_grid_rad((pitch + dp).clamp(-MAX_PITCH, MAX_PITCH));

        self.last_yaw_turn_deg = (yaw - yaw_before).to_degrees().abs();
        *state.look.lock().unwrap() = (yaw, pitch);
        (yaw, pitch)
    }

    fn resolve_sprint(
        &mut self,
        bot: &azalea::Client,
        raw: &Action,
        walk: WalkDirection,
        using_item: bool,
    ) -> Option<SprintDirection> {
        if !raw.sprint {
            return None;
        }
        let food = bot
            .get_component::<azalea::local_player::Hunger>()
            .map(|h| h.food as f64)
            .unwrap_or(20.0);
        let legal = self.cfg.disabled
            || (!raw.sneak && !using_item && food > self.cfg.min_food_to_sprint);
        if !legal {
            self.suppressed_sprints += 1;
            return None;
        }
        // Vanilla also has no sprinting sideways / backward - `None` here is
        // just an un-sprintable direction, not a violation.
        sprint_direction_for(walk)
    }

    /// Vanilla only starts a jump from the ground (mid-air `jump` presses do
    /// nothing); sending one anyway is a textbook Fly / NoFall signature.
    fn resolve_jump(&mut self, bot: &azalea::Client, raw: &Action) -> bool {
        if !raw.jump {
            return false;
        }
        if self.cfg.disabled {
            return true;
        }
        let on_ground = bot.get_component::<Physics>().map(|p| p.on_ground()).unwrap_or(false);
        if !on_ground {
            self.suppressed_jumps += 1;
        }
        on_ground
    }

    /// Debounce the sneak toggle: once a state is committed it must be held
    /// for `min_sneak_hold_ticks` before it can flip (rapid crouch spam is
    /// its own anticheat flag and packet-spams START/STOP_SNEAKING).
    fn resolve_sneak(&mut self, raw: &Action) -> bool {
        self.sneak_held_ticks = self.sneak_held_ticks.saturating_add(1);
        if self.cfg.disabled {
            self.sneak_state = raw.sneak;
            return raw.sneak;
        }
        if raw.sneak != self.sneak_state && self.sneak_held_ticks >= self.cfg.min_sneak_hold_ticks {
            self.sneak_state = raw.sneak;
            self.sneak_held_ticks = 0;
        }
        self.sneak_state
    }

    /// Pick the entity the crosshair is actually on (nearest hit within
    /// reach, LOS permitting), enforce the humanised click cadence, and hold
    /// fire for a tick right after a big turn.
    fn resolve_attack(&mut self, bot: &azalea::Client, state: &State, yaw: f64, pitch: f64) -> Option<Entity> {
        if self.cfg.disabled {
            // Legacy behaviour: swing at the nearest player, unconditionally.
            return nearest_player_entities(bot, state).into_iter().next();
        }

        if self.last_yaw_turn_deg > self.cfg.aim_settle_deg {
            self.suppressed_attacks += 1;
            return None;
        }
        if !self.click_budget_ok() {
            self.suppressed_attacks += 1;
            return None;
        }

        let self_pos = bot.get_component::<Position>()?;
        let eye = [self_pos.x, self_pos.y + EYE_HEIGHT, self_pos.z];
        let dir = look_direction(yaw, pitch);

        // Snapshot candidate hitboxes before taking the world lock.
        let candidates: Vec<(Entity, [f64; 3])> = nearest_player_entities(bot, state)
            .into_iter()
            .filter_map(|e| {
                let p = bot.get_entity_component::<Position>(e)?;
                Some((e, [p.x, p.y, p.z]))
            })
            .collect();
        if candidates.is_empty() {
            self.suppressed_attacks += 1;
            return None;
        }

        let world = bot.world();
        let world = world.read();
        let exp = self.cfg.hitbox_expansion;

        let mut best: Option<(Entity, f64)> = None;
        for (e, feet) in candidates {
            let min = [
                feet[0] - PLAYER_RADIUS - exp,
                feet[1] - exp,
                feet[2] - PLAYER_RADIUS - exp,
            ];
            let max = [
                feet[0] + PLAYER_RADIUS + exp,
                feet[1] + PLAYER_HEIGHT + exp,
                feet[2] + PLAYER_RADIUS + exp,
            ];
            let Some(t) = ray_aabb(eye, dir, min, max) else {
                continue;
            };
            if t > self.cfg.reach {
                continue;
            }
            if self.cfg.require_line_of_sight && segment_hits_block(&world, eye, dir, t) {
                continue;
            }
            if best.is_none_or(|(_, bt)| t < bt) {
                best = Some((e, t));
            }
        }

        match best {
            Some((e, _)) => {
                self.register_click();
                Some(e)
            }
            None => {
                // The policy wanted to hit but nothing legal was in front
                // of the crosshair (out of reach / off-aim / walled off).
                self.suppressed_attacks += 1;
                None
            }
        }
    }

    /// True if enough time has passed since the last click *and* the 1 s
    /// sliding window is under `max_cps` (a hard ceiling on top of the gap).
    fn click_budget_ok(&mut self) -> bool {
        let now = Instant::now();
        if now < self.next_click_at {
            return false;
        }
        let cutoff = now - Duration::from_secs(1);
        while self.clicks.front().is_some_and(|&t| t < cutoff) {
            self.clicks.pop_front();
        }
        (self.clicks.len() as f64) < self.cfg.max_cps
    }

    fn register_click(&mut self) {
        let now = Instant::now();
        self.clicks.push_back(now);
        // Base gap for the target rate, nudged by gaussian jitter so the
        // inter-click series has a human-like spread rather than a fixed period.
        let base_ms = 1000.0 / self.cfg.max_cps;
        let gap_ms = (base_ms + gaussian() * self.cfg.click_jitter_ms).max(base_ms * 0.5);
        self.next_click_at = now + Duration::from_secs_f64(gap_ms / 1000.0);
    }

    fn maybe_report(&mut self) {
        if self.last_report.elapsed() < Duration::from_secs(15) {
            return;
        }
        self.last_report = Instant::now();
        let (a, s, j, r) = (
            self.suppressed_attacks,
            self.suppressed_sprints,
            self.suppressed_jumps,
            self.clamped_rotations,
        );
        if a | s | j | r != 0 {
            info!(
                "legality guard (last 15s): {a} illegal attacks dropped, {s} sprint requests denied, \
                 {j} mid-air jumps denied, {r} rotations rate-limited"
            );
        }
        self.suppressed_attacks = 0;
        self.suppressed_sprints = 0;
        self.suppressed_jumps = 0;
        self.clamped_rotations = 0;
    }
}

/// Snap an absolute angle (radians) to the vanilla `ROTATION_GCD_DEG` mouse
/// grid, matching `azalea::entity::LookDirection::update`.
fn snap_to_grid_rad(angle_rad: f64) -> f64 {
    let g = ROTATION_GCD_DEG.to_radians();
    (angle_rad / g).round() * g
}

/// One draw from an approx. standard normal (sum of 3 uniforms - the
/// Irwin-Hall triangular-ish approximation; plenty for sub-degree aim /
/// millisecond click noise, and no extra dependency beyond `fastrand`).
fn gaussian() -> f64 {
    // sum of 3 U(0,1) has mean 1.5 and std 0.5; rescale to ~N(0, 1).
    (fastrand::f64() + fastrand::f64() + fastrand::f64() - 1.5) * 2.0
}

/// Sim look convention (`sim/src/physics.rs::look_direction`): yaw 0 faces
/// `+z`, `+yaw` turns toward `-x`. Result is a unit vector.
fn look_direction(yaw: f64, pitch: f64) -> [f64; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    [-sy * cp, -sp, cy * cp]
}

/// Slab-method ray/AABB entry distance (`>= 0`), or `None` if the ray
/// misses or the box is entirely behind the origin. Mirrors
/// `sim/src/physics.rs::Aabb::ray_intersect`.
fn ray_aabb(origin: [f64; 3], dir: [f64; 3], min: [f64; 3], max: [f64; 3]) -> Option<f64> {
    let inside = (0..3).all(|i| origin[i] >= min[i] && origin[i] <= max[i]);
    if inside {
        return Some(0.0);
    }
    let mut t_min = f64::NEG_INFINITY;
    let mut t_max = f64::INFINITY;
    for i in 0..3 {
        if dir[i].abs() < 1e-9 {
            if origin[i] < min[i] || origin[i] > max[i] {
                return None;
            }
        } else {
            let inv = 1.0 / dir[i];
            let mut t1 = (min[i] - origin[i]) * inv;
            let mut t2 = (max[i] - origin[i]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max < 0.0 {
        None
    } else {
        Some(t_min.max(0.0))
    }
}

/// Amanatides-Woo voxel walk: is there a full-collision block strictly
/// between the eye and distance `max_t` along `dir` (unit)? Ignores the
/// voxel the eye sits in. Only full cubes block sight, matching the sim's
/// "full blocks only" world.
fn segment_hits_block(world: &Instance, origin: [f64; 3], dir: [f64; 3], max_t: f64) -> bool {
    let mut cell = [
        origin[0].floor() as i32,
        origin[1].floor() as i32,
        origin[2].floor() as i32,
    ];
    let mut step = [0i32; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for i in 0..3 {
        if dir[i].abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / dir[i];
        step[i] = if dir[i] > 0.0 { 1 } else { -1 };
        let boundary = if dir[i] > 0.0 {
            (cell[i] as f64) + 1.0
        } else {
            cell[i] as f64
        };
        t_max[i] = (boundary - origin[i]) * inv;
        t_delta[i] = inv.abs();
    }

    let mut t;
    for _ in 0..256 {
        // advance to the next voxel first (skips the eye's own voxel)
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        cell[axis] += step[axis];
        t = t_max[axis];
        t_max[axis] += t_delta[axis];
        if t > max_t {
            return false;
        }
        let solid = world
            .get_block_state(BlockPos::new(cell[0], cell[1], cell[2]))
            .unwrap_or(BlockState::AIR)
            .is_collision_shape_full();
        if solid {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_direction_matches_sim_convention() {
        let f = look_direction(0.0, 0.0);
        assert!((f[0]).abs() < 1e-9 && (f[1]).abs() < 1e-9 && (f[2] - 1.0).abs() < 1e-9);
        let l = look_direction(std::f64::consts::FRAC_PI_2, 0.0);
        assert!((l[0] + 1.0).abs() < 1e-9 && l[2].abs() < 1e-9);
        // looking straight down
        let d = look_direction(0.0, std::f64::consts::FRAC_PI_2);
        assert!((d[1] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn ray_aabb_hits_a_box_dead_ahead() {
        let eye = [0.0, 1.62, 0.0];
        let dir = [0.0, 0.0, 1.0];
        // a 0.6-wide box whose near face is at z = 2.0
        let t = ray_aabb(eye, dir, [-0.3, 0.0, 2.0], [0.3, 1.8, 2.6]).unwrap();
        assert!((t - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ray_aabb_misses_when_pointing_away_or_off_axis() {
        let eye = [0.0, 1.62, 0.0];
        assert!(ray_aabb(eye, [0.0, 0.0, -1.0], [-0.3, 0.0, 2.0], [0.3, 1.8, 2.6]).is_none());
        assert!(ray_aabb(eye, [1.0, 0.0, 0.0], [-0.3, 0.0, 2.0], [0.3, 1.8, 2.6]).is_none());
    }

    #[test]
    fn ray_aabb_from_inside_is_zero() {
        let t = ray_aabb([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]);
        assert_eq!(t, Some(0.0));
    }

    #[test]
    fn guard_config_env_overrides_apply() {
        // Not run in parallel with other env users; scoped set/remove.
        std::env::set_var("AZALEA_GUARD_MAX_CPS", "9");
        std::env::set_var("AZALEA_GUARD_DISABLE", "1");
        let c = GuardConfig::from_env();
        assert_eq!(c.max_cps, 9.0);
        assert!(c.disabled);
        std::env::remove_var("AZALEA_GUARD_MAX_CPS");
        std::env::remove_var("AZALEA_GUARD_DISABLE");
    }

    #[test]
    fn grid_snap_produces_multiples_of_the_vanilla_quantum() {
        let g = ROTATION_GCD_DEG.to_radians();
        for raw_deg in [0.02_f64, 0.9, -13.37, 42.0, 179.4] {
            let snapped = snap_to_grid_rad(raw_deg.to_radians());
            let k = snapped / g;
            assert!((k - k.round()).abs() < 1e-6, "{raw_deg} deg not on the grid");
        }
        // consecutive snapped absolutes => the delta is also a grid multiple
        // (what a rotation-GCD anticheat actually inspects).
        let a = snap_to_grid_rad(10.0_f64.to_radians());
        let b = snap_to_grid_rad(10.0_f64.to_radians() + 0.37_f64.to_radians());
        let k = (b - a) / g;
        assert!((k - k.round()).abs() < 1e-6);
    }

    #[test]
    fn gaussian_is_roughly_centred_and_bounded() {
        let mut sum = 0.0;
        for _ in 0..5000 {
            let v = gaussian();
            assert!(v.abs() <= 3.0);
            sum += v;
        }
        assert!((sum / 5000.0).abs() < 0.15);
    }

    fn cfg() -> GuardConfig {
        GuardConfig::default()
    }

    #[test]
    fn sneak_toggle_is_debounced() {
        // default min_sneak_hold_ticks = 3.
        let mut g = Guard::new(cfg());
        let on = Action { sneak: true, ..raw_noop() };
        let off = Action { sneak: false, ..raw_noop() };
        assert!(g.resolve_sneak(&on)); // commits `true`, resets the hold timer
        assert!(g.resolve_sneak(&off)); // still true - only held 1 tick
        assert!(g.resolve_sneak(&off)); // still true - held 2 ticks
        assert!(!g.resolve_sneak(&off)); // held >= 3 - now flips to false
        assert!(!g.resolve_sneak(&on)); // held 1 tick since the flip
    }

    #[test]
    fn click_cadence_respects_the_minimum_gap() {
        let mut g = Guard::new(GuardConfig { max_cps: 10.0, click_jitter_ms: 0.0, ..cfg() });
        assert!(g.click_budget_ok());
        g.register_click();
        // immediately after a click the budget is closed (100 ms gap at 10 CPS)
        assert!(!g.click_budget_ok());
    }

    fn raw_noop() -> Action {
        Action {
            move_x: 0.0,
            move_z: 0.0,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            jump: false,
            attack: false,
            sprint: false,
            use_item: false,
            sneak: false,
            held_slot: 0,
            lstm_state: None,
        }
    }
}
