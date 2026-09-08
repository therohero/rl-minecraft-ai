//! Runtime-tunable simulation parameters.
//!
//! Everything here used to be a compile-time `const` scattered across
//! `arena.rs`, `combat.rs`, and `terrain.rs`. Pulling the knobs an RL
//! experiment actually wants to sweep (arena size, match length, terrain
//! roughness, reward weights, a few combat numbers) into one struct means
//! they can be changed from a JSON file or a CLI flag with no rebuild, and
//! the exact config a run used is echoed in the `Hello` handshake so the
//! Python side never has to hand-copy these values again.
//!
//! Deliberately *not* configurable: `MAX_HP`, the vanilla per-tick physics
//! constants (gravity/drag/accel in `physics.rs`), and the armor-reduction
//! formula constants. Those are chosen to match vanilla Minecraft exactly -
//! tuning them wouldn't make the sim "more configurable", it would just make
//! it wrong.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static CONFIG: OnceLock<SimConfig> = OnceLock::new();

/// Installs the process-wide config. Call exactly once, early in `main`,
/// before any arena is created. A second call is ignored (returns `Err`).
#[allow(clippy::result_large_err)] // returns the rejected config by value; callers ignore it
pub fn install(config: SimConfig) -> Result<(), SimConfig> {
    CONFIG.set(config)
}

/// The process-wide config. Falls back to `SimConfig::default()` if
/// `install` was never called (e.g. in unit tests).
pub fn cfg() -> &'static SimConfig {
    CONFIG.get_or_init(SimConfig::default)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CombatConfig {
    /// Base melee damage before charge/crit/armor (diamond sword = 7.0).
    pub sword_base_damage: f32,
    /// Attack range (blocks) measured along the attacker's eye ray. Hit
    /// detection is a real raycast now (see `combat::raycast_target`): the
    /// look ray is intersected against each candidate's hitbox and the
    /// nearest one within this distance is struck. Vanilla's survival entity
    /// interaction range is 3.0.
    pub attack_reach: f32,
    /// Seconds for the attack charge to recover from 0.0 to 1.0.
    pub attack_recharge_seconds: f32,
    /// Server-side interaction range (blocks), the distance from the
    /// attacker's *current* eye to the *current* hitbox of the target its
    /// client picked. Vanilla melee is a two-step exchange: the client
    /// raycasts against its latency-stale view to choose whom to hit (that's
    /// `attack_reach` above, tested against `delayed_hitbox`), then the
    /// server re-checks reach against where both players actually are now
    /// (`ServerGamePacketListenerImpl::handleInteract`, a `36.0` squared
    /// distance ⇒ 6.0 blocks) before applying any damage. A target that
    /// dodged out of range during the round-trip takes nothing even though
    /// the attacker's screen showed a clean hit. Set equal to `attack_reach`
    /// to collapse the two steps into one (pre-latency behaviour).
    pub server_interact_range: f32,
    /// How much each target's hitbox is inflated before the attack raycast
    /// is tested against it (vanilla inflates entity boxes slightly when
    /// picking an entity to interact with). 0.0 = exact hitbox.
    pub hitbox_expansion: f32,
    /// Retained for backwards compatibility with older config files; the
    /// raycast hit model no longer uses an aim cone, so this is ignored.
    #[serde(default)]
    pub aim_cos_threshold: f32,
    /// Seconds a target is hurt-invulnerable after taking damage (vanilla's
    /// 20-tick `invulnerableTime`, 1.0 s). Only during the window's *first
    /// half* does a further hit land solely if it exceeds the hit that
    /// opened the window (and then only the difference, without refreshing
    /// the timer); past the halfway mark the next hit lands in full and
    /// resets the window. This is why focus-firing a single target has
    /// sharply diminishing returns in real PvP.
    pub hurt_invulnerability_seconds: f32,
    /// Minimum attack-charge strength (0..1) required for a falling hit to
    /// crit instead of falling back to a plain sweep (vanilla `> 0.9`).
    pub crit_strength_threshold: f32,
    /// Damage multiplier applied to a critical hit.
    pub crit_damage_multiplier: f32,
    /// Base knockback impulse (blocks/tick).
    pub base_knockback: f32,
    /// Extra push a sprint ("lunge") hit adds on top of `base_knockback`.
    /// Vanilla applies this as a separate, unhalved impulse in the
    /// *attacker's own facing direction* rather than toward/away from the
    /// target, which is what makes "knockback displacement" (aiming away
    /// from the target right before a sprint-hit) a real vanilla PvP
    /// mechanic.
    pub sprint_knockback_bonus: f32,
    /// Vertical component of the sprint-hit bonus push above. Added
    /// unconditionally (grounded or airborne) and uncapped, unlike
    /// `knockback_vertical_cap` below which only applies to the base hop.
    pub sprint_knockback_vertical_bonus: f32,
    /// Cap on the vertical "hop" knockback imparts to a grounded target.
    pub knockback_vertical_cap: f32,

    // --- sweep attack (vanilla's area-of-effect swing) ---
    /// Flat damage a sweep attack deals to every *other* player caught in
    /// the sweep box around the primary target (vanilla: `1 + sweeping-edge`,
    /// which is 1.0 with no enchant). The primary target still takes the
    /// normal hit; these are the extra victims.
    pub sweep_damage: f32,
    /// Horizontal inflation (blocks) of the primary target's hitbox that
    /// defines the sweep area (vanilla inflates by 1.0 on x/z, 0.25 on y).
    pub sweep_range: f32,
    /// Knockback impulse applied to each secondary sweep victim.
    pub sweep_knockback: f32,
    /// Minimum attack charge (0..1) for a hit to sweep - vanilla requires a
    /// nearly full swing (`> 0.9`) for the sweep to trigger.
    pub sweep_strength_threshold: f32,

    // --- shield ---
    /// Seconds the `use_item` input must be held continuously before the
    /// shield actually blocks (vanilla: 5 ticks of use time).
    pub shield_raise_seconds: f32,
    /// Fraction of incoming damage (melee *and* arrows) a raised shield
    /// negates when the hit comes from within the frontal arc (1.0 = fully
    /// blocked, as in vanilla). A blocked hit also imparts no knockback.
    pub shield_damage_block: f32,
    /// Movement-speed multiplier while the shield is raised (vanilla slows
    /// you to sneak speed, ~0.2).
    pub shield_move_multiplier: f32,
    /// A shield blocks a hit whose incoming direction is within this arc
    /// (degrees, total) centred on the holder's facing. Vanilla is a full
    /// 180 (any hit from the front hemisphere).
    pub shield_block_arc_degrees: f32,
    /// Seconds an attack / use is locked out after changing the selected
    /// hotbar slot. Only applied under `input_order = "modern"` (the
    /// 26.2-pre-2 `attack -> use -> hotbar` input order, where every swap
    /// eats a tick before anything can follow it). Ignored - and attribute
    /// swapping used instead - under `"legacy"`. Vanilla 1 tick = 0.05 s.
    pub swap_lockout_seconds: f32,
    /// Seconds an axe hit disables the target's shield for (vanilla: 5 s).
    /// Also triggered by an attribute-swapped hit that carries the axe's
    /// disable trait (see `attribute_swapping`).
    pub axe_shield_disable_seconds: f32,

    // --- weapons (per-kit; see `kit.rs` for which kit carries which) ---
    /// Diamond axe base melee damage (before charge/crit/armor).
    pub axe_base_damage: f32,
    /// Diamond axe attack-charge recovery time - slower than the sword
    /// (attack speed 1.0 vs 1.6), which is the whole reason axe->sword
    /// attribute swapping is worth doing.
    pub axe_attack_recharge_seconds: f32,
    /// Diamond pickaxe base melee damage.
    pub pickaxe_base_damage: f32,
    /// Mining (`uhc` only): holding `attack` with a pickaxe **or axe** while
    /// the eye ray is on a **placed** block within `place_reach` breaks it.
    /// Break time (seconds) = `block hardness * this / speed`, where `speed`
    /// is 1.0 with the wrong tool (an axe on stone, a pickaxe on planks) and
    /// `mine_correct_tool_speed (+ efficiency * mine_efficiency_speed_per_level)`
    /// with the right one. The terrain floor and the arena wall are never
    /// mineable. Block hardness is a fixed table in `blocks.rs` (planks 2,
    /// stone 1.5, cobblestone 2, obsidian 50, cobweb 0.8).
    pub mine_seconds_per_hardness: f32,
    /// Mining-speed multiplier of the correct diamond tool (pickaxe for
    /// stone/obsidian, axe for planks) before Efficiency.
    pub mine_correct_tool_speed: f32,
    /// Each Efficiency level adds this to `mine_correct_tool_speed` - but only
    /// when the held tool is the correct one for the block (`uhc` gear = 3).
    pub mine_efficiency_speed_per_level: f32,
    /// Extra melee damage per level of Sharpness (vanilla: 0.5*level + 0.5
    /// for level >= 1).
    pub sharpness_per_level: f32,

    // --- bow / crossbow (projectiles - see `arena.rs`) ---
    /// Seconds to a full bow draw.
    pub bow_max_draw_seconds: f32,
    /// Minimum bow *charge* (0..1, vanilla `(t^2 + 2t)/3`) for the arrow to
    /// fire on release (vanilla ~0.1, about 3 ticks of draw).
    pub bow_min_draw_fraction: f32,
    /// Arrow launch speed (blocks/tick) at a full draw (vanilla 3.0).
    pub arrow_max_speed: f32,
    /// Vanilla `AbstractArrow::baseDamage` (2.0): impact damage is
    /// `ceil(speed_at_impact * this)`, so a full-draw 3.0-speed arrow hits
    /// for 6 before armour, and a drag-slowed long shot hits for less.
    pub arrow_damage_per_speed: f32,
    /// Damage added to `arrow_damage_per_speed` per level of Power (vanilla
    /// `level * 0.5`, plus a flat +0.5 whenever Power >= 1).
    pub power_per_level: f32,
    /// Per-axis triangular aim deviation applied to a launched arrow / bolt,
    /// as a multiple of vanilla's `0.0172275` (bow inaccuracy = 1.0).
    pub projectile_inaccuracy: f32,
    /// Per-tick downward acceleration on an arrow in flight.
    pub arrow_gravity: f32,
    /// Per-tick horizontal+vertical drag multiplier on an arrow.
    pub arrow_drag: f32,
    /// Seconds to load a crossbow while `use_item` is held.
    pub crossbow_load_seconds: f32,
    /// Crossbow bolt launch speed (blocks/tick) - fixed, unlike the bow
    /// (vanilla 3.15). Impact damage uses `arrow_damage_per_speed` like the
    /// bow (no Power on a crossbow).
    pub crossbow_arrow_speed: f32,
    /// How many players a bolt can pass through per level of Piercing before
    /// it stops (vanilla: +1 target per level).
    pub piercing_per_level: u32,

    // --- consumables (UHC) ---
    /// Seconds the `use_item` input must be held to finish eating a golden
    /// apple (vanilla food use time ~1.6 s).
    pub golden_apple_eat_seconds: f32,
    /// Absorption ("extra hearts") granted the instant a golden apple
    /// finishes (this part *is* immediate in vanilla).
    pub golden_apple_absorption: f32,
    /// Seconds of Regeneration a golden apple grants - NO instant heal, the
    /// HP comes back over this window at `golden_apple_regen_rate` (vanilla
    /// Regeneration II ~= 4 HP over 5 s).
    pub golden_apple_regen_seconds: f32,
    /// Golden-apple Regeneration rate (HP per second).
    pub golden_apple_regen_rate: f32,
    /// Hunger + saturation restored (a golden apple is also food).
    pub golden_apple_food: f32,
    pub golden_apple_saturation: f32,
    /// Golden head (UHC): more absorption, longer Regen, and it heals at
    /// **twice** the golden apple's rate.
    pub golden_head_eat_seconds: f32,
    pub golden_head_absorption: f32,
    pub golden_head_regen_seconds: f32,
    pub golden_head_regen_rate: f32,
    pub golden_head_food: f32,
    pub golden_head_saturation: f32,

    // --- hunger / saturation / natural regen ---
    /// Peak sneak-speed movement multiplier (vanilla ~0.3 of walk speed).
    pub sneak_speed_multiplier: f32,
    /// Player hitbox height while sneaking (vanilla 1.5 vs 1.8 standing) -
    /// used by the attack raycast so a crouched target is genuinely harder
    /// to hit high.
    pub sneak_hitbox_height: f32,
    /// How far a solid-block top may drop between the cell a sneaking player
    /// stands on and the cell they're trying to step to before the step is
    /// blocked (vanilla "sneak doesn't walk off ledges").
    pub sneak_ledge_drop: f32,
    /// Exhaustion added per block sprinted / per jump / per sprint-jump /
    /// per landed attack / per 1 HP of natural regen (vanilla 0.1 / 0.05 /
    /// 0.2 / 0.1 / 6.0). At `exhaustion_per_unit` accumulated, one point of
    /// saturation (then food) is spent.
    pub sprint_exhaustion_per_block: f32,
    pub jump_exhaustion: f32,
    pub sprint_jump_exhaustion: f32,
    pub attack_exhaustion: f32,
    pub regen_exhaustion: f32,
    pub exhaustion_per_unit: f32,
    /// Food at/below which the player can no longer start sprinting
    /// (vanilla 6).
    pub min_food_to_sprint: f32,
    /// Food at/above which natural regen ticks (vanilla 18).
    pub min_food_to_regen: f32,
    /// Seconds per 1 HP of natural regen while saturation remains (fast,
    /// vanilla ~0.5 s) vs while merely well-fed (slow, vanilla ~4 s).
    pub saturated_regen_seconds: f32,
    pub unsaturated_regen_seconds: f32,
    /// Seconds per 1 HP of starvation damage at food 0 (vanilla ~4 s).
    pub starve_damage_seconds: f32,

    // --- swimming (vanilla water physics) ---
    /// Downward accel per tick while submerged (vanilla ~0.02, vs 0.08 in
    /// air) - buoyancy.
    pub swim_gravity: f32,
    /// Per-tick velocity multiplier on all three axes while submerged
    /// (vanilla water drag 0.8).
    pub swim_drag: f32,
    /// Upward impulse per tick when `jump` is held while submerged (swim up).
    pub swim_up_impulse: f32,

    // --- placeable blocks + flowing fluids (UHC - see `arena.rs`'s block grid) ---
    /// Seconds between one player's placements (any placeable).
    pub place_cooldown_seconds: f32,
    /// Reach (blocks) of the eye raycast that picks where a placeable lands -
    /// vanilla's `player.block_interaction_range` (4.5). Placement fails
    /// (item kept) if the look ray hits nothing within this distance.
    pub place_reach: f32,
    /// Game ticks between water / lava flow recomputes (vanilla: water 5,
    /// overworld lava 30). Each pass re-floods from the live sources and,
    /// for an un-fed source, drains one flow-level ring (from the middle
    /// out - see `FluidSource`).
    pub water_tick_ticks: u32,
    pub lava_tick_ticks: u32,
    /// Highest flow-level a fluid reaches before it stops (vanilla water 7,
    /// overworld lava 6). With `lava_level_step` = 2 that's a 3-block lava
    /// spread, 7-block water.
    pub water_max_level: u32,
    pub lava_max_level: u32,
    /// Flow-level increment per block of horizontal spread (water 1, lava 2).
    pub lava_level_step: u32,
    /// Per-tick velocity a flowing fluid adds to an overlapping entity along
    /// the normalized flow direction (vanilla water 0.014, lava 0.007).
    pub water_push_per_tick: f32,
    pub lava_push_per_tick: f32,
    /// How many block layers above the terrain floor fluids and placed
    /// blocks may occupy.
    pub block_ceiling: u32,
    /// Hard cap on stored blocks per arena (placed + fluid + generated), so
    /// a pathological pour can't blow up memory or the step time.
    pub max_blocks: usize,
    /// Seconds a placed cobweb / planks block lasts before it decays
    /// (generated stone / cobblestone / obsidian is permanent for the match).
    pub cobweb_block_seconds: f32,
    pub planks_block_seconds: f32,
    /// Seconds a placed water / lava *source* keeps feeding its flow.
    pub water_source_seconds: f32,
    pub lava_source_seconds: f32,
    /// Per-tick horizontal velocity multiplier for a player overlapping a
    /// cobweb block (vanilla cobweb ~0.25).
    pub cobweb_velocity_multiplier: f32,
    /// Per-tick *vertical* velocity multiplier in a cobweb - vanilla crushes
    /// y harder than x/z (~0.05), which (plus a per-tick fall-distance reset)
    /// is why a cobweb negates fall damage.
    pub cobweb_fall_multiplier: f32,
    /// Per-tick horizontal velocity multiplier for a player in water.
    pub water_velocity_multiplier: f32,
    /// HP/s dealt to a player overlapping lava, and the seconds of burning
    /// (out-of-lava fire) a lava touch sets, burning at `lava_burn_rate` HP/s.
    pub lava_damage_rate: f32,
    pub lava_burn_seconds: f32,
    pub lava_burn_rate: f32,

    // --- splash potions (opt-in per config via `SimConfig::splash_potions`;
    //     effect model in `effects.rs`, flight/AoE in `projectile.rs`) ---
    /// Launch speed (blocks/tick) of a thrown splash potion.
    pub splash_potion_speed: f32,
    /// Per-tick downward acceleration on a potion in flight.
    pub splash_potion_gravity: f32,
    /// Per-tick velocity multiplier (drag) on a potion in flight.
    pub splash_potion_drag: f32,
    /// Radius (blocks) of the splash cloud on impact. A caught player's
    /// effect strength scales linearly from full at the centre to nothing
    /// at this distance (vanilla `1 - dist/4`).
    pub splash_radius: f32,
    /// Direct-hit duration (seconds) of each timed splash effect, before the
    /// distance falloff. Instant Health / Damage scale magnitude instead.
    pub potion_speed_seconds: f32,
    pub potion_strength_seconds: f32,
    pub potion_poison_seconds: f32,
    /// Amplifier (0 = level I) each splash potion applies.
    pub potion_speed_amplifier: u32,
    pub potion_strength_amplifier: u32,
    pub potion_poison_amplifier: u32,
    pub potion_healing_amplifier: u32,
    pub potion_harming_amplifier: u32,
    /// Instant Health / Damage HP at amplifier 0 (doubled per level).
    pub instant_health_hp: f32,
    pub instant_damage_hp: f32,
}

impl Default for CombatConfig {
    fn default() -> Self {
        Self {
            sword_base_damage: 7.0,
            attack_reach: 3.0,
            attack_recharge_seconds: 0.625,
            server_interact_range: 6.0,
            hitbox_expansion: 0.1,
            aim_cos_threshold: 0.906,
            hurt_invulnerability_seconds: 1.0,
            crit_strength_threshold: 0.9,
            crit_damage_multiplier: 1.5,
            base_knockback: 0.4,
            sprint_knockback_bonus: 0.5,
            sprint_knockback_vertical_bonus: 0.1,
            knockback_vertical_cap: 0.4,
            sweep_damage: 1.0,
            sweep_range: 1.0,
            sweep_knockback: 0.4,
            sweep_strength_threshold: 0.9,
            shield_raise_seconds: 0.25,
            shield_damage_block: 1.0,
            shield_move_multiplier: 0.2,
            shield_block_arc_degrees: 180.0,
            swap_lockout_seconds: 0.05,
            axe_shield_disable_seconds: 5.0,

            axe_base_damage: 9.0,
            axe_attack_recharge_seconds: 1.0,
            pickaxe_base_damage: 5.0,
            mine_seconds_per_hardness: 1.3,
            mine_correct_tool_speed: 8.0,
            mine_efficiency_speed_per_level: 3.0,
            sharpness_per_level: 0.5,

            bow_max_draw_seconds: 1.0,
            bow_min_draw_fraction: 0.1,
            arrow_max_speed: 3.0,
            arrow_damage_per_speed: 2.0,
            power_per_level: 0.5,
            projectile_inaccuracy: 1.0,
            arrow_gravity: 0.05,
            arrow_drag: 0.99,
            crossbow_load_seconds: 1.25,
            crossbow_arrow_speed: 3.15,
            piercing_per_level: 1,

            golden_apple_eat_seconds: 1.6,
            golden_apple_absorption: 4.0,
            golden_apple_regen_seconds: 5.0,
            golden_apple_regen_rate: 0.8, // Regeneration II ~= 4 HP over 5 s
            golden_apple_food: 4.0,
            golden_apple_saturation: 9.6,
            golden_head_eat_seconds: 1.6,
            golden_head_absorption: 6.0,
            golden_head_regen_seconds: 8.0,
            golden_head_regen_rate: 1.6, // 2x the golden apple
            golden_head_food: 4.0,
            golden_head_saturation: 9.6,

            sneak_speed_multiplier: 0.3,
            sneak_hitbox_height: 1.5,
            sneak_ledge_drop: 0.5,
            sprint_exhaustion_per_block: 0.1,
            jump_exhaustion: 0.05,
            sprint_jump_exhaustion: 0.2,
            attack_exhaustion: 0.1,
            regen_exhaustion: 6.0,
            exhaustion_per_unit: 4.0,
            min_food_to_sprint: 6.0,
            min_food_to_regen: 18.0,
            saturated_regen_seconds: 0.5,
            unsaturated_regen_seconds: 4.0,
            starve_damage_seconds: 4.0,

            swim_gravity: 0.02,
            swim_drag: 0.8,
            swim_up_impulse: 0.039,

            place_cooldown_seconds: 0.3,
            place_reach: 4.5,
            water_tick_ticks: 5,
            lava_tick_ticks: 30,
            water_max_level: 7,
            lava_max_level: 6,
            lava_level_step: 2,
            water_push_per_tick: 0.014,
            lava_push_per_tick: 0.007,
            block_ceiling: 4,
            max_blocks: 512,
            cobweb_block_seconds: 12.0,
            planks_block_seconds: 20.0,
            water_source_seconds: 10.0,
            lava_source_seconds: 10.0,
            cobweb_velocity_multiplier: 0.2,
            cobweb_fall_multiplier: 0.05,
            water_velocity_multiplier: 0.5,
            lava_damage_rate: 4.0,
            lava_burn_seconds: 5.0,
            lava_burn_rate: 1.0,

            splash_potion_speed: 0.5,
            splash_potion_gravity: 0.05,
            splash_potion_drag: 0.99,
            splash_radius: 4.0,
            potion_speed_seconds: 30.0,
            potion_strength_seconds: 30.0,
            potion_poison_seconds: 11.0,
            potion_speed_amplifier: 1,     // splash Speed II
            potion_strength_amplifier: 0,  // Strength I
            potion_poison_amplifier: 0,    // Poison I
            potion_healing_amplifier: 1,   // splash Healing II
            potion_harming_amplifier: 0,   // Harming I
            instant_health_hp: 4.0,        // Healing I = 4, II = 8
            instant_damage_hp: 6.0,        // Harming I = 6, II = 12
        }
    }
}

/// Splash-potion counts a player spawns with. Every field defaults to 0 -
/// no kit carries potions unless a config asks for them, so existing
/// training runs and checkpoints are unaffected.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplashPotionLoadout {
    pub healing: u32,
    pub harming: u32,
    pub poison: u32,
    pub speed: u32,
    pub strength: u32,
}

/// Which combat kit both teams play. `sword` is the classic 1v1 loadout
/// (diamond sword + full diamond armor, no shield); `axe` adds a shield,
/// a diamond axe, a bow, a crossbow and 6 arrows; `uhc` is the Ultra
/// Hardcore kit (a shield, Sharpness/Power/Piercing gear, placeable blocks
/// and buckets, golden apples and heads, and - inherently - no natural
/// regeneration). See `kit.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kit {
    Sword,
    Axe,
    Uhc,
}

/// Which order the client resolves a tick's inputs in - the single knob
/// behind both "attribute swapping" and the "tick lockout".
///
/// - `Legacy` (`hotbar -> attack -> use`): the pre-26.2 order shipped in
///   1.21.11 and every release before it. A slot switch on the same tick as
///   an attack resolves the hit with the *previous* item's attributes
///   (MC-28289 attribute swapping), and a switch costs nothing.
/// - `Modern` (`attack -> use -> hotbar`): the 26.2-pre-2 order. Attribute
///   swapping is impossible (the switch happens after the hit is resolved),
///   and every switch imposes a `combat.swap_lockout_seconds` lockout on the
///   next attack / use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputOrder {
    Legacy,
    Modern,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    /// Reward per HP of damage dealt to the opponent.
    pub per_hp_dealt: f32,
    /// Penalty (subtracted) per HP of damage taken.
    pub per_hp_taken: f32,
    /// Bonus on the step a match is won (awarded to every member of the
    /// winning team in a team fight).
    pub win: f32,
    /// Penalty (subtracted) on the step a match is lost (applied to every
    /// member of the losing team).
    pub loss: f32,
    /// Penalty (subtracted) for landing a plain sweep hit. Now that the
    /// sweep is a real area-of-effect mechanic rather than "the lazy hit",
    /// this defaults to 0 - set it back above 0 only if you specifically
    /// want to discourage sweeps.
    pub sweep_penalty: f32,
    /// Penalty (subtracted) per HP of damage dealt to a *teammate* (only
    /// possible when `friendly_fire` is on). Discourages the policy from
    /// hitting or sweeping its own allies while keeping the mechanic
    /// vanilla-authentic.
    pub friendly_fire_penalty: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            per_hp_dealt: 1.0,
            per_hp_taken: 1.0,
            win: 100.0,
            loss: 100.0,
            sweep_penalty: 0.0,
            friendly_fire_penalty: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimConfig {
    /// Which combat kit both teams play (`sword` / `axe` / `uhc`).
    pub kit: Kit,
    /// Whether the vanilla attribute-swap bug (MC-28289) is modelled:
    /// switching the held hotbar slot on the *same tick* as an attack makes
    /// the hit use the previously held item's damage, attack speed and
    /// on-hit traits (e.g. the axe's shield-disable) while still costing the
    /// new item's charge. This is a load-bearing modern-PvP mechanic (only
    /// patched in 26.2+), and the `axe` / `uhc` kits are built around it.
    ///
    /// Forced `false` by `normalize()` when `input_order = "modern"` - that
    /// input order resolves the switch after the hit, so the bug can't occur.
    pub attribute_swapping: bool,
    /// Which order the client resolves a tick's inputs in (`legacy` =
    /// 1.21.11 `hotbar -> attack -> use`, the default; `modern` = 26.2-pre-2
    /// `attack -> use -> hotbar`). See `InputOrder`.
    pub input_order: InputOrder,
    /// Vanilla's `naturalRegeneration` gamerule: when on, a well-fed player
    /// (food >= `combat.min_food_to_regen`) heals over time, fast while
    /// saturation lasts then slowly, each HP spending `combat.regen_exhaustion`.
    /// Off for `uhc` (golden apples / heads are the only healing there); the
    /// hunger + saturation model itself always runs.
    pub natural_regen: bool,
    /// Players per team. `1` is a 1v1 duel (the default); `2` is a 2v2,
    /// `3` a 3v3, and so on. Each arena always has exactly two teams, so it
    /// holds `2 * team_size` players / policy slots.
    pub team_size: usize,
    /// Whether melee and sweep attacks can damage a teammate. Vanilla has
    /// no team-damage rules without scoreboard teams, so this defaults to
    /// `true` (authentic); the `reward.friendly_fire_penalty` discourages it
    /// without making it impossible.
    pub friendly_fire: bool,
    /// How many of the nearest living enemies each player's observation
    /// describes in full (position, velocity, HP, ...). Enemies past this
    /// count are not individually visible to the policy. Sized for the
    /// largest team fight you intend to train.
    pub max_observed_enemies: usize,
    /// How many of the nearest living teammates each player's observation
    /// describes in full. `0` in a 1v1 (no teammates to see).
    pub max_observed_teammates: usize,
    /// How many of the nearest in-flight arrows/bolts each observation
    /// describes (so a policy can dodge). Only relevant to the `axe` / `uhc`
    /// kits; harmless (always absent) for `sword`.
    pub max_observed_projectiles: usize,
    /// Side length (in block columns) of the square, yaw-rotated block-grid
    /// view in each observation - lets the policy "see" the terrain, placed
    /// blocks, water and lava around it. `5` = a 5x5 patch centred on the
    /// player. Set `0` to omit the block view (e.g. for the `sword` kit).
    pub block_view_size: usize,
    /// Radius (blocks) of the circular platform.
    pub arena_radius: f32,
    /// A solid ring wall `arena_wall_height` blocks tall around the platform
    /// rim. Gives a policy a surface to pin an opponent against (head-webbing).
    /// `normalize()` forces this **on** for the `uhc` kit and **off**
    /// otherwise; a config may still enable it for another kit.
    pub arena_walls: bool,
    /// Height (blocks) of the `arena_walls` ring. `3` is enough to stop a
    /// jump-out and to web an enemy's head against.
    pub arena_wall_height: u32,
    /// Match length (simulated seconds) before a timeout-by-HP decision.
    pub match_time_seconds: f32,
    /// Max terrain height deviation from y=0 (blocks). Observation terrain
    /// fields are normalized by this.
    pub terrain_max_amplitude: f32,
    /// Force every arena to a flat platform (ignores `terrain_max_amplitude`
    /// for shape selection). Handy for a fast first-pass curriculum stage.
    pub terrain_flat_only: bool,
    /// Server-side clamp on per-step yaw/pitch change (radians). The policy's
    /// [-1, 1] look output maps onto [-this, this].
    ///
    /// Vanilla has **no** per-tick limit on how far the crosshair moves - a
    /// mouse flick can snap 180 deg in one tick. This cap is a deliberate,
    /// non-vanilla regularizer: ~3.0 rad/tick (~170 deg) still allows flick
    /// shots and hard turns, but not the instant teleport-aim a policy would
    /// otherwise learn and no human could reproduce against a live server.
    /// Changing it rescales the yaw/pitch action head, so a checkpoint
    /// trained at one value does not transfer to another.
    pub max_look_delta: f32,
    /// How much forward input counts as "pressing W" for sprint eligibility.
    pub sprint_forward_threshold: f32,
    /// How far ahead (blocks) terrain slope is sampled for the observation.
    pub slope_sample_distance: f32,
    /// Lower bound (ms) of the simulated AI<->server network latency. Rolled
    /// once per match per player as a baseline (`base_ping_ms`); the observed
    /// value jitters within `+/- ping_jitter_ms` each tick. Not a vanilla
    /// physics constant - a domain-randomization knob so a trained policy is
    /// robust to the latency it'll see against a live bot (see
    /// `azalea-bot/azalea_bot`, which reports its real tab-list latency the
    /// same way). Latency ages an observer's view of *other* players and
    /// arrows (via a per-tick snapshot ring): both the observation it reads
    /// and its own melee hit detection - a swing raycasts against where the
    /// target *was* one round-trip ago, matching vanilla client-side hit
    /// registration (a laggy attacker lands hits on stale positions). The
    /// observer's own movement and aim stay un-delayed, as in vanilla
    /// client-authoritative movement.
    pub min_ping_ms: f32,
    /// Upper bound (ms) of the per-match baseline latency range above.
    pub max_ping_ms: f32,
    /// Per-tick jitter (ms, +/-) added to `base_ping_ms` for the reported /
    /// applied ping.
    pub ping_jitter_ms: f32,
    /// Splash-potion counts every player spawns with. All-zero by default
    /// (no kit carries potions); set any field to hand that potion to both
    /// teams. See `SplashPotionLoadout` and `effects.rs`.
    pub splash_potions: SplashPotionLoadout,
    pub combat: CombatConfig,
    pub reward: RewardConfig,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            kit: Kit::Sword,
            attribute_swapping: true,
            input_order: InputOrder::Legacy,
            // A vanilla survival duel regenerates, but the sim has trained
            // without it - keep it opt-in (`--natural-regen`); `normalize()`
            // still forces it off for uhc. The hunger/exhaustion model that
            // gates sprinting always runs regardless.
            natural_regen: false,
            team_size: 1,
            friendly_fire: true,
            max_observed_enemies: 3,
            max_observed_teammates: 2,
            max_observed_projectiles: 2,
            block_view_size: 5,
            arena_radius: 12.0,
            arena_walls: false,
            arena_wall_height: 3,
            match_time_seconds: 90.0,
            terrain_max_amplitude: 3.0,
            terrain_flat_only: false,
            max_look_delta: 3.0,
            sprint_forward_threshold: 0.5,
            slope_sample_distance: 1.0,
            min_ping_ms: 5.0,
            max_ping_ms: 100.0,
            ping_jitter_ms: 15.0,
            splash_potions: SplashPotionLoadout::default(),
            combat: CombatConfig::default(),
            reward: RewardConfig::default(),
        }
    }
}

impl SimConfig {
    /// Loads a config from a JSON file. Any field the file omits keeps its
    /// default, so a partial `{"arena_radius": 20}` is valid and only
    /// overrides that one knob.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        cfg.normalize();
        cfg.validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(cfg)
    }

    /// Applies invariants that depend on other fields (rather than rejecting
    /// them). Call once after loading and before `install`.
    pub fn normalize(&mut self) {
        // UHC has no natural regeneration by definition - golden apples and
        // heads are the only healing - and always plays inside the perimeter
        // wall (something to head-web an opponent against).
        if self.kit == Kit::Uhc {
            self.natural_regen = false;
            self.arena_walls = true;
        }
        // The "modern" (26.2-pre-2) input order resolves a slot switch
        // *after* the attack, so the MC-28289 attribute-swap bug cannot
        // occur - keep the two settings consistent rather than letting a
        // config express a combination no real Minecraft version has.
        if self.input_order == InputOrder::Modern && self.attribute_swapping {
            log::info!(
                "input_order=modern forces attribute_swapping=off (the switch resolves after the hit)"
            );
            self.attribute_swapping = false;
        }
    }

    /// Rejects nonsensical combinations early (before any arena is built)
    /// with a human-readable message instead of a panic deep in the sim.
    pub fn validate(&self) -> Result<(), String> {
        if self.team_size == 0 {
            return Err("team_size must be >= 1".into());
        }
        if self.max_observed_enemies == 0 {
            return Err("max_observed_enemies must be >= 1 (each player must be able to see at least one enemy)".into());
        }
        Ok(())
    }

    /// Total players (and policy slots) in one arena: two teams of
    /// `team_size`.
    pub fn players_per_arena(&self) -> usize {
        2 * self.team_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_json_only_overrides_named_fields() {
        let json = r#"{ "arena_radius": 20.0, "reward": { "win": 42.0 } }"#;
        let cfg: SimConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.arena_radius, 20.0);
        assert_eq!(cfg.reward.win, 42.0);
        // untouched fields keep their defaults
        assert_eq!(cfg.match_time_seconds, SimConfig::default().match_time_seconds);
        assert_eq!(cfg.reward.loss, RewardConfig::default().loss);
        assert_eq!(cfg.combat.attack_reach, CombatConfig::default().attack_reach);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let json = r#"{ "arena_radiuss": 20.0 }"#;
        assert!(serde_json::from_str::<SimConfig>(json).is_err());
    }

    #[test]
    fn default_roundtrips_through_json() {
        let text = serde_json::to_string(&SimConfig::default()).unwrap();
        let back: SimConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.combat.sword_base_damage, 7.0);
    }

    #[test]
    fn modern_input_order_forces_attribute_swapping_off() {
        let mut cfg: SimConfig =
            serde_json::from_str(r#"{ "input_order": "modern", "attribute_swapping": true }"#)
                .unwrap();
        cfg.normalize();
        assert!(!cfg.attribute_swapping);
        // legacy (the default) leaves it alone
        let mut legacy = SimConfig::default();
        legacy.normalize();
        assert!(legacy.attribute_swapping);
    }

    #[test]
    fn no_kit_carries_splash_potions_by_default() {
        let cfg = SimConfig::default();
        let sp = cfg.splash_potions;
        assert_eq!((sp.healing, sp.harming, sp.poison, sp.speed, sp.strength), (0, 0, 0, 0, 0));
        for kit in [Kit::Sword, Kit::Axe, Kit::Uhc] {
            let l = crate::kit::loadout(kit);
            for it in [
                crate::kit::Item::SplashHealing,
                crate::kit::Item::SplashHarming,
                crate::kit::Item::SplashPoison,
                crate::kit::Item::SplashSpeed,
                crate::kit::Item::SplashStrength,
            ] {
                assert_eq!(l.counts[it.index()], 0, "{kit:?} must not carry {it:?} by default");
            }
        }
    }

    #[test]
    fn a_config_can_hand_out_splash_potions() {
        let cfg: SimConfig =
            serde_json::from_str(r#"{ "splash_potions": { "poison": 3, "speed": 1 } }"#).unwrap();
        assert_eq!(cfg.splash_potions.poison, 3);
        assert_eq!(cfg.splash_potions.speed, 1);
        assert_eq!(cfg.splash_potions.healing, 0);
    }

    #[test]
    fn uhc_kit_forces_natural_regen_off() {
        let mut cfg: SimConfig =
            serde_json::from_str(r#"{ "kit": "uhc", "natural_regen": true }"#).unwrap();
        cfg.normalize();
        assert!(!cfg.natural_regen);
    }

    #[test]
    fn uhc_kit_enables_the_perimeter_wall_but_other_kits_do_not() {
        let mut uhc: SimConfig = serde_json::from_str(r#"{ "kit": "uhc" }"#).unwrap();
        uhc.normalize();
        assert!(uhc.arena_walls && uhc.arena_wall_height > 0);
        let mut sword = SimConfig::default();
        sword.normalize();
        assert!(!sword.arena_walls);
    }
}
