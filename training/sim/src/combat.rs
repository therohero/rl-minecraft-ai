//! Melee combat: vanilla's attack-charge damage scaling, its three hit
//! types (sweep, critical, sprint) and the knockback each imparts, the
//! 0.5 s hurt-invulnerability window, 180-deg shield blocking, the sweep
//! attack's area of effect, and per-weapon stats (sword / axe / pickaxe,
//! Sharpness) including the axe's shield-disable.
//!
//! `resolve_melee` is the whole swing: an eye-ray target pick, the damage /
//! knockback math (`resolve_attack` + `apply_*_knockback`), the sprint-hit
//! sprint-reset and attack exhaustion, and the sweep AoE. Bows/crossbows
//! live in `projectile.rs`.

pub const MAX_HP: f32 = 20.0;

use crate::arena::StepEvents;
use crate::blocks::BlockWorld;
use crate::config::cfg;
use crate::kit::Item;
use crate::physics::{aabb_overlap, look_direction, pair_mut, Aabb, Vec3, EYE_HEIGHT, PLAYER_HEIGHT, PLAYER_RADIUS};
use crate::player::{add_exhaustion, Player};
use crate::terrain::Terrain;

/// Which of vanilla's three melee hit types landed. `Sweep` is the default:
/// no special condition (falling, sprinting) applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitType {
    Sweep,
    Critical,
    SprintKnockback,
}

/// The victim's armour profile (from their kit's `Loadout`).
#[derive(Debug, Clone, Copy)]
pub struct Armor {
    pub points: f32,
    pub toughness: f32,
    pub protection_epf: f32,
    /// Knockback Resistance attribute, 0..1: fraction of an incoming
    /// knockback impulse ignored (0 unless a config's `enchants` sets it).
    pub knockback_resistance: f32,
}

/// The stats of the weapon a melee hit is resolved with. When attribute
/// swapping is in play this is the *previously* held weapon, not the one
/// now in hand (see `arena::try_attack`).
#[derive(Debug, Clone, Copy)]
pub struct Weapon {
    pub base_damage: f32,
    pub sharpness_bonus: f32,
    pub recharge_seconds: f32,
    /// A hit with this weapon disables a blocking target's shield (axe).
    pub disables_shield: bool,
    /// A full-charge, near-stationary, grounded swing with this weapon
    /// triggers the sweep AoE. Vanilla: swords only.
    pub sweeps: bool,
}

pub struct AttackResult {
    /// Damage after armor and shield but *before* the hurt-invulnerability /
    /// last-damage rule - the caller applies that (see `apply_iframes`).
    pub full_damage: f32,
    pub hit_type: HitType,
    /// The target's raised shield caught this hit (from within the frontal
    /// arc): damage reduced, no knockback.
    pub blocked: bool,
    /// This hit should disable the target's shield for
    /// `combat.axe_shield_disable_seconds` (an axe hit, or an
    /// attribute-swapped hit carrying the axe trait, landing on a shield).
    pub disable_shield: bool,
    /// This swing is a vanilla sweep attack: also deal `combat.sweep_damage`
    /// + knockback to every *other* player in the sweep box.
    pub triggers_sweep: bool,
}

/// The subset of attacker state needed to resolve a melee attack.
pub struct AttackerState {
    pub pos: Vec3,
    pub vel_y: f32,
    pub on_ground: bool,
    pub sprinting: bool,
    /// In water - vanilla can't land a critical hit while submerged.
    pub in_water: bool,
    /// Horizontal distance moved *last* tick (vanilla `walkDist` delta) - a
    /// sweep only triggers when moving no faster than a walk.
    pub dist_moved_last_tick: f32,
    /// Seconds since this attacker last swung.
    pub time_since_last_attack: f32,
    /// Flat melee-damage delta from Strength / Weakness (see `effects`),
    /// added to the weapon base before the attack-charge multiplier.
    pub effect_damage_add: f32,
}

/// The subset of target state needed to resolve a melee attack.
pub struct TargetState {
    pub pos: Vec3,
    pub yaw: f32,
    /// Shield raised far enough to block *and* not currently axe-disabled.
    pub shield_up: bool,
    pub hurt_time_left: f32,
    pub last_damage: f32,
    pub armor: Armor,
}

/// Vanilla's "attack strength scale" for a weapon with the given recharge
/// time: 0 right after swinging, ramping to 1.0 once recharged.
pub fn strength_scale(time_since_last_attack: f32, recharge_seconds: f32) -> f32 {
    ((time_since_last_attack + 0.5 * crate::physics::DT) / recharge_seconds.max(1e-3)).clamp(0.0, 1.0)
}

/// Vanilla's damage multiplier from attack strength: 20% floor right after
/// swinging, ramping to 100% at full charge.
pub fn damage_multiplier(strength_scale: f32) -> f32 {
    0.2 + strength_scale * strength_scale * 0.8
}

/// Dot-product threshold for a shield to catch a hit, from the configured
/// arc (degrees, total). 180 deg -> 0.0 (front hemisphere).
fn shield_block_dot() -> f32 {
    (cfg().combat.shield_block_arc_degrees.to_radians() * 0.5).cos()
}

/// Is a hit from `attacker_pos` caught by `target`'s raised shield?
pub fn shield_blocks(target: &TargetState, attacker_pos: Vec3) -> bool {
    if !target.shield_up {
        return false;
    }
    let to_attacker = Vec3::new(attacker_pos.x - target.pos.x, 0.0, attacker_pos.z - target.pos.z);
    let len = (to_attacker.x * to_attacker.x + to_attacker.z * to_attacker.z).sqrt();
    if len < 1e-4 {
        return true;
    }
    let look = look_direction(target.yaw, 0.0);
    let dot = (look.x * to_attacker.x + look.z * to_attacker.z) / len;
    dot > shield_block_dot()
}

/// Applies vanilla's hurt-invulnerability / last-damage rule. Returns
/// `(damage_to_apply, new_last_damage, refresh_timer)`.
///
/// Vanilla `LivingEntity.hurt`: the invulnerability timer is 20 ticks
/// (`hurt_invulnerability_seconds`). While it is in its *first half*
/// (`invulnerableTime > 10`) a further hit only lands if it exceeds the hit
/// that opened the window, only the difference is dealt, and the timer is
/// **not** refreshed. Once past the halfway point the next hit lands in full
/// and resets the timer. `refresh_timer` reports which branch was taken.
pub fn apply_iframes(
    incoming: f32,
    invuln_time_left: f32,
    last_damage: f32,
) -> (f32, f32, bool) {
    let window = cfg().combat.hurt_invulnerability_seconds;
    if invuln_time_left > 0.5 * window {
        if incoming > last_damage {
            (incoming - last_damage, incoming, false)
        } else {
            (0.0, last_damage, false)
        }
    } else {
        (incoming, incoming, true)
    }
}

/// Resolves a melee hit that raycasting has already confirmed connects.
pub fn resolve_attack(
    attacker: &AttackerState,
    weapon: &Weapon,
    target: &TargetState,
) -> AttackResult {
    let combat = &cfg().combat;

    let strength = strength_scale(attacker.time_since_last_attack, weapon.recharge_seconds);
    let mut damage = (weapon.base_damage + weapon.sharpness_bonus + attacker.effect_damage_add).max(0.0)
        * damage_multiplier(strength);

    // Vanilla crit: falling, not on the ground, not in water, not sprinting,
    // and a nearly-full swing.
    let can_crit = attacker.vel_y < 0.0 && !attacker.on_ground && !attacker.in_water;
    let hit_type = if attacker.sprinting {
        HitType::SprintKnockback
    } else if can_crit && strength >= combat.crit_strength_threshold {
        damage *= combat.crit_damage_multiplier;
        HitType::Critical
    } else {
        HitType::Sweep
    };

    let triggers_sweep = hit_type == HitType::Sweep
        && weapon.sweeps
        && attacker.on_ground
        && strength > combat.sweep_strength_threshold
        && attacker.dist_moved_last_tick < crate::physics::WALK_ACCEL_PER_TICK;

    let mut damage = apply_armor_reduction(damage, target.armor);

    let blocked = shield_blocks(target, attacker.pos);
    // An axe hit (or attribute-swapped axe trait) that lands on a *raised*
    // shield disables it - whether or not the block itself succeeds.
    let disable_shield = weapon.disables_shield && target.shield_up;
    if blocked {
        damage *= 1.0 - combat.shield_damage_block;
    }

    AttackResult { full_damage: damage, hit_type, blocked, disable_shield, triggers_sweep }
}

/// Pre-i-frame damage a secondary sweep victim takes.
pub fn resolve_sweep_hit(armor: Armor) -> f32 {
    apply_armor_reduction(cfg().combat.sweep_damage, armor)
}

/// Applies armor points + Protection to incoming damage, in vanilla's two
/// steps (`CombatRules.getDamageAfterAbsorb` then `getDamageAfterMagicAbsorb`).
/// The pre-1.9 `nextFloat()*0.5+0.5` roll on the enchantment factor is gone:
/// modern Java Edition sums the EPF, clamps to 20, and reduces by `epf/25`
/// deterministically.
pub fn apply_armor_reduction(damage: f32, armor: Armor) -> f32 {
    let g = (armor.points - 4.0 * damage / (8.0 + armor.toughness)).clamp(armor.points * 0.2, 20.0);
    apply_epf_only(damage * (1.0 - g / 25.0), armor)
}

/// The Protection half of the formula only - for damage that bypasses armor
/// points (`#minecraft:bypasses_armor`: fall, magic, ...), where the generic
/// Protection enchantment still applies via its EPF.
pub fn apply_epf_only(damage: f32, armor: Armor) -> f32 {
    let epf = armor.protection_epf.clamp(0.0, 20.0);
    damage * (1.0 - epf / 25.0)
}

/// Weapon stats for a melee hit. `base_item` is what the game resolves the
/// hit *with* (the previously held item under attribute swapping); `held`
/// is what is in hand now and provides the enchantment bonus.
pub(crate) fn weapon_for(base_item: Item, held: Item, sharpness: u32) -> Weapon {
    let c = &cfg().combat;
    let (base, recharge, disables, sweeps) = match base_item {
        Item::Sword => (c.sword_base_damage, c.attack_recharge_seconds, false, true),
        Item::Axe => (c.axe_base_damage, c.axe_attack_recharge_seconds, true, false),
        Item::Pickaxe => (c.pickaxe_base_damage, 0.5, false, false),
        // fist / bow / anything else
        _ => (1.0, 0.25, false, false),
    };
    // Sharpness lives on the sword only (the UHC axe/pick carry Efficiency).
    let sharpness_bonus = if held == Item::Sword && sharpness > 0 {
        c.sharpness_per_level * sharpness as f32 + 0.5
    } else {
        0.0
    };
    Weapon {
        base_damage: base,
        sharpness_bonus,
        recharge_seconds: recharge,
        disables_shield: disables,
        sweeps,
    }
}

/// Resolve player `i`'s melee swing this tick: eye-ray target pick, damage +
/// knockback, the sprint-reset + attack exhaustion, and the sweep AoE.
#[allow(clippy::needless_range_loop)]
pub(crate) fn resolve_melee(
    i: usize,
    players: &mut [Player],
    world: &BlockWorld,
    terrain: &Terrain,
    ev: &mut [StepEvents],
) {
    let n = players.len();
    let combat_cfg = &cfg().combat;

    let atk = &players[i];
    let eye = Vec3::new(atk.pos.x, atk.pos.y + EYE_HEIGHT, atk.pos.z);
    let look = look_direction(atk.yaw, atk.pitch);
    // Vanilla melee hit detection is client-side: the attacker raycasts
    // against its own (latency-stale) view of everyone else, so a laggy
    // swing lands where the target *was*, not where the server has it now.
    let atk_ping = atk.ping_ms;
    let mut target: Option<(usize, f32)> = None;
    for j in 0..n {
        if j == i || !players[j].alive() {
            continue;
        }
        let hitbox = players[j].delayed_hitbox(atk_ping).inflate(combat_cfg.hitbox_expansion);
        if let Some(t) = hitbox.ray_intersect(eye, look) {
            if t <= combat_cfg.attack_reach
                && target.is_none_or(|(_, bt)| t < bt)
                && !world.ray_blocked(eye, look, t, terrain)
            {
                target = Some((j, t));
            }
        }
    }

    // Server-side re-validation (vanilla `handleInteract`): the pick above
    // used the attacker's latency-stale view, but the server now checks the
    // chosen target's *current* hitbox against the attacker's *current* eye.
    // A target that sprinted out of range during the round-trip takes
    // nothing - only the swing itself (and its attack-cooldown reset below)
    // happened. `eye` is unchanged this tick, so reuse it.
    if let Some((tj, _)) = target {
        let live_box = Aabb::player_box(players[tj].pos, players[tj].hitbox_height());
        if live_box.distance_to_point(eye) > combat_cfg.server_interact_range {
            target = None;
        }
    }

    let strength_time = players[i].time_since_last_attack;
    players[i].time_since_last_attack = 0.0;

    let Some((tj, _)) = target else { return };

    // Attribute swap (MC-28289): a same-tick slot switch makes the hit
    // resolve with the *previously* held item's damage / speed / traits.
    let swapped = cfg().attribute_swapping && players[i].held_changed_this_step;
    let base_item = if swapped { players[i].prev_held } else { players[i].held };
    let weapon = weapon_for(base_item, players[i].held, players[i].sharpness);

    let attacker_state = AttackerState {
        pos: players[i].pos,
        vel_y: players[i].vel.y,
        on_ground: players[i].on_ground,
        sprinting: players[i].sprinting,
        in_water: world.player_contact(players[i].pos).water,
        dist_moved_last_tick: players[i].dist_moved_last_tick,
        time_since_last_attack: strength_time,
        effect_damage_add: players[i].effects.attack_damage_add(),
    };
    let same_team = players[i].team == players[tj].team;
    let friendly_fire = cfg().friendly_fire;

    let target_state = players[tj].target_state();
    let result = resolve_attack(&attacker_state, &weapon, &target_state);
    ev[i].swept = result.hit_type == HitType::Sweep;

    // Vanilla: a sprint-attack ends your sprint - you re-press it for the
    // next one (this is why W-tapping exists). Any landed hit costs 0.1
    // exhaustion.
    if result.hit_type == HitType::SprintKnockback {
        players[i].sprinting = false;
    }
    add_exhaustion(&mut players[i], combat_cfg.attack_exhaustion);

    if same_team && !friendly_fire {
        return;
    }

    if result.disable_shield {
        let tp = &mut players[tj];
        tp.shield_disabled_time = cfg().combat.axe_shield_disable_seconds;
        tp.shield_time = 0.0;
    }

    let (applied, new_last, refresh) =
        apply_iframes(result.full_damage, target_state.hurt_time_left, target_state.last_damage);
    if applied > 0.0 {
        players[tj].apply_hit(applied, new_last, refresh);
        if same_team {
            ev[i].friendly_damage += applied;
        } else {
            ev[i].damage_dealt += applied;
        }
        ev[tj].damage_taken += applied;
        if !result.blocked {
            // Fire Aspect (sword): a landed hit ignites the target.
            let fa = players[i].fire_aspect;
            if fa > 0 {
                let secs = fa as f32 * combat_cfg.fire_aspect_seconds_per_level;
                players[tj].burn_time_left = players[tj].burn_time_left.max(secs);
            }
            let (attacker, victim) = pair_mut(players, i, tj);
            apply_knockback(attacker, victim, result.hit_type);
        }
    }

    if result.triggers_sweep {
        let tp = players[tj].pos;
        let sr = cfg().combat.sweep_range;
        let sweep_box = Aabb {
            min: Vec3::new(tp.x - PLAYER_RADIUS - sr, tp.y - 0.25, tp.z - PLAYER_RADIUS - sr),
            max: Vec3::new(tp.x + PLAYER_RADIUS + sr, tp.y + PLAYER_HEIGHT + 0.25, tp.z + PLAYER_RADIUS + sr),
        };
        for k in 0..n {
            if k == i || k == tj || !players[k].alive() {
                continue;
            }
            let k_ally = players[k].team == players[i].team;
            if k_ally && !friendly_fire {
                continue;
            }
            if !aabb_overlap(&Aabb::player_at(players[k].pos), &sweep_box) {
                continue;
            }
            let ks = players[k].target_state();
            let raw = resolve_sweep_hit(ks.armor);
            let (dmg, nl, refresh) = apply_iframes(raw, ks.hurt_time_left, ks.last_damage);
            if dmg > 0.0 {
                players[k].apply_hit(dmg, nl, refresh);
                if k_ally {
                    ev[i].friendly_damage += dmg;
                } else {
                    ev[i].damage_dealt += dmg;
                }
                ev[k].damage_taken += dmg;
                let (attacker, victim) = pair_mut(players, i, k);
                apply_sweep_knockback(attacker, victim);
            }
        }
    }
}

fn apply_knockback(attacker: &mut Player, target: &mut Player, hit_type: HitType) {
    let dx = attacker.pos.x - target.pos.x;
    let dz = attacker.pos.z - target.pos.z;
    let dist = (dx * dx + dz * dz).sqrt();
    if dist < 1e-4 {
        return;
    }
    let kb = cfg().combat;
    let nx = dx / dist;
    let nz = dz / dist;
    // Knockback enchant (sword): extra push along the same direction.
    let horiz = kb.base_knockback + attacker.knockback as f32 * kb.knockback_enchant_per_level;
    // Knockback Resistance (armour attribute): scales the whole impulse.
    let resist = (1.0 - target.armor.knockback_resistance).clamp(0.0, 1.0);
    target.vel.x = target.vel.x / 2.0 - nx * horiz * resist;
    target.vel.z = target.vel.z / 2.0 - nz * horiz * resist;
    if target.on_ground {
        target.vel.y = (target.vel.y / 2.0 + kb.base_knockback * resist).min(kb.knockback_vertical_cap);
        target.on_ground = false;
    }
    if hit_type == HitType::SprintKnockback {
        let fwd = attacker.forward();
        target.vel.x += fwd.x * kb.sprint_knockback_bonus * resist;
        target.vel.z += fwd.z * kb.sprint_knockback_bonus * resist;
        target.vel.y += kb.sprint_knockback_vertical_bonus * resist;
        target.on_ground = false;
        attacker.vel.x *= 0.6;
        attacker.vel.z *= 0.6;
        attacker.sprinting = false;
    }
}

fn apply_sweep_knockback(attacker: &mut Player, target: &mut Player) {
    let dx = target.pos.x - attacker.pos.x;
    let dz = target.pos.z - attacker.pos.z;
    let dist = (dx * dx + dz * dz).sqrt();
    if dist < 1e-4 {
        return;
    }
    let resist = (1.0 - target.armor.knockback_resistance).clamp(0.0, 1.0);
    let s = cfg().combat.sweep_knockback * resist;
    let nx = dx / dist;
    let nz = dz / dist;
    target.vel.x = target.vel.x / 2.0 + nx * s;
    target.vel.z = target.vel.z / 2.0 + nz * s;
    if target.on_ground {
        target.vel.y = (target.vel.y / 2.0 + s).min(cfg().combat.knockback_vertical_cap);
        target.on_ground = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond() -> Armor {
        Armor { points: 20.0, toughness: 8.0, protection_epf: 0.0, knockback_resistance: 0.0 }
    }

    fn atk() -> AttackerState {
        AttackerState {
            pos: Vec3::ZERO,
            vel_y: 0.0,
            on_ground: true,
            sprinting: false,
            in_water: false,
            dist_moved_last_tick: 0.0,
            time_since_last_attack: crate::player::FULLY_CHARGED,
            effect_damage_add: 0.0,
        }
    }

    fn tgt() -> TargetState {
        TargetState {
            pos: Vec3::new(0.0, 0.0, 2.0),
            yaw: 0.0,
            shield_up: false,
            hurt_time_left: 0.0,
            last_damage: 0.0,
            armor: diamond(),
        }
    }

    #[test]
    fn armor_reduction_is_deterministic_and_drops_the_pre_1_9_roll() {
        // Full diamond, no Protection: two identical calls must agree exactly
        // (the old formula rolled `nextFloat()*0.5+0.5` and would not).
        let a = apply_armor_reduction(10.0, diamond());
        let b = apply_armor_reduction(10.0, diamond());
        assert_eq!(a, b);
        assert!(a < 10.0 && a > 0.0);
    }

    #[test]
    fn epf_only_bypasses_armor_points() {
        let armored = Armor { protection_epf: 10.0, ..diamond() };
        // 10 EPF -> 1 - 10/25 = 0.6
        assert!((apply_epf_only(5.0, armored) - 3.0).abs() < 1e-4);
        // and armor points do nothing here
        assert_eq!(apply_epf_only(5.0, diamond()), 5.0);
    }

    #[test]
    fn iframes_partial_rule_only_applies_in_the_first_half_and_does_not_refresh() {
        let w = cfg().combat.hurt_invulnerability_seconds;
        // Fresh window, a bigger follow-up: only the difference lands, timer
        // is NOT refreshed.
        let (dmg, last, refresh) = apply_iframes(8.0, w, 5.0);
        assert_eq!((dmg, last, refresh), (3.0, 8.0, false));
        // A weaker follow-up in the first half: nothing.
        assert_eq!(apply_iframes(4.0, 0.9 * w, 5.0), (0.0, 5.0, false));
        // Past the halfway mark: a hit lands in full and refreshes.
        let (dmg, last, refresh) = apply_iframes(4.0, 0.4 * w, 5.0);
        assert_eq!((dmg, last, refresh), (4.0, 4.0, true));
    }

    #[test]
    fn only_a_sword_triggers_the_sweep_aoe() {
        let sword = weapon_for(Item::Sword, Item::Sword, 0);
        let axe = weapon_for(Item::Axe, Item::Axe, 0);
        let fist = weapon_for(Item::Empty, Item::Empty, 0);
        assert!(resolve_attack(&atk(), &sword, &tgt()).triggers_sweep);
        assert!(!resolve_attack(&atk(), &axe, &tgt()).triggers_sweep);
        assert!(!resolve_attack(&atk(), &fist, &tgt()).triggers_sweep);
    }

    fn knockback_pair(atk_knockback: u32, target_resist: f32) -> (Player, Player) {
        let mut a = Player::new(0);
        a.pos = Vec3::new(0.0, 0.0, 0.0);
        a.knockback = atk_knockback;
        let mut t = Player::new(1);
        t.pos = Vec3::new(0.0, 0.0, 2.0); // 2 blocks in +z from the attacker
        t.vel = Vec3::ZERO;
        t.on_ground = false; // isolate the horizontal push
        t.armor.knockback_resistance = target_resist;
        (a, t)
    }

    #[test]
    fn knockback_enchant_adds_push_and_resistance_removes_it() {
        let plain = {
            let (mut a, mut t) = knockback_pair(0, 0.0);
            apply_knockback(&mut a, &mut t, HitType::Sweep);
            t.vel.z.abs()
        };
        let enchanted = {
            let (mut a, mut t) = knockback_pair(2, 0.0);
            apply_knockback(&mut a, &mut t, HitType::Sweep);
            t.vel.z.abs()
        };
        let resisted = {
            let (mut a, mut t) = knockback_pair(2, 1.0); // full resistance
            apply_knockback(&mut a, &mut t, HitType::Sweep);
            t.vel.z.abs()
        };
        assert!(enchanted > plain, "Knockback II pushes harder: {enchanted} vs {plain}");
        assert!(resisted < 1e-6, "full Knockback Resistance cancels the push: {resisted}");
    }

    #[test]
    fn fire_aspect_ignites_a_melee_target() {
        // Drive a real swing: attacker with Fire Aspect 1, target in reach.
        let mut players = [Player::new(0), Player::new(1)];
        players[0].pos = Vec3::new(0.0, 0.0, 0.0);
        players[0].held = Item::Sword;
        players[0].fire_aspect = 1;
        players[0].time_since_last_attack = crate::player::FULLY_CHARGED;
        players[1].pos = Vec3::new(0.0, 0.0, 2.0);
        players[1].armor = diamond();
        let world = BlockWorld::default();
        let terrain = Terrain::flat();
        let mut ev = [StepEvents::default(); 2];
        resolve_melee(0, &mut players, &world, &terrain, &mut ev);
        assert!(ev[0].damage_dealt > 0.0, "the hit landed");
        assert!(
            players[1].burn_time_left >= cfg().combat.fire_aspect_seconds_per_level - 1e-3,
            "target is on fire for a Fire Aspect duration: {}",
            players[1].burn_time_left
        );
    }
}
