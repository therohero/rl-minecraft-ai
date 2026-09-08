//! In-flight arrows and crossbow bolts: spawning from a released bow /
//! fired crossbow, per-tick ballistic flight (gravity + drag), swept-ray
//! collision against players and blocks, damage through armour / shields /
//! i-frames, Piercing, and knockback.

use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::Rng;

use crate::arena::StepEvents;
use crate::blocks::BlockWorld;
use crate::combat::{self};
use crate::config::cfg;
use crate::physics::{look_direction, Aabb, Vec3, EYE_HEIGHT};
use crate::player::{delay_back, Player, MAX_DELAY_TICKS};
use crate::terrain::Terrain;

/// A shot the policy queued this tick (a released bow / fired crossbow),
/// consumed by `Arena::step` which owns the projectile list.
#[derive(Clone, Copy)]
pub struct PendingShot {
    pub speed: f32,
    /// Damage-per-unit-of-speed (vanilla `AbstractArrow::baseDamage`, 2.0,
    /// plus any Power bonus). Impact damage is `ceil(speed_at_impact * this)`.
    pub base_damage: f32,
    pub piercing: i32,
    pub crossbow: bool,
}

#[derive(Clone)]
pub struct Projectile {
    pub pos: Vec3,
    pub vel: Vec3,
    pub owner: usize,
    pub owner_team: u8,
    /// Damage-per-unit-of-speed; the hit reads the arrow's *current* speed,
    /// so a drag-slowed long shot lands softer (vanilla behaviour).
    pub base_damage: f32,
    pub piercing_left: i32,
    pub hit: Vec<usize>,
    pub life_ticks: u32,
    /// Last `MAX_DELAY_TICKS + 1` (pos, vel) snapshots so a laggy observer
    /// sees the arrow where it was, not where it is (newest last).
    pub history: VecDeque<(Vec3, Vec3)>,
}

impl Projectile {
    pub(crate) fn push_snapshot(&mut self) {
        self.history.push_back((self.pos, self.vel));
        while self.history.len() > MAX_DELAY_TICKS + 1 {
            self.history.pop_front();
        }
    }

    /// (pos, vel) as seen by an observer with `observer_ping_ms` of latency.
    pub(crate) fn delayed_view(&self, observer_ping_ms: f32) -> (Vec3, Vec3) {
        if self.history.is_empty() {
            return (self.pos, self.vel);
        }
        let back = delay_back(self.history.len(), observer_ping_ms);
        self.history[self.history.len() - 1 - back]
    }
}

/// Vanilla `RandomSource::triangle(mode, deviation)` =
/// `mode + deviation * (rand() - rand())` - a symmetric triangular distribution.
fn triangle(rng: &mut StdRng, deviation: f32) -> f32 {
    deviation * (rng.gen::<f32>() - rng.gen::<f32>())
}

/// Spawn a projectile from `shooter`'s eye for a queued `shot`, consuming
/// the arrow / crossbow charge. A no-op if a bow shot has no ammo left.
pub(crate) fn spawn(
    projectiles: &mut Vec<Projectile>,
    shooter: &mut Player,
    owner: usize,
    shot: PendingShot,
    rng: &mut StdRng,
) {
    if shot.crossbow {
        shooter.crossbow_loaded = false;
    } else if shooter.arrows == 0 {
        return;
    } else {
        shooter.arrows -= 1;
    }
    let eye = Vec3::new(shooter.pos.x, shooter.pos.y + EYE_HEIGHT, shooter.pos.z);
    let dir = look_direction(shooter.yaw, shooter.pitch);
    // Vanilla `AbstractArrow::shoot`: normalized aim + a per-axis triangular
    // deviation of `0.0172275 * inaccuracy`, then scaled by speed.
    let spread = 0.0172275 * cfg().combat.projectile_inaccuracy;
    let mut vel = Vec3::new(
        (dir.x + triangle(rng, spread)) * shot.speed,
        (dir.y + triangle(rng, spread)) * shot.speed,
        (dir.z + triangle(rng, spread)) * shot.speed,
    );
    // Vanilla `Projectile::shootFromRotation` inherits the shooter's motion -
    // horizontally always, vertically only while the shooter is airborne.
    vel.x += shooter.vel.x;
    vel.z += shooter.vel.z;
    if !shooter.on_ground {
        vel.y += shooter.vel.y;
    }
    projectiles.push(Projectile {
        pos: eye,
        vel,
        owner,
        owner_team: shooter.team,
        base_damage: shot.base_damage,
        piercing_left: shot.piercing,
        hit: Vec::new(),
        life_ticks: 0,
        history: VecDeque::new(),
    });
}

/// Advance every projectile one tick and resolve any hits. Dead projectiles
/// (grounded, off-platform, expired, spent) are removed.
#[allow(clippy::needless_range_loop)]
pub(crate) fn step_all(
    projectiles: &mut Vec<Projectile>,
    players: &mut [Player],
    world: &BlockWorld,
    terrain: &Terrain,
    ev: &mut [StepEvents],
) {
    let n = players.len();
    let c = &cfg().combat;
    let arena_r = cfg().arena_radius + 2.0;
    let mut idx = 0;
    while idx < projectiles.len() {
        let (old, new_pos) = {
            let pr = &mut projectiles[idx];
            pr.vel.y -= c.arrow_gravity;
            pr.vel.x *= c.arrow_drag;
            pr.vel.y *= c.arrow_drag;
            pr.vel.z *= c.arrow_drag;
            let old = pr.pos;
            pr.pos = Vec3::new(pr.pos.x + pr.vel.x, pr.pos.y + pr.vel.y, pr.pos.z + pr.vel.z);
            pr.life_ticks += 1;
            (old, pr.pos)
        };
        let seg = Vec3::new(new_pos.x - old.x, new_pos.y - old.y, new_pos.z - old.z);

        // Terrain / placed block in the way / arena bounds / lifetime.
        let r2 = new_pos.x * new_pos.x + new_pos.z * new_pos.z;
        let mut dead = world.is_solid_cell(crate::blocks::Cell::of(new_pos), terrain)
            || r2 > arena_r * arena_r
            || projectiles[idx].life_ticks > 200
            || world.ray_blocked(old, seg, 1.0, terrain);

        if !dead {
            // Nearest player the segment enters this tick.
            let owner = projectiles[idx].owner;
            let mut best: Option<(usize, f32)> = None;
            for j in 0..n {
                if j == owner || !players[j].alive() || projectiles[idx].hit.contains(&j) {
                    continue;
                }
                let box_j = Aabb::player_at(players[j].pos).inflate(0.3);
                if let Some(t) = box_j.ray_intersect(old, seg) {
                    if t <= 1.0 && best.is_none_or(|(_, bt)| t < bt) {
                        best = Some((j, t));
                    }
                }
            }
            if let Some((j, _)) = best {
                let v = projectiles[idx].vel;
                let speed = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
                let pr_dmg = (speed * projectiles[idx].base_damage).ceil().max(0.0);
                let pr_team = projectiles[idx].owner_team;
                let same_team = players[j].team == pr_team;
                let ff = cfg().friendly_fire;
                if !same_team || ff {
                    let ts = players[j].target_state();
                    // Shields catch arrows within the frontal arc too.
                    let src = projectiles[idx].pos;
                    let arrow_from = Vec3::new(src.x - seg.x, src.y - seg.y, src.z - seg.z);
                    let blocked = combat::shield_blocks(&ts, arrow_from);
                    let mut raw = combat::apply_armor_reduction(pr_dmg, ts.armor);
                    if blocked {
                        raw *= 1.0 - c.shield_damage_block;
                    }
                    let (applied, nl, refresh) =
                        combat::apply_iframes(raw, ts.hurt_time_left, ts.last_damage);
                    if applied > 0.0 {
                        players[j].apply_hit(applied, nl, refresh);
                        if same_team {
                            ev[owner].friendly_damage += applied;
                        } else {
                            ev[owner].damage_dealt += applied;
                        }
                        ev[j].damage_taken += applied;
                        if !blocked {
                            let sp = (v.x * v.x + v.z * v.z).sqrt().max(1e-4);
                            let p = &mut players[j];
                            p.vel.x += v.x / sp * c.base_knockback;
                            p.vel.z += v.z / sp * c.base_knockback;
                            if p.on_ground {
                                p.vel.y = (p.vel.y + c.base_knockback * 0.6).min(c.knockback_vertical_cap);
                                p.on_ground = false;
                            }
                        }
                    }
                }
                let pr = &mut projectiles[idx];
                pr.hit.push(j);
                pr.piercing_left -= 1;
                if pr.piercing_left < 0 {
                    dead = true;
                }
            }
        }

        if dead {
            projectiles.swap_remove(idx);
        } else {
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn shooter() -> Player {
        let mut p = Player::new(0);
        p.pos = Vec3::ZERO;
        p.yaw = 0.0; // faces +z
        p.pitch = 0.0;
        p.on_ground = true;
        p.arrows = 8;
        p
    }

    fn full_bow_shot() -> PendingShot {
        PendingShot {
            speed: cfg().combat.arrow_max_speed,
            base_damage: cfg().combat.arrow_damage_per_speed,
            piercing: 0,
            crossbow: false,
        }
    }

    #[test]
    fn a_moving_shooter_lends_the_arrow_its_horizontal_velocity() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut s = shooter();
        s.vel = Vec3::new(0.5, 0.3, 0.0);
        let mut ps = Vec::new();
        spawn(&mut ps, &mut s, 0, full_bow_shot(), &mut rng);
        let v = ps[0].vel;
        // +z aim at speed 3.0; x picks up the full 0.5, y picks up nothing
        // (grounded), spread is small.
        assert!((v.x - 0.5).abs() < 0.1, "x inherits shooter motion: {}", v.x);
        assert!(v.y.abs() < 0.1, "grounded: no vertical inheritance: {}", v.y);
        assert!(v.z > 2.5);
    }

    #[test]
    fn spread_is_random_but_small() {
        let mut rng = StdRng::seed_from_u64(7);
        let s = shooter();
        let mut zs = Vec::new();
        for _ in 0..32 {
            let mut sh = s.clone();
            let mut ps = Vec::new();
            spawn(&mut ps, &mut sh, 0, full_bow_shot(), &mut rng);
            zs.push((ps[0].vel.x, ps[0].vel.y));
        }
        let spread_x = zs.iter().map(|&(x, _)| x.abs()).fold(0.0_f32, f32::max);
        assert!(spread_x > 0.0, "there is some spread");
        assert!(spread_x < 0.2, "but it is bounded: {spread_x}");
        assert!(zs.iter().any(|&(x, _)| x != zs[0].0), "not identical every shot");
    }

    #[test]
    fn a_slower_arrow_hits_softer() {
        // Two arrows straight at a target, one full-draw one half; the
        // faster one must carry more impact damage.
        let hit_dmg = |speed: f32| -> f32 {
            let mut rng = StdRng::seed_from_u64(3);
            let mut atk = shooter();
            let mut tgt = Player::new(1);
            tgt.pos = Vec3::new(0.0, 0.0, 4.0);
            tgt.armor.points = 0.0;
            tgt.armor.protection_epf = 0.0;
            let mut players = [atk.clone(), tgt];
            let mut ps = Vec::new();
            let shot = PendingShot { speed, ..full_bow_shot() };
            spawn(&mut ps, &mut atk, 0, shot, &mut rng);
            let terrain = Terrain::flat();
            let world = BlockWorld::default();
            let mut ev = [StepEvents::default(); 2];
            for _ in 0..40 {
                step_all(&mut ps, &mut players, &world, &terrain, &mut ev);
            }
            ev[1].damage_taken
        };
        let fast = hit_dmg(3.0);
        let slow = hit_dmg(1.5);
        assert!(fast > slow && slow > 0.0, "fast {fast} vs slow {slow}");
    }
}
