//! Physics constants and helpers for the PvP arena, modeled directly on
//! vanilla Minecraft's own per-tick movement algorithm (see
//! `LivingEntity.travel` / `getFrictionInfluencedSpeed` in the game's
//! source) rather than a from-scratch approximation: acceleration is added
//! once per tick, then horizontal velocity is scaled by a drag constant
//! every tick (ground drag = block slipperiness * 0.91, air drag = 0.91),
//! so top speed emerges as the accel/drag equilibrium exactly like vanilla,
//! instead of being a separately hand-tuned max-speed clamp.

/// Fixed simulation timestep, in seconds. One step == one Minecraft tick,
/// so all the per-tick constants below (accel, drag, gravity) can be
/// applied directly once per step with no unit conversion. The server
/// never sleeps between steps - only *simulated* time advances in fixed
/// increments, which keeps physics stable regardless of host speed.
pub const DT: f32 = 1.0 / 20.0; // 20 simulated ticks/sec, same cadence as MC

/// Vanilla player gravity: blocks/tick^2 subtracted from vertical velocity
/// every tick.
pub const GRAVITY_PER_TICK: f32 = 0.08;
/// Vanilla's vertical drag multiplier, applied to vertical velocity every
/// tick (after gravity) regardless of ground/air state.
pub const Y_DRAG: f32 = 0.98;
/// Vanilla jump impulse, blocks/tick. Combined with `GRAVITY_PER_TICK` and
/// `Y_DRAG` this reproduces vanilla's real ~1.25 block jump height.
pub const JUMP_VELOCITY_PER_TICK: f32 = 0.42;
/// Extra horizontal speed added in the jump's facing direction when
/// jumping while sprinting (vanilla's "sprint-jump" boost).
pub const SPRINT_JUMP_BOOST: f32 = 0.2;

/// Default block slipperiness (grass/stone/etc, i.e. not ice or a slime
/// block - the arena has no special block types).
pub const GROUND_SLIPPERINESS: f32 = 0.6;
/// Horizontal velocity multiplier applied every tick while grounded:
/// `slipperiness * 0.91`, vanilla's own combined ground-drag term.
pub const GROUND_DRAG: f32 = GROUND_SLIPPERINESS * 0.91;
/// Horizontal velocity multiplier applied every tick while airborne.
pub const AIR_DRAG: f32 = 0.91;

/// Vanilla base movement speed attribute, blocks/tick, under default
/// (0.6) friction - `getFrictionInfluencedSpeed` reduces to exactly this
/// value since `0.1 * (0.6/0.6)^3 == 0.1`.
pub const WALK_ACCEL_PER_TICK: f32 = 0.1;
/// Vanilla sprinting is a +30% speed-attribute modifier over walking.
pub const SPRINT_ACCEL_PER_TICK: f32 = WALK_ACCEL_PER_TICK * 1.3;
/// Vanilla's air-control constant (`flyingSpeed` for non-flying entities) -
/// much weaker than ground accel, which is why strafing/reversing direction
/// mid-air barely changes your trajectory compared to on the ground.
pub const AIR_ACCEL_PER_TICK: f32 = 0.02;

/// Vanilla's real terminal fall velocity for an entity with no built-in
/// fall-speed cap (blocks/tick), derived from the gravity/drag equilibrium
/// above: `v = (v - GRAVITY_PER_TICK) * Y_DRAG` solved for `v`.
pub const TERMINAL_VELOCITY_PER_TICK: f32 = -3.92;

/// Default circular-platform radius. Runtime code reads `cfg().arena_radius`
/// (which this is the default for); kept as a constant for the unit tests,
/// which assume the default config.
#[allow(dead_code)]
pub const ARENA_RADIUS: f32 = 12.0;
#[allow(dead_code)] // y=0 is still the flat-terrain baseline; used directly by unit tests
pub const GROUND_Y: f32 = 0.0;
/// Vanilla player hitbox height, used for the fall-damage/AABB-vs-AABB
/// collision checks and for eye height below.
pub const PLAYER_HEIGHT: f32 = 1.8;
/// Vanilla player hitbox half-width (hitbox is 0.6 wide).
pub const PLAYER_RADIUS: f32 = 0.3;
/// Vanilla player eye height (used as the origin for aim/reach checks,
/// not the feet position).
pub const EYE_HEIGHT: f32 = 1.62;
/// Vanilla fall damage is waived for falls up to this many blocks.
pub const FALL_DAMAGE_SAFE_DISTANCE: f32 = 3.0;

pub const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Vanilla `maxUpStep` for a player: the height of a step (slab, one block of
/// terrain) you walk up without jumping.
pub const MAX_UP_STEP: f32 = 0.6;
/// Collision slack (vanilla `Shapes.EPSILON` / `AABB.clip` fudge).
pub const COLLISION_EPSILON: f32 = 1e-7;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    #[allow(dead_code)] // used by tests and kept as a general-purpose helper
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[allow(dead_code)] // used by unit tests to compare knockback magnitudes
    pub fn horizontal_length(&self) -> f32 {
        (self.x * self.x + self.z * self.z).sqrt()
    }
}

/// Clamp a value into `[lo, hi]`.
pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

/// Two distinct elements of a slice, both mutably (e.g. an attacker and its
/// target in the same `players` vec).
pub fn pair_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j, "pair_mut needs distinct indices");
    if i < j {
        let (lo, hi) = v.split_at_mut(j);
        (&mut lo[i], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(i);
        (&mut hi[0], &mut lo[j])
    }
}

/// Axis-aligned bounding box, `min` corner to `max` corner (world space).
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// The vanilla player hitbox for a player whose feet are at `feet`:
    /// `PLAYER_RADIUS` to either side horizontally, `PLAYER_HEIGHT` tall.
    pub fn player_at(feet: Vec3) -> Self {
        Self::player_box(feet, PLAYER_HEIGHT)
    }

    /// A player hitbox with an explicit height (`PLAYER_HEIGHT` standing,
    /// `combat.sneak_hitbox_height` crouched).
    pub fn player_box(feet: Vec3, height: f32) -> Self {
        Aabb {
            min: Vec3::new(feet.x - PLAYER_RADIUS, feet.y, feet.z - PLAYER_RADIUS),
            max: Vec3::new(feet.x + PLAYER_RADIUS, feet.y + height, feet.z + PLAYER_RADIUS),
        }
    }

    /// Grow the box by `pad` on every axis (vanilla `AABB.inflate`).
    pub fn inflate(self, pad: f32) -> Self {
        Aabb {
            min: Vec3::new(self.min.x - pad, self.min.y - pad, self.min.z - pad),
            max: Vec3::new(self.max.x + pad, self.max.y + pad, self.max.z + pad),
        }
    }

    /// Translate the box by `d`.
    pub fn translated(self, d: Vec3) -> Self {
        Aabb {
            min: Vec3::new(self.min.x + d.x, self.min.y + d.y, self.min.z + d.z),
            max: Vec3::new(self.max.x + d.x, self.max.y + d.y, self.max.z + d.z),
        }
    }

    /// The smallest box containing both `self` and `other` (vanilla `minmax`).
    pub fn union(self, other: Aabb) -> Self {
        Aabb {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Euclidean distance from `p` to the nearest point on (or in) the box.
    /// `0.0` when `p` is inside. Vanilla's `AABB.distanceToSqr` without the
    /// square, used for the server-side interact-range check.
    pub fn distance_to_point(&self, p: Vec3) -> f32 {
        let dx = (self.min.x - p.x).max(0.0).max(p.x - self.max.x);
        let dy = (self.min.y - p.y).max(0.0).max(p.y - self.max.y);
        let dz = (self.min.z - p.z).max(0.0).max(p.z - self.max.z);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Ray/AABB intersection by the slab method - vanilla's own
    /// `AABB.clip(from, to)` picking algorithm. `origin` is the ray start,
    /// `dir` a (not necessarily unit) direction. Returns the parametric
    /// distance `t` along `dir` at which the ray first enters the box, or
    /// `None` if it never does. `t == 0.0` when `origin` is already inside.
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        if self.contains(origin) {
            return Some(0.0);
        }
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;
        for (o, d, lo, hi) in [
            (origin.x, dir.x, self.min.x, self.max.x),
            (origin.y, dir.y, self.min.y, self.max.y),
            (origin.z, dir.z, self.min.z, self.max.z),
        ] {
            if d.abs() < 1e-8 {
                // Ray parallel to this slab: miss unless the origin is
                // already between the slab planes.
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let mut t1 = (lo - o) * inv;
                let mut t2 = (hi - o) * inv;
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
            None // box is entirely behind the ray origin
        } else {
            Some(t_min.max(0.0))
        }
    }
}

/// Do two AABBs overlap on all three axes?
pub fn aabb_overlap(a: &Aabb, b: &Aabb) -> bool {
    a.min.x < b.max.x
        && a.max.x > b.min.x
        && a.min.y < b.max.y
        && a.max.y > b.min.y
        && a.min.z < b.max.z
        && a.max.z > b.min.z
}

/// Unit look direction from yaw+pitch, MC's convention: yaw=0/pitch=0 faces
/// +z, positive pitch looks down.
pub fn look_direction(yaw: f32, pitch: f32) -> Vec3 {
    let (sin_y, cos_y) = yaw.sin_cos();
    let (sin_p, cos_p) = pitch.sin_cos();
    Vec3::new(-sin_y * cos_p, -sin_p, cos_y * cos_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_box_dead_ahead() {
        let b = Aabb::player_at(Vec3::new(0.0, 0.0, 3.0));
        let t = b.ray_intersect(Vec3::new(0.0, EYE_HEIGHT, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(t.is_some());
        assert!((t.unwrap() - (3.0 - PLAYER_RADIUS)).abs() < 1e-3);
    }

    #[test]
    fn ray_misses_box_off_to_the_side() {
        let b = Aabb::player_at(Vec3::new(5.0, 0.0, 3.0));
        assert!(b
            .ray_intersect(Vec3::new(0.0, EYE_HEIGHT, 0.0), Vec3::new(0.0, 0.0, 1.0))
            .is_none());
    }

    #[test]
    fn ray_from_inside_box_is_distance_zero() {
        let b = Aabb::player_at(Vec3::ZERO);
        assert_eq!(
            b.ray_intersect(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            Some(0.0)
        );
    }

    #[test]
    fn ray_pointing_away_from_box_misses() {
        let b = Aabb::player_at(Vec3::new(0.0, 0.0, 3.0));
        assert!(b
            .ray_intersect(Vec3::new(0.0, EYE_HEIGHT, 0.0), Vec3::new(0.0, 0.0, -1.0))
            .is_none());
    }
}
