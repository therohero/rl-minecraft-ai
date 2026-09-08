//! Vanilla per-axis AABB collision against the voxel world (`Entity.move` /
//! `Entity.collide`), with the 0.6-block `maxUpStep` auto step-up.
//!
//! The player box is swept by the intended `delta`; every solid unit cell
//! the swept box (grown by the step height) overlaps becomes a collider, and
//! the move is resolved one axis at a time - **Y first** (it decides
//! `on_ground`, which gates the step-up), then the larger horizontal axis,
//! then the smaller (vanilla avoids a fixed X-then-Z corner-slide bias).

use crate::blocks::{BlockWorld, Cell};
use crate::physics::{Aabb, Vec3, COLLISION_EPSILON};
use crate::terrain::Terrain;

#[derive(Default, Clone, Copy, Debug)]
pub struct MoveResult {
    pub applied: Vec3,
    pub hit_x: bool,
    pub hit_z: bool,
    /// Clipped while moving down - landed on something.
    pub hit_y_neg: bool,
    /// Clipped while moving up - bonked a ceiling.
    pub hit_y_pos: bool,
    /// The horizontal move only succeeded via the auto step-up.
    pub stepped_up: bool,
}

/// Every solid unit-cell box overlapping `b`, appended to `out` (cleared
/// first). Only the integer cell range of `b` is visited - not the block map.
pub fn solid_boxes(world: &BlockWorld, terrain: &Terrain, b: Aabb, out: &mut Vec<Aabb>) {
    out.clear();
    let x0 = b.min.x.floor() as i32;
    let x1 = (b.max.x - COLLISION_EPSILON).floor() as i32;
    let y0 = b.min.y.floor() as i32;
    let y1 = (b.max.y - COLLISION_EPSILON).floor() as i32;
    let z0 = b.min.z.floor() as i32;
    let z1 = (b.max.z - COLLISION_EPSILON).floor() as i32;
    for cx in x0..=x1 {
        for cz in z0..=z1 {
            for cy in y0..=y1 {
                let cell = Cell::new(cx, cy, cz);
                if world.is_solid_cell(cell, terrain) {
                    out.push(cell.box_of());
                }
            }
        }
    }
}

/// Do `a` and `o` overlap on the two axes *other* than `axis`?
fn crosses(a: &Aabb, o: &Aabb, axis: usize) -> bool {
    (axis == 0 || (a.min.x < o.max.x && a.max.x > o.min.x))
        && (axis == 1 || (a.min.y < o.max.y && a.max.y > o.min.y))
        && (axis == 2 || (a.min.z < o.max.z && a.max.z > o.min.z))
}

/// Vanilla `AABB.clipOnAxis`: shorten `d` so `b` cannot pass through any box
/// in `boxes` along `axis`.
fn clip(boxes: &[Aabb], b: Aabb, axis: usize, mut d: f32) -> f32 {
    let (lo, hi) = match axis {
        0 => (b.min.x, b.max.x),
        1 => (b.min.y, b.max.y),
        _ => (b.min.z, b.max.z),
    };
    for o in boxes {
        if !crosses(&b, o, axis) {
            continue;
        }
        let (olo, ohi) = match axis {
            0 => (o.min.x, o.max.x),
            1 => (o.min.y, o.max.y),
            _ => (o.min.z, o.max.z),
        };
        if d > 0.0 && olo >= hi - COLLISION_EPSILON {
            d = d.min(olo - hi);
        } else if d < 0.0 && ohi <= lo + COLLISION_EPSILON {
            d = d.max(ohi - lo);
        }
    }
    d
}

fn moved(b: Aabb, axis: usize, d: f32) -> Aabb {
    let v = match axis {
        0 => Vec3::new(d, 0.0, 0.0),
        1 => Vec3::new(0.0, d, 0.0),
        _ => Vec3::new(0.0, 0.0, d),
    };
    b.translated(v)
}

/// Slide the player box (feet at `*feet`, `height` tall) by `delta`,
/// clipping against the voxel world, with `max_up_step` auto step-up.
/// Updates `*feet` and returns what happened.
#[allow(clippy::too_many_arguments)]
pub fn move_with_collision(
    feet: &mut Vec3,
    delta: Vec3,
    height: f32,
    on_ground: bool,
    max_up_step: f32,
    world: &BlockWorld,
    terrain: &Terrain,
    scratch: &mut Vec<Aabb>,
) -> MoveResult {
    let start = Aabb::player_box(*feet, height);
    let full = start.union(start.translated(delta)).inflate(COLLISION_EPSILON);
    // Grow downward by the step height so the step-up retry (which drops the
    // box back onto whatever it stepped onto) has its collider in scratch.
    let swept = Aabb {
        min: Vec3::new(full.min.x, full.min.y - max_up_step, full.min.z),
        max: Vec3::new(full.max.x, full.max.y + max_up_step, full.max.z),
    };
    solid_boxes(world, terrain, swept, scratch);

    // --- plain resolve: Y, then larger horizontal, then smaller ---
    let (a1, a2) = if delta.x.abs() >= delta.z.abs() { (0usize, 2usize) } else { (2, 0) };
    let mut b = start;

    let dy = clip(scratch, b, 1, delta.y);
    b = moved(b, 1, dy);
    let mut d1 = clip(scratch, b, a1, axis_of(delta, a1));
    b = moved(b, a1, d1);
    let mut d2 = clip(scratch, b, a2, axis_of(delta, a2));

    let mut res = MoveResult {
        hit_y_neg: dy != delta.y && delta.y < 0.0,
        hit_y_pos: dy != delta.y && delta.y > 0.0,
        ..Default::default()
    };

    // --- step-up retry (vanilla `Entity.collide`) ---
    let blocked = d1 != axis_of(delta, a1) || d2 != axis_of(delta, a2);
    let mut dy_total = dy;
    if max_up_step > 0.0 && blocked && (on_ground || (delta.y < 0.0 && dy != delta.y)) {
        let mut sb = start;
        let up = clip(scratch, sb, 1, max_up_step);
        sb = moved(sb, 1, up);
        let s1 = clip(scratch, sb, a1, axis_of(delta, a1));
        sb = moved(sb, a1, s1);
        let s2 = clip(scratch, sb, a2, axis_of(delta, a2));
        sb = moved(sb, a2, s2);
        let down = clip(scratch, sb, 1, -up);

        if s1 * s1 + s2 * s2 > d1 * d1 + d2 * d2 {
            d1 = s1;
            d2 = s2;
            dy_total = up + down;
            res.stepped_up = true;
            res.hit_y_neg = down < -COLLISION_EPSILON;
        }
    }

    let (mx, mz) = match a1 {
        0 => (d1, d2),
        _ => (d2, d1),
    };
    res.hit_x = mx != delta.x;
    res.hit_z = mz != delta.z;
    res.applied = Vec3::new(mx, dy_total, mz);
    feet.x += mx;
    feet.y += dy_total;
    feet.z += mz;
    res
}

fn axis_of(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Vanilla `Player.maybeBackOffFromEdge`: a sneaking, grounded player will
/// not step off a ledge - shrink the horizontal delta (per axis) until the
/// box still has solid support under a foot.
pub fn back_off_from_edge(
    feet: Vec3,
    dx: &mut f32,
    dz: &mut f32,
    world: &BlockWorld,
    terrain: &Terrain,
) {
    const STEP: f32 = 0.05;
    let supported = |fx: f32, fz: f32| -> bool {
        // A cell directly under any corner of the (moved) foot box.
        let probe_y = feet.y - 0.55;
        for cx in [(fx - 0.3).floor() as i32, (fx + 0.3).floor() as i32] {
            for cz in [(fz - 0.3).floor() as i32, (fz + 0.3).floor() as i32] {
                if world.is_solid_cell(Cell::new(cx, probe_y.floor() as i32, cz), terrain) {
                    return true;
                }
            }
        }
        false
    };
    while *dx != 0.0 && !supported(feet.x + *dx, feet.z) {
        if dx.abs() <= STEP {
            *dx = 0.0;
        } else {
            *dx -= dx.signum() * STEP;
        }
    }
    while *dz != 0.0 && !supported(feet.x, feet.z + *dz) {
        if dz.abs() <= STEP {
            *dz = 0.0;
        } else {
            *dz -= dz.signum() * STEP;
        }
    }
}

/// Smallest-axis de-penetration for a box that has been moved *without*
/// collision (the player-player push, the arena rim clamp) into a solid
/// cell. Pushes horizontally only.
pub fn push_out_of_solids(feet: &mut Vec3, vel: &mut Vec3, height: f32, world: &BlockWorld, terrain: &Terrain) {
    let mut scratch = Vec::new();
    let b = Aabb::player_box(*feet, height);
    solid_boxes(world, terrain, b, &mut scratch);
    for o in &scratch {
        let pb = Aabb::player_box(*feet, height);
        if !(pb.min.x < o.max.x && pb.max.x > o.min.x && pb.min.z < o.max.z && pb.max.z > o.min.z) {
            continue;
        }
        if pb.min.y >= o.max.y - COLLISION_EPSILON {
            continue; // standing on top
        }
        let px = (o.max.x - pb.min.x).min(pb.max.x - o.min.x);
        let pz = (o.max.z - pb.min.z).min(pb.max.z - o.min.z);
        if px < pz {
            let dir = if (pb.min.x + pb.max.x) * 0.5 < (o.min.x + o.max.x) * 0.5 { -1.0 } else { 1.0 };
            feet.x += dir * px;
            vel.x = 0.0;
        } else {
            let dir = if (pb.min.z + pb.max.z) * 0.5 < (o.min.z + o.max.z) * 0.5 { -1.0 } else { 1.0 };
            feet.z += dir * pz;
            vel.z = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::Block;
    use crate::physics::{MAX_UP_STEP, PLAYER_HEIGHT, PLAYER_RADIUS, TERMINAL_VELOCITY_PER_TICK};

    fn world_with(cells: &[(i32, i32, i32)]) -> (BlockWorld, Terrain) {
        let mut w = BlockWorld::default();
        for &(x, y, z) in cells {
            w.place_block_for_test(crate::blocks::Cell::new(x, y, z), Block::Planks);
        }
        (w, Terrain::flat())
    }

    /// Walk `feet` forward in +z by `dz`/tick for `ticks`, with gravity.
    fn walk(feet: &mut Vec3, dz: f32, w: &BlockWorld, t: &Terrain, step: f32, ticks: usize) -> Vec<MoveResult> {
        let mut scratch = Vec::new();
        let mut on_ground = true;
        let mut vy = 0.0;
        let mut out = Vec::new();
        for _ in 0..ticks {
            vy = (vy - 0.08) * 0.98;
            let r = move_with_collision(
                feet,
                Vec3::new(0.0, vy, dz),
                PLAYER_HEIGHT,
                on_ground,
                step,
                w,
                t,
                &mut scratch,
            );
            on_ground = r.hit_y_neg;
            if r.hit_y_neg || r.hit_y_pos {
                vy = 0.0;
            }
            out.push(r);
        }
        out
    }

    #[test]
    fn walking_into_a_one_block_step_stops_you_you_must_jump() {
        // Vanilla `maxUpStep` is 0.6 - a full block is not auto-climbed.
        let (w, t) = world_with(&[(0, 0, 1)]);
        let mut feet = Vec3::new(0.5, 0.0, 0.0);
        walk(&mut feet, 0.15, &w, &t, MAX_UP_STEP, 25);
        assert!(feet.y < 0.5, "did not climb the block: y={}", feet.y);
        assert!(feet.z <= 1.0 - PLAYER_RADIUS + 1e-3, "stopped at its face: z={}", feet.z);
    }

    #[test]
    fn a_two_block_wall_also_just_stops_you() {
        let (w, t) = world_with(&[(0, 0, 1), (0, 1, 1)]);
        let mut feet = Vec3::new(0.5, 0.0, 0.0);
        walk(&mut feet, 0.15, &w, &t, MAX_UP_STEP, 30);
        assert!(feet.y < 0.5 && feet.z <= 1.0 - PLAYER_RADIUS + 1e-3, "{feet:?}");
    }

    #[test]
    fn a_jump_carries_you_onto_a_one_block_ledge() {
        // A wide 1-block ledge from z=1 on: a jump lands you on top of it.
        let (w, t) = world_with(&[(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)]);
        let mut feet = Vec3::new(0.5, 0.0, 0.0);
        let mut scratch = Vec::new();
        let mut vy = 0.42;
        let mut on_ground = false;
        for _ in 0..25 {
            let r = move_with_collision(
                &mut feet,
                Vec3::new(0.0, vy, 0.1),
                PLAYER_HEIGHT,
                on_ground,
                MAX_UP_STEP,
                &w,
                &t,
                &mut scratch,
            );
            on_ground = r.hit_y_neg;
            if r.hit_y_neg || r.hit_y_pos {
                vy = 0.0;
            }
            vy = (vy - 0.08) * 0.98;
        }
        assert!(feet.z > 1.0 && (feet.y - 1.0).abs() < 1e-3, "landed on the ledge: {feet:?}");
    }

    #[test]
    fn head_bonk_on_a_ceiling_zeroes_upward_velocity() {
        let (w, t) = world_with(&[(0, 2, 0)]); // ceiling block, bottom at y=2
        let mut feet = Vec3::new(0.5, 0.0, 0.5);
        let mut scratch = Vec::new();
        let r = move_with_collision(
            &mut feet,
            Vec3::new(0.0, 0.42, 0.0),
            PLAYER_HEIGHT,
            false,
            MAX_UP_STEP,
            &w,
            &t,
            &mut scratch,
        );
        assert!(r.hit_y_pos, "bonked the ceiling");
        assert!(feet.y + PLAYER_HEIGHT <= 2.0 + 1e-3, "stopped under it: y={}", feet.y);
    }

    #[test]
    fn no_tunnelling_through_a_thin_shelf_at_terminal_velocity() {
        let (w, t) = world_with(&[(0, 3, 0)]); // 1-cell-thick shelf, top at y=4
        let mut feet = Vec3::new(0.5, 6.0, 0.5); // one terminal-speed tick above it
        let mut scratch = Vec::new();
        let r = move_with_collision(
            &mut feet,
            Vec3::new(0.0, TERMINAL_VELOCITY_PER_TICK, 0.0),
            PLAYER_HEIGHT,
            false,
            MAX_UP_STEP,
            &w,
            &t,
            &mut scratch,
        );
        assert!(r.hit_y_neg, "landed on the shelf instead of passing through");
        assert!((feet.y - 4.0).abs() < 1e-3, "resting on top: y={}", feet.y);
    }

    #[test]
    fn a_player_pushed_into_a_wall_is_de_penetrated() {
        let (w, t) = world_with(&[(1, 0, 0)]);
        let mut feet = Vec3::new(0.85, 0.0, 0.0); // box max.x = 1.15, inside the wall
        let mut vel = Vec3::new(1.0, 0.0, 0.0);
        push_out_of_solids(&mut feet, &mut vel, PLAYER_HEIGHT, &w, &t);
        assert!(feet.x + PLAYER_RADIUS <= 1.0 + 1e-3, "pushed clear: x={}", feet.x);
        assert_eq!(vel.x, 0.0);
    }
}
