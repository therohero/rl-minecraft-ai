//! Procedurally varied arena terrain, quantized to a voxel column grid.
//!
//! Training on a perfectly flat platform teaches a policy that falls apart
//! the moment real terrain isn't flat. Every match instead gets a randomly
//! chosen ground shape - a slope, a rim-walled bowl, uneven bumps, or
//! stepped rings - so the agent has to learn to read and react to terrain.
//!
//! Vanilla Minecraft has no fractional ground: the world is unit blocks.
//! So a shape's continuous height is **floored per column** into
//! `column_top` (the y of the first air cell; every cell below is solid),
//! and the player collides with real block AABBs and auto-steps 0.6 up.
//! The `random` parameter ranges are derived from the amplitude so that no
//! two neighbouring columns ever differ by more than one block - a 2-block
//! face is unclimbable even by jumping and would trap a player.

use crate::config::cfg;
use crate::physics::clamp;
use rand::Rng;

/// Default max terrain height deviation from y=0, in blocks - the value used
/// unless a config overrides `terrain_max_amplitude`. Kept modest relative
/// to vanilla's ~1.25-block jump height. Runtime code reads
/// `cfg().terrain_max_amplitude`; this constant is the documented baseline
/// and the value the unit tests assume.
#[allow(dead_code)]
pub const MAX_AMPLITUDE: f32 = 3.0;

/// The most two 4-neighbour columns may differ after flooring. Exactly 1:
/// a step the player auto-climbs, never a wall.
const MAX_COLUMN_STEP: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
enum TerrainKind {
    Flat,
    Slope,
    Bowl,
    Bumpy,
    Steps,
}

#[derive(Clone, Copy, Debug)]
pub struct Terrain {
    kind: TerrainKind,
    amplitude: f32,
    freq_x: f32,
    freq_z: f32,
    phase_x: f32,
    phase_z: f32,
    slope_x: f32,
    slope_z: f32,
    step_size: f32,
}

impl Terrain {
    #[allow(dead_code)] // used by unit tests as a deterministic baseline shape
    pub fn flat() -> Self {
        Terrain {
            kind: TerrainKind::Flat,
            amplitude: 0.0,
            freq_x: 0.0,
            freq_z: 0.0,
            phase_x: 0.0,
            phase_z: 0.0,
            slope_x: 0.0,
            slope_z: 0.0,
            step_size: 1.0,
        }
    }

    /// Picks a fresh random terrain shape - called once per match so
    /// consecutive episodes (even within the same arena slot) vary. Every
    /// shape parameter is capped so the per-column height never steps by
    /// more than `MAX_COLUMN_STEP` between neighbours, whatever the
    /// configured amplitude.
    pub fn random(rng: &mut impl Rng) -> Self {
        let kind = match rng.gen_range(0..5) {
            0 => TerrainKind::Flat,
            1 => TerrainKind::Slope,
            2 => TerrainKind::Bowl,
            3 => TerrainKind::Bumpy,
            _ => TerrainKind::Steps,
        };
        let max_amplitude = cfg().terrain_max_amplitude.max(1.0);
        let amplitude = rng.gen_range(1.0..=max_amplitude);
        // Per-block gradient bounds (see `raw_height`): a sine bump's max
        // slope is `amp * 0.5 * freq`, a linear slope's is `|slope|`, a
        // step ring's rise is `amp / 4`. Cap each so one column-to-column
        // move changes the floored height by at most one.
        let freq_cap = (0.9 * MAX_COLUMN_STEP / (0.5 * amplitude)).min(0.4);
        let slope_cap = MAX_COLUMN_STEP.min(0.15);
        let bowl_amp = amplitude.min(0.45 * MAX_COLUMN_STEP * cfg().arena_radius);
        Terrain {
            kind,
            amplitude: if matches!(kind, TerrainKind::Bowl) { bowl_amp } else { amplitude },
            freq_x: rng.gen_range(0.05..=freq_cap.max(0.05)),
            freq_z: rng.gen_range(0.05..=freq_cap.max(0.05)),
            phase_x: rng.gen_range(0.0..std::f32::consts::TAU),
            phase_z: rng.gen_range(0.0..std::f32::consts::TAU),
            slope_x: rng.gen_range(-slope_cap..=slope_cap),
            slope_z: rng.gen_range(-slope_cap..=slope_cap),
            // A ring at least `amp/4` wide keeps each step to one floored block.
            step_size: rng.gen_range(1.5_f32..=3.0).max(amplitude / 4.0),
        }
    }

    /// The continuous generator output at a world position (blocks). Private:
    /// callers see only the voxel-quantized `column_top` / `surface_y`.
    fn raw_height(&self, x: f32, z: f32) -> f32 {
        match self.kind {
            TerrainKind::Flat => 0.0,
            TerrainKind::Slope => {
                clamp(self.slope_x * x + self.slope_z * z, -self.amplitude, self.amplitude)
            }
            TerrainKind::Bowl => {
                let dist = (x * x + z * z).sqrt();
                self.amplitude * (dist / cfg().arena_radius).min(1.0).powi(2)
            }
            TerrainKind::Bumpy => {
                self.amplitude
                    * 0.5
                    * ((self.freq_x * x + self.phase_x).sin() + (self.freq_z * z + self.phase_z).cos())
            }
            TerrainKind::Steps => {
                // Ring width >= 1.5 keeps `floor(dist/step)` to a 1-per-column
                // change; the rise per ring is capped at one block so the
                // floored `column_top` never jumps by more than one.
                let dist = (x * x + z * z).sqrt();
                let rise = (self.amplitude / 4.0).min(MAX_COLUMN_STEP);
                let raw = (dist / self.step_size).floor() * rise;
                clamp(raw, -self.amplitude, self.amplitude)
            }
        }
    }

    /// The y of the first *air* cell in column `(cx, cz)`: every cell with
    /// `y < column_top` is solid ground, all the way down (there is no
    /// void). Sampled at the column centre so the value is constant across
    /// the whole cell.
    pub fn column_top(&self, cx: i32, cz: i32) -> i32 {
        if matches!(self.kind, TerrainKind::Flat) {
            return 0;
        }
        self.raw_height(cx as f32 + 0.5, cz as f32 + 0.5).floor() as i32
    }

    /// Feet-level surface y for whichever column contains `(x, z)`.
    pub fn surface_y(&self, x: f32, z: f32) -> f32 {
        self.column_top(x.floor() as i32, z.floor() as i32) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ARENA_RADIUS;

    #[test]
    fn flat_terrain_is_always_zero() {
        let t = Terrain::flat();
        assert_eq!(t.column_top(0, 0), 0);
        assert_eq!(t.column_top(5, -3), 0);
        assert_eq!(t.surface_y(5.7, -3.2), 0.0);
    }

    #[test]
    fn random_terrain_column_tops_stay_within_bounds() {
        let mut rng = rand::thread_rng();
        let bound = MAX_AMPLITUDE.ceil() as i32 + 1;
        for _ in 0..200 {
            let t = Terrain::random(&mut rng);
            for cx in -(ARENA_RADIUS as i32)..=(ARENA_RADIUS as i32) {
                for cz in -(ARENA_RADIUS as i32)..=(ARENA_RADIUS as i32) {
                    let h = t.column_top(cx, cz);
                    assert!(h.abs() <= bound, "column_top {h} at ({cx},{cz}) for {:?}", t.kind);
                }
            }
        }
    }

    #[test]
    fn every_random_terrain_is_walkable() {
        // No two 4-neighbour columns inside the arena disc differ by more
        // than one block - otherwise a player hits an unclimbable face.
        let mut rng = rand::thread_rng();
        let r = ARENA_RADIUS as i32 + 1;
        for _ in 0..300 {
            let t = Terrain::random(&mut rng);
            for cx in -r..=r {
                for cz in -r..=r {
                    if ((cx * cx + cz * cz) as f32).sqrt() > ARENA_RADIUS + 1.0 {
                        continue;
                    }
                    let h = t.column_top(cx, cz);
                    for (nx, nz) in [(cx + 1, cz), (cx, cz + 1)] {
                        let d = (h - t.column_top(nx, nz)).abs();
                        assert!(d <= 1, "step of {d} at ({cx},{cz})->({nx},{nz}) for {:?}", t.kind);
                    }
                }
            }
        }
    }
}
