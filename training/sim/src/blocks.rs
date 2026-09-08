//! A sparse block grid layered over the terrain heightfield: placed blocks
//! (planks, cobweb), flowing water and lava spread from placed *sources*,
//! and the stone / cobblestone / obsidian that lava+water contact
//! generates. Only modified cells are stored (`BTreeMap` - deterministic
//! iteration order matters for reproducible runs).
//!
//! A `Cell { x, y, z }` occupies the world box `[x, x+1) x [y, y+1) x
//! [z, z+1)`. Anything at or below the terrain surface is implicitly solid
//! "ground"; the grid only tracks what's been added or changed.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::config::cfg;
use crate::kit::Item;
use crate::physics::{aabb_overlap, Aabb, Vec3, DT};
use crate::terrain::Terrain;

/// An integer block coordinate. Stored in the grid ordered by `(x, z, y)` so
/// every cell of one column is a contiguous `BTreeMap` range - the collision
/// / fluid / ray code slices columns constantly. Construction stays
/// axis-explicit (`Cell::new(x, y, z)`); only the storage order differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Cell {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Cell { x, y, z }
    }
    #[inline]
    pub fn of(p: Vec3) -> Self {
        Cell::new(p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
    }
    #[inline]
    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Cell::new(self.x + dx, self.y + dy, self.z + dz)
    }
    #[inline]
    fn key(self) -> (i32, i32, i32) {
        (self.x, self.z, self.y)
    }
    /// World-space horizontal centre of the column.
    #[inline]
    pub fn center_xz(self) -> (f32, f32) {
        (self.x as f32 + 0.5, self.z as f32 + 0.5)
    }
    #[inline]
    pub fn box_of(self) -> Aabb {
        Aabb {
            min: Vec3::new(self.x as f32, self.y as f32, self.z as f32),
            max: Vec3::new(self.x as f32 + 1.0, self.y as f32 + 1.0, self.z as f32 + 1.0),
        }
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Block {
    Planks,
    Cobweb,
    Stone,
    Cobblestone,
    Obsidian,
    /// Flowing/source water; `u8` is the flow distance (0 = source-strength).
    Water(u8),
    /// Flowing/source lava; `u8` is the flow distance.
    Lava(u8),
}

impl Block {
    pub fn is_solid(self) -> bool {
        matches!(self, Block::Planks | Block::Stone | Block::Cobblestone | Block::Obsidian)
    }
    pub fn is_fluid(self) -> bool {
        matches!(self, Block::Water(_) | Block::Lava(_))
    }
    /// Can a tool break this? Every *placed* block except a fluid - the
    /// terrain floor and the arena wall aren't `Block`s at all, so they're
    /// implicitly excluded.
    pub fn is_mineable(self) -> bool {
        self.is_solid() || matches!(self, Block::Cobweb)
    }
    /// Vanilla-style block hardness - break time scales with it (see
    /// `CombatConfig::mine_seconds_per_hardness`). 0 = not mineable.
    pub fn mine_hardness(self) -> f32 {
        match self {
            Block::Planks => 2.0,
            Block::Stone => 1.5,
            Block::Cobblestone => 2.0,
            Block::Obsidian => 50.0,
            Block::Cobweb => 0.8,
            _ => 0.0,
        }
    }
    /// The tool that mines this block *fast* (its "correct tool"). Anything
    /// else mines it at wrong-tool speed (1.0, no Efficiency). `None` = no
    /// tool here is fast (cobweb - low hardness, so it's quick anyway).
    pub fn mine_tool(self) -> Option<Item> {
        match self {
            Block::Planks => Some(Item::Axe),
            Block::Stone | Block::Cobblestone | Block::Obsidian => Some(Item::Pickaxe),
            _ => None,
        }
    }
    #[allow(dead_code)] // used by tests
    pub fn is_water(self) -> bool {
        matches!(self, Block::Water(_))
    }
    /// Observation id: planks 1, cobweb 2, stone 3, cobblestone 4,
    /// obsidian 5, water 6, lava 7. For a live bridge / debug dump.
    #[allow(dead_code)]
    pub fn wire_id(self) -> f32 {
        match self {
            Block::Planks => 1.0,
            Block::Cobweb => 2.0,
            Block::Stone => 3.0,
            Block::Cobblestone => 4.0,
            Block::Obsidian => 5.0,
            Block::Water(_) => 6.0,
            Block::Lava(_) => 7.0,
        }
    }
}

#[derive(Clone, Copy)]
struct TimedBlock {
    block: Block,
    /// Ticks left before decay; `u32::MAX` = permanent for the match.
    ttl: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FluidKind {
    Water,
    Lava,
}

#[derive(Clone)]
struct FluidSource {
    kind: FluidKind,
    cell: Cell,
    /// Game ticks the source keeps feeding at full strength.
    ttl: u32,
    /// Fluid ticks since the feed stopped. `None` while still fed. Vanilla
    /// drains a flow **source-first**: a cell at flow-level `L` loses its
    /// feed `L` fluid ticks after the source dies, so the puddle empties
    /// from the middle outward. The re-flood skips levels `<= dead_ticks`.
    dead_ticks: Option<u32>,
}

#[derive(Clone, Default)]
pub struct BlockWorld {
    blocks: BTreeMap<Cell, TimedBlock>,
    sources: Vec<FluidSource>,
    tick: u32,
    /// Reused across `recompute_fluids` so a per-tick reflood allocates
    /// nothing after warm-up. `flood` / `flood_queue` run one source's BFS;
    /// `level` merges every source's result in deterministic (sorted) order
    /// before it's written back into `blocks`.
    flood: HashMap<Cell, u8>,
    flood_queue: VecDeque<(Cell, u8)>,
    level: BTreeMap<Cell, (FluidKind, u8)>,
}


impl BlockWorld {
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.sources.clear();
        self.tick = 0;
    }

    #[allow(dead_code)] // used by tests
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    fn get(&self, c: Cell) -> Option<Block> {
        self.blocks.get(&c).map(|t| t.block)
    }

    /// Is this unit cell solid? True for the implicit terrain ground (every
    /// cell below the column's top) and for a placed solid block. Fluids and
    /// cobweb are not solid. This is the single predicate both collision and
    /// line-of-sight consult - terrain is never materialized into the map.
    #[inline]
    pub fn is_solid_cell(&self, c: Cell, terrain: &Terrain) -> bool {
        c.y < terrain.column_top(c.x, c.z)
            || self.get(c).is_some_and(Block::is_solid)
            || self.is_wall_cell(c, terrain)
    }

    /// A cell inside the UHC perimeter wall: the ring of columns at the
    /// platform rim, `arena_wall_height` blocks tall, resting on the terrain
    /// surface. Implicit static geometry like the terrain ground - it's never
    /// materialized into the block map, so it costs no memory and survives
    /// `clear()`. Collision, line-of-sight and placement all consult this via
    /// `is_solid_cell`.
    #[inline]
    pub fn is_wall_cell(&self, c: Cell, terrain: &Terrain) -> bool {
        let k = cfg();
        if !k.arena_walls || k.arena_wall_height == 0 {
            return false;
        }
        wall_cell(c, terrain.column_top(c.x, c.z), k.arena_radius, k.arena_wall_height)
    }

    /// Above this column's build ceiling, or outside the platform - nothing
    /// (fluid or placed block) may exist here.
    #[inline]
    fn out_of_bounds(&self, c: Cell, terrain: &Terrain) -> bool {
        if c.y > terrain.column_top(c.x, c.z) + cfg().combat.block_ceiling as i32 {
            return true;
        }
        let (cx, cz) = c.center_xz();
        (cx * cx + cz * cz).sqrt() > cfg().arena_radius + 1.0
    }

    /// Is this cell blocked to a flowing fluid?
    fn blocks_fluid(&self, c: Cell, terrain: &Terrain) -> bool {
        self.is_solid_cell(c, terrain) || self.out_of_bounds(c, terrain)
    }

    fn place(&mut self, c: Cell, block: Block, ttl: u32) {
        if self.blocks.len() >= cfg().combat.max_blocks && !self.blocks.contains_key(&c) {
            return;
        }
        self.blocks.insert(c, TimedBlock { block, ttl });
    }

    /// A cell a new solid block / source may occupy: empty of a solid block
    /// and terrain, in bounds. An existing *fluid* cell is allowed - the
    /// placed block replaces it.
    fn placeable_cell(&self, c: Cell, terrain: &Terrain) -> bool {
        !self.is_solid_cell(c, terrain) && !self.out_of_bounds(c, terrain)
    }

    /// Voxel-march an eye ray and return `(hit_cell, face_normal)` for the
    /// first solid cell the ray enters within `reach` - the face normal
    /// points back toward the ray, so `hit_cell + face_normal` is the empty
    /// cell a block would be placed against. For a bucket, returns the first
    /// fluid cell (normal zero) so it pours into what you aim at. `None` if
    /// the ray hits nothing placeable (e.g. looking at the sky) or leaves
    /// the platform first.
    pub fn raycast_place_target(
        &self,
        eye: Vec3,
        dir: Vec3,
        reach: f32,
        terrain: &Terrain,
        is_bucket: bool,
    ) -> Option<(Cell, Cell)> {
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if len < 1e-6 {
            return None;
        }
        let d = Vec3::new(dir.x / len, dir.y / len, dir.z / len);
        let mut hit: Option<(Cell, usize, i32)> = None;
        march_voxels(eye, d, reach, |cell, axis, step, p| {
            if (p.x * p.x + p.z * p.z).sqrt() > cfg().arena_radius + 1.0 {
                return MarchStep::Stop; // left the platform
            }
            if is_bucket && self.get(cell).is_some_and(Block::is_fluid) {
                hit = Some((cell, 3, 0)); // axis 3 = "no face" sentinel
                return MarchStep::Stop;
            }
            if self.is_solid_cell(cell, terrain) {
                hit = Some((cell, axis, step));
                return MarchStep::Stop;
            }
            MarchStep::Continue
        });
        let (cell, axis, step) = hit?;
        let normal = match axis {
            0 => Cell::new(-step, 0, 0),
            1 => Cell::new(0, -step, 0),
            2 => Cell::new(0, 0, -step),
            _ => Cell::new(0, 0, 0), // bucket into a fluid cell
        };
        if axis == 3 {
            return Some((cell, normal));
        }
        let target = cell.offset(normal.x, normal.y, normal.z);
        self.placeable_cell(target, terrain).then_some((target, normal))
    }

    /// Voxel-march an eye ray and return the first **placed** block it hits
    /// within `reach`, with what's in that cell. The terrain floor and the
    /// arena perimeter wall stop the ray and are never returned - a pickaxe
    /// can't mine them. Fluids don't stop the ray (you mine the solid behind
    /// them). `None` if the ray hits only air, or leaves the platform first.
    pub fn raycast_mine_target(
        &self,
        eye: Vec3,
        dir: Vec3,
        reach: f32,
        terrain: &Terrain,
    ) -> Option<(Cell, Block)> {
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        if len < 1e-6 {
            return None;
        }
        let d = Vec3::new(dir.x / len, dir.y / len, dir.z / len);
        let mut hit = None;
        march_voxels(eye, d, reach, |cell, _axis, _step, p| {
            if (p.x * p.x + p.z * p.z).sqrt() > cfg().arena_radius + 1.0 {
                return MarchStep::Stop; // left the platform
            }
            if cell.y < terrain.column_top(cell.x, cell.z) || self.is_wall_cell(cell, terrain) {
                return MarchStep::Stop; // floor / wall: not mineable, blocks the ray
            }
            match self.get(cell) {
                Some(b) if b.is_mineable() => {
                    hit = Some((cell, b));
                    MarchStep::Stop
                }
                _ => MarchStep::Continue,
            }
        });
        hit
    }

    /// Remove a placed block (a completed mine). `true` if a block was there.
    pub fn mine_block(&mut self, cell: Cell) -> bool {
        self.blocks.remove(&cell).is_some()
    }

    /// Place one block of `kind` at `target` if the cell is free (vanilla is
    /// one block per click - the policy builds a wall block by block).
    pub fn place_block(&mut self, target: Cell, kind: Block, terrain: &Terrain) {
        let c = &cfg().combat;
        let ttl = match kind {
            Block::Planks => (c.planks_block_seconds / DT) as u32,
            Block::Cobweb => (c.cobweb_block_seconds / DT) as u32,
            _ => u32::MAX,
        };
        if self.placeable_cell(target, terrain) {
            self.place(target, kind, ttl);
        }
    }

    pub fn place_source(&mut self, kind: FluidKind, target: Cell, terrain: &Terrain) {
        if !self.placeable_cell(target, terrain) {
            return;
        }
        let secs = match kind {
            FluidKind::Water => cfg().combat.water_source_seconds,
            FluidKind::Lava => cfg().combat.lava_source_seconds,
        };
        self.sources.push(FluidSource {
            kind,
            cell: target,
            ttl: (secs / DT) as u32,
            dead_ticks: None,
        });
    }

    fn max_level(kind: FluidKind) -> u32 {
        match kind {
            FluidKind::Water => cfg().combat.water_max_level,
            FluidKind::Lava => cfg().combat.lava_max_level,
        }
    }
    fn level_step(kind: FluidKind) -> u32 {
        match kind {
            FluidKind::Water => 1,
            FluidKind::Lava => cfg().combat.lava_level_step.max(1),
        }
    }

    /// One simulation tick: decay timed blocks, count down source feed, and
    /// on each fluid's own schedule (water every `water_tick_ticks`, lava
    /// every `lava_tick_ticks`) re-flood the flow and resolve lava/water
    /// contacts.
    pub fn tick(&mut self, terrain: &Terrain) {
        self.tick = self.tick.wrapping_add(1);
        self.blocks.retain(|_, tb| {
            if tb.ttl == u32::MAX {
                return true;
            }
            if tb.ttl == 0 {
                return false;
            }
            tb.ttl -= 1;
            true
        });
        for s in &mut self.sources {
            if s.ttl > 0 {
                s.ttl -= 1;
            }
        }

        let do_water = self.tick.is_multiple_of(cfg().combat.water_tick_ticks.max(1));
        let do_lava = self.tick.is_multiple_of(cfg().combat.lava_tick_ticks.max(1));
        if !do_water && !do_lava {
            return;
        }
        if self.sources.is_empty() {
            // Nothing feeds a flow any more: clear whatever the last dead
            // source left behind (once) and skip the reflood entirely - the
            // common case for the sword / axe kits and for early UHC.
            if self.blocks.values().any(|tb| tb.block.is_fluid()) {
                self.blocks.retain(|_, tb| !tb.block.is_fluid());
            }
            return;
        }
        // Advance the drain counter for each un-fed source whose fluid ticks
        // this pass, and drop a source once its whole flow has drained.
        for s in &mut self.sources {
            let ticks_now = match s.kind {
                FluidKind::Water => do_water,
                FluidKind::Lava => do_lava,
            };
            if ticks_now && s.ttl == 0 {
                s.dead_ticks = Some(s.dead_ticks.map_or(1, |k| k + 1));
            }
        }
        self.sources.retain(|s| match s.dead_ticks {
            // Drop once the drain front `(k + 1) * step` has passed `max`.
            Some(k) => (k + 1) * Self::level_step(s.kind) <= Self::max_level(s.kind),
            None => true,
        });

        self.recompute_fluids(terrain, do_water, do_lava);
        self.resolve_contacts();
    }

    /// Re-flood the flow for whichever kinds tick this pass (`do_water` /
    /// `do_lava`). A kind that isn't recomputing keeps its existing cells,
    /// which are seeded into `level` so the recomputing kind still flows
    /// around them - so a lava-only pass need not also rebuild all the water.
    /// Scratch buffers (`flood` / `flood_queue` / `level`) are reused across
    /// calls, so a steady-state reflood allocates nothing.
    fn recompute_fluids(&mut self, terrain: &Terrain, do_water: bool, do_lava: bool) {
        self.blocks.retain(|_, tb| match tb.block {
            Block::Water(_) => !do_water,
            Block::Lava(_) => !do_lava,
            _ => true,
        });

        let mut level = std::mem::take(&mut self.level);
        let mut flood = std::mem::take(&mut self.flood);
        let mut queue = std::mem::take(&mut self.flood_queue);
        level.clear();

        // Fluid of a kind that isn't recomputing survived the retain above -
        // seed it so the recomputing kind's flood stops against it. Then seed
        // every live source cell at level 0 so a source always owns its own
        // cell. A *dead* source's inner rings have already drained.
        for (&c, tb) in &self.blocks {
            match tb.block {
                Block::Water(l) => {
                    level.insert(c, (FluidKind::Water, l));
                }
                Block::Lava(l) => {
                    level.insert(c, (FluidKind::Lava, l));
                }
                _ => {}
            }
        }
        for s in &self.sources {
            if s.dead_ticks.is_none() {
                level.insert(s.cell, (s.kind, 0));
            }
        }

        for s in &self.sources {
            let recompute = match s.kind {
                FluidKind::Water => do_water,
                FluidKind::Lava => do_lava,
            };
            if !recompute {
                continue;
            }
            let max_range = Self::max_level(s.kind) as u8;
            let step = Self::level_step(s.kind) as u8;
            // A dead source's drain front is `(k + 1) * step` levels out.
            let min_level = s.dead_ticks.map_or(0, |k| (k + 1) * Self::level_step(s.kind)) as u8;

            flood.clear();
            queue.clear();
            queue.push_back((s.cell, 0));
            while let Some((c, lvl)) = queue.pop_front() {
                if self.blocks_fluid(c, terrain) {
                    continue;
                }
                if flood.get(&c).is_some_and(|&l| l <= lvl) {
                    continue;
                }
                flood.insert(c, lvl);
                let below = c.offset(0, -1, 0);
                if !self.blocks_fluid(below, terrain) {
                    queue.push_back((below, 0)); // falling resets the horizontal budget
                }
                if lvl < max_range {
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        queue.push_back((c.offset(dx, 0, dz), lvl + step));
                    }
                }
            }
            // Merge order-independently into `level` (the final write below is
            // driven by `level`'s sorted iteration, so `flood`'s hash order
            // never reaches observable state).
            for (&c, &lvl) in &flood {
                if lvl < min_level {
                    continue; // drained inner ring of a dead source
                }
                match level.get(&c) {
                    Some(&(k, l)) if k == s.kind && l <= lvl => {}
                    Some(&(k, _)) if k != s.kind => {}
                    _ => {
                        level.insert(c, (s.kind, lvl));
                    }
                }
            }
        }
        for (&c, &(kind, lvl)) in &level {
            let b = match kind {
                FluidKind::Water => Block::Water(lvl),
                FluidKind::Lava => Block::Lava(lvl),
            };
            self.place(c, b, u32::MAX);
        }

        flood.clear();
        queue.clear();
        self.level = level;
        self.flood = flood;
        self.flood_queue = queue;
    }

    /// Lava touching water: source lava -> obsidian, flowing lava with water
    /// above/below -> stone, otherwise -> cobblestone. Consumed flowing
    /// water is removed; a converted source stops feeding.
    fn resolve_contacts(&mut self) {
        let dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
        let mut set_solid: Vec<(Cell, Block)> = Vec::new();
        let mut drop_water: Vec<Cell> = Vec::new();
        let mut kill_sources: Vec<Cell> = Vec::new();
        for (&c, tb) in &self.blocks {
            let Block::Lava(lvl) = tb.block else { continue };
            let mut water_dir = None;
            for d in dirs {
                let n = c.offset(d.0, d.1, d.2);
                if matches!(self.get(n), Some(Block::Water(_))) {
                    water_dir = Some(d);
                    break;
                }
            }
            let Some(d) = water_dir else { continue };
            let new = if lvl == 0 {
                Block::Obsidian
            } else if d.1 != 0 {
                Block::Stone
            } else {
                Block::Cobblestone
            };
            set_solid.push((c, new));
            if lvl == 0 {
                kill_sources.push(c);
            }
            let n = c.offset(d.0, d.1, d.2);
            if matches!(self.get(n), Some(Block::Water(w)) if w > 0) {
                drop_water.push(n);
            }
        }
        for (c, b) in set_solid {
            self.place(c, b, u32::MAX);
        }
        for c in drop_water {
            self.blocks.remove(&c);
        }
        self.sources.retain(|s| !kill_sources.contains(&s.cell));
    }

    /// The y a player standing in column `(cx, cz)` rests on: the top of the
    /// highest solid cell at or below `ceil_cell`, else the terrain top.
    /// (Cells above `ceil_cell` are ignored so a player can't snap up the
    /// face of a tall wall - that needs a jump / step-up.)
    pub fn support_y(&self, cx: i32, cz: i32, terrain: &Terrain, ceil_cell: i32) -> f32 {
        let base = terrain.column_top(cx, cz);
        let mut best = base;
        // The UHC perimeter wall reads as a tall obstacle here so it shows up
        // in the block-grid observation (and nobody snaps onto its top).
        if self.is_wall_cell(Cell::new(cx, base, cz), terrain) {
            best = best.max((base + cfg().arena_wall_height as i32).min(ceil_cell));
        }
        for (c, tb) in self
            .blocks
            .range(Cell::new(cx, i32::MIN, cz)..=Cell::new(cx, i32::MAX, cz))
        {
            if tb.block.is_solid() && c.y >= base && c.y < ceil_cell && c.y + 1 > best {
                best = c.y + 1;
            }
        }
        best as f32
    }

    /// The horizontal flow direction of the fluid at `cell` - toward its
    /// lower-level (further-from-source) or open-air neighbours, away from
    /// higher-level ones. Zero for a source cell or still water (no gradient).
    /// Vanilla `FlowingFluid::getFlow`, horizontal component.
    pub fn flow_vector(&self, cell: Cell, terrain: &Terrain) -> Vec3 {
        let Some(here) = self.fluid_level(cell) else { return Vec3::ZERO };
        let mut flow = Vec3::ZERO;
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let n = cell.offset(dx, 0, dz);
            let weight = match self.fluid_level(n) {
                Some(l) => l as f32 - here as f32, // toward lower level (higher number)
                None if !self.blocks_fluid(n, terrain) => 1.0, // spill into open air
                None => 0.0,                       // wall: no push
            };
            flow.x += dx as f32 * weight;
            flow.z += dz as f32 * weight;
        }
        flow
    }

    fn fluid_level(&self, c: Cell) -> Option<u8> {
        match self.get(c) {
            Some(Block::Water(l)) | Some(Block::Lava(l)) => Some(l),
            _ => None,
        }
    }

    /// Does a solid cell (terrain or a placed block) sit between `origin` and
    /// a hit at parametric distance `max_t` along `dir`? Cobweb and fluids
    /// don't block. Voxel-marched, so the cost is the ray's length in cells,
    /// not the size of the block map.
    pub fn ray_blocked(&self, origin: Vec3, dir: Vec3, max_t: f32, terrain: &Terrain) -> bool {
        let mut blocked = false;
        march_voxels(origin, dir, max_t, |cell, _axis, _step, _p| {
            if self.is_solid_cell(cell, terrain) {
                blocked = true;
                MarchStep::Stop
            } else {
                MarchStep::Continue
            }
        });
        blocked
    }

    /// Which fluids / cobweb does a player's hitbox overlap - even by a
    /// sliver (any AABB overlap counts, not just "feet inside").
    pub fn player_contact(&self, feet: Vec3) -> Contact {
        let pbox = Aabb::player_at(feet);
        let lo = Cell::of(pbox.min);
        let hi = Cell::of(pbox.max);
        let mut ct = Contact::default();
        for x in lo.x..=hi.x {
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    let c = Cell::new(x, y, z);
                    let Some(b) = self.get(c) else { continue };
                    if !aabb_overlap(&pbox, &c.box_of()) {
                        continue;
                    }
                    match b {
                        Block::Water(_) => ct.water = true,
                        Block::Lava(_) => ct.lava = true,
                        Block::Cobweb => ct.cobweb = true,
                        _ => {}
                    }
                }
            }
        }
        ct
    }

    /// Accumulated horizontal flow-push direction over every fluid cell the
    /// player's hitbox overlaps (see `flow_vector`). Fed into
    /// `apply_block_effects` where it's normalized and scaled.
    pub fn flow_push(&self, feet: Vec3, terrain: &Terrain) -> Vec3 {
        let pbox = Aabb::player_at(feet);
        let lo = Cell::of(pbox.min);
        let hi = Cell::of(pbox.max);
        let mut push = Vec3::ZERO;
        for x in lo.x..=hi.x {
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    let c = Cell::new(x, y, z);
                    if self.fluid_level(c).is_some() && aabb_overlap(&pbox, &c.box_of()) {
                        let f = self.flow_vector(c, terrain);
                        push.x += f.x;
                        push.y += f.y;
                        push.z += f.z;
                    }
                }
            }
        }
        push
    }

    /// The yaw-rotated `size x size` column view around a player: per column
    /// `[surface_rel_height, water, lava, cobweb]`, row-major, front-left
    /// first. `yaw` is the observer's yaw (so +row = forward).
    pub fn column_view(
        &self,
        pos: Vec3,
        yaw: f32,
        terrain: &Terrain,
        size: usize,
    ) -> Vec<[f32; 4]> {
        let (sin_y, cos_y) = yaw.sin_cos();
        let half = size as i32 / 2;
        let ceiling = cfg().combat.block_ceiling as i32;
        let mut out = Vec::with_capacity(size * size);
        for row in 0..size as i32 {
            for col in 0..size as i32 {
                // local: col across (= player's right), row = forward.
                let ox = (col - half) as f32;
                let oz = (row - half) as f32;
                // world offset = ox * right(yaw) + oz * forward(yaw),
                // right = (cos, sin), forward = (-sin, cos).
                let wx = pos.x + ox * cos_y - oz * sin_y;
                let wz = pos.z + ox * sin_y + oz * cos_y;
                let (cx, cz) = (wx.floor() as i32, wz.floor() as i32);
                let ceil_cell = pos.y.floor() as i32 + ceiling as i32;
                let surface = self.support_y(cx, cz, terrain, ceil_cell);
                let mut water = 0.0;
                let mut lava = 0.0;
                let mut cobweb = 0.0;
                let ylo = (pos.y.floor() as i32) - 1;
                let yhi = (pos.y.floor() as i32) + 3;
                for cy in ylo..=yhi {
                    match self.get(Cell::new(cx, cy, cz)) {
                        Some(Block::Water(_)) => water = 1.0,
                        Some(Block::Lava(_)) => lava = 1.0,
                        Some(Block::Cobweb) => cobweb = 1.0,
                        _ => {}
                    }
                }
                out.push([surface - pos.y, water, lava, cobweb]);
            }
        }
        out
    }

    /// Drop a permanent block straight into a cell - test-only shortcut past
    /// the raycast/placement rules.
    #[cfg(test)]
    pub fn place_block_for_test(&mut self, cell: Cell, block: Block) {
        self.place(cell, block, u32::MAX);
    }

    /// Nearby blocks for a live-bridge / debugging dump (not used by the
    /// training observation).
    #[allow(dead_code)]
    pub fn iter_blocks(&self) -> impl Iterator<Item = (Cell, Block)> + '_ {
        self.blocks.iter().map(|(&c, tb)| (c, tb.block))
    }
}

/// Pure geometry of the UHC perimeter wall (see `BlockWorld::is_wall_cell`):
/// `column_base` is the terrain top at `c`'s column, the wall is a ring one
/// column wide at the platform rim, `height` blocks tall on that surface.
#[inline]
pub(crate) fn wall_cell(c: Cell, column_base: i32, arena_radius: f32, height: u32) -> bool {
    let (cx, cz) = c.center_xz();
    let r = (cx * cx + cz * cz).sqrt();
    if r < arena_radius - 0.5 || r > arena_radius + 1.0 {
        return false;
    }
    c.y >= column_base && c.y < column_base + height as i32
}

#[derive(Default, Clone, Copy)]
pub struct Contact {
    pub water: bool,
    pub lava: bool,
    pub cobweb: bool,
}

pub(crate) enum MarchStep {
    Continue,
    Stop,
}

/// Amanatides-Woo voxel traversal. Calls `f(cell, axis, step, entry_point)`
/// for every unit cell the ray `origin + dir * t` enters, in order, for
/// `t` in `(0, max_t]` - the origin cell is skipped. `axis` is 0/1/2 (x/y/z)
/// and `step` is +1/-1: the entered cell's face normal is `-step` on `axis`.
/// `dir` need not be unit; `max_t` is in the same units as `dir` (so a
/// caller can pass a whole segment with `max_t = 1.0`).
pub(crate) fn march_voxels(
    origin: Vec3,
    dir: Vec3,
    max_t: f32,
    mut f: impl FnMut(Cell, usize, i32, Vec3) -> MarchStep,
) {
    let mut c = Cell::of(origin);
    let o = [origin.x, origin.y, origin.z];
    let d = [dir.x, dir.y, dir.z];
    let cc = [c.x, c.y, c.z];
    let mut step = [0i32; 3];
    let mut t_max = [f32::INFINITY; 3];
    let mut t_delta = [f32::INFINITY; 3];
    for a in 0..3 {
        if d[a] > 0.0 {
            step[a] = 1;
            t_max[a] = (cc[a] as f32 + 1.0 - o[a]) / d[a];
            t_delta[a] = 1.0 / d[a];
        } else if d[a] < 0.0 {
            step[a] = -1;
            t_max[a] = (cc[a] as f32 - o[a]) / d[a];
            t_delta[a] = -1.0 / d[a];
        }
    }
    for _ in 0..8192 {
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        let t = t_max[axis];
        if t > max_t || t.is_nan() {
            return;
        }
        c = c.offset(
            if axis == 0 { step[0] } else { 0 },
            if axis == 1 { step[1] } else { 0 },
            if axis == 2 { step[2] } else { 0 },
        );
        let p = Vec3::new(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);
        if let MarchStep::Stop = f(c, axis, step[axis], p) {
            return;
        }
        t_max[axis] += t_delta[axis];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Kit;
    use rand::SeedableRng;

    fn flat() -> Terrain {
        Terrain::flat()
    }

    fn source(kind: FluidKind, cell: Cell) -> FluidSource {
        FluidSource { kind, cell, ttl: 10_000, dead_ticks: None }
    }

    #[test]
    fn water_flows_downhill_and_outward_from_a_source() {
        let mut w = BlockWorld::default();
        let t = flat();
        w.place_source(FluidKind::Water, Cell::new(0, 3, 0), &t);
        for _ in 0..40 {
            w.tick(&t);
        }
        let n = w.blocks.values().filter(|b| b.block.is_water()).count();
        assert!(n > 3, "water spread to {n} cells");
    }

    #[test]
    fn a_drained_source_drains_from_the_middle_outward() {
        let mut w = BlockWorld::default();
        let t = flat();
        w.place_source(FluidKind::Water, Cell::new(0, 0, 0), &t);
        for _ in 0..60 {
            w.tick(&t);
        }
        let spread = w.blocks.values().filter(|b| b.block.is_water()).count();
        assert!(spread > 4, "water spread to {spread} cells before draining");
        // The cell right next to the source (flow level 1) is present now.
        assert!(matches!(w.get(Cell::new(1, 0, 0)), Some(Block::Water(_))));
        w.sources[0].ttl = 0;
        // One water fluid-tick (5 game ticks) later the level-1 ring is gone
        // but the puddle as a whole is not.
        for _ in 0..6 {
            w.tick(&t);
        }
        let mid = w.blocks.values().filter(|b| b.block.is_water()).count();
        assert!(w.get(Cell::new(1, 0, 0)).is_none(), "the source-adjacent ring drained first");
        assert!(mid > 0 && mid < spread, "still partly full: {mid} of {spread}");
        for _ in 0..60 {
            w.tick(&t);
        }
        assert_eq!(w.blocks.values().filter(|b| b.block.is_water()).count(), 0);
    }

    #[test]
    fn lava_meeting_water_makes_stone_family_blocks() {
        let mut w = BlockWorld::default();
        let t = flat();
        w.sources.push(source(FluidKind::Lava, Cell::new(0, 0, 0)));
        w.sources.push(source(FluidKind::Water, Cell::new(2, 0, 0)));
        for _ in 0..60 {
            w.tick(&t);
        }
        let solids: Vec<Block> = w
            .blocks
            .values()
            .map(|b| b.block)
            .filter(|b| matches!(b, Block::Stone | Block::Cobblestone | Block::Obsidian))
            .collect();
        assert!(!solids.is_empty(), "lava+water generated {solids:?}");
    }

    #[test]
    fn a_player_touching_a_lava_cell_by_a_sliver_still_registers() {
        let mut w = BlockWorld::default();
        w.place(Cell::of(Vec3::new(1.0, 0.0, 0.0)), Block::Lava(0), u32::MAX);
        let ct = w.player_contact(Vec3::new(0.72, 0.0, 0.0));
        assert!(ct.lava, "a sliver of overlap with the lava cell counts");
    }

    #[test]
    fn the_perimeter_wall_is_a_rim_ring_three_blocks_tall() {
        // radius 12: the rim ring of columns is solid from the surface up,
        // the interior and the airspace above the wall are not.
        assert!(wall_cell(Cell::new(12, 0, 0), 0, 12.0, 3));
        assert!(wall_cell(Cell::new(12, 2, 0), 0, 12.0, 3));
        assert!(!wall_cell(Cell::new(12, 3, 0), 0, 12.0, 3), "above the wall top");
        assert!(!wall_cell(Cell::new(0, 0, 0), 0, 12.0, 3), "arena centre");
        assert!(!wall_cell(Cell::new(20, 0, 0), 0, 12.0, 3), "well outside");
        // it rides the terrain surface, not y=0
        assert!(wall_cell(Cell::new(0, 5, 12), 5, 12.0, 3));
        assert!(!wall_cell(Cell::new(0, 4, 12), 5, 12.0, 3), "below the raised surface");
    }

    #[test]
    fn placed_planks_block_a_ray() {
        let _ = Kit::Sword;
        let mut w = BlockWorld::default();
        let t = flat();
        w.place_block(Cell::new(0, 0, 0), Block::Planks, &t);
        assert!(w.ray_blocked(Vec3::new(0.0, 0.5, -1.0), Vec3::new(0.0, 0.0, 1.0), 10.0, &t));
    }

    #[test]
    fn a_ray_is_blocked_by_a_hill() {
        // Slope terrain: a level ray fired into the rising side hits ground.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut t = Terrain::flat();
        while t.column_top(6, 0) <= t.column_top(-6, 0) {
            t = Terrain::random(&mut rng);
        }
        let low = t.column_top(-6, 0) as f32;
        assert!(BlockWorld::default().ray_blocked(
            Vec3::new(-6.0, low + 0.5, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            12.0,
            &t,
        ));
    }

    #[test]
    fn raycast_place_target_hits_the_ground_ahead_and_misses_the_sky() {
        let w = BlockWorld::default();
        let t = flat(); // surface y = 0
        let eye = Vec3::new(0.0, 1.62, 0.0);
        let down = crate::physics::look_direction(0.0, 0.6);
        let hit = w.raycast_place_target(eye, down, 4.5, &t, false);
        assert!(hit.is_some_and(|(c, _)| c.y == 0), "found a ground cell ahead: {hit:?}");
        let up = crate::physics::look_direction(0.0, -0.6);
        assert_eq!(w.raycast_place_target(eye, up, 4.5, &t, false), None);
    }

    #[test]
    fn one_click_places_one_block_into_a_fluid_cell() {
        let mut w = BlockWorld::default();
        let t = flat();
        w.place(Cell::new(0, 0, 2), Block::Water(1), u32::MAX);
        w.place_block(Cell::new(0, 0, 2), Block::Cobweb, &t);
        assert!(matches!(w.get(Cell::new(0, 0, 2)), Some(Block::Cobweb)));
        assert_eq!(w.blocks.values().filter(|b| b.block == Block::Cobweb).count(), 1);
    }
}
