//! One self-contained match: two teams of `team_size` players, the live
//! projectile list and the block grid. `step` runs the whole tick - per
//! player input + physics (`player`), wall / player collisions, melee
//! (`combat`), projectiles (`projectile`), the block grid, match-end - then
//! builds each slot's observation (`observation`) and re-randomizes the
//! spawns in place for the next match.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::blocks::{Block, BlockWorld, FluidKind};
use crate::collision;
use crate::combat;
use crate::config::cfg;
use crate::kit::{loadout, Item};
use crate::observation;
use crate::physics::{look_direction, pair_mut, Aabb, Vec3, DT, EYE_HEIGHT, PLAYER_HEIGHT};
use crate::player::{self, Player};
use crate::projectile::{self, Projectile};
use crate::protocol::{Action, Observation};
use crate::terrain::Terrain;

pub struct Arena {
    pub players: Vec<Player>,
    pub projectiles: Vec<Projectile>,
    pub world: BlockWorld,
    pub time_left: f32,
    pub terrain: Terrain,
    team_size: usize,
    rng: StdRng,
    /// Reused collision-candidate buffer for `player::integrate` - cleared
    /// and refilled each call, never reallocated after warm-up.
    collision_scratch: Vec<Aabb>,
}

/// Per-agent-slot outcome of one `step`: the raw damage / event flags that
/// feed both the scalar `reward` and the tail of the `Observation`.
#[derive(Default, Clone, Copy)]
pub(crate) struct StepEvents {
    pub damage_dealt: f32,
    pub damage_taken: f32,
    pub friendly_damage: f32,
    pub swept: bool,
    pub won: bool,
    pub lost: bool,
    pub done: bool,
}

impl StepEvents {
    pub(crate) fn reward(&self) -> f32 {
        let r = cfg().reward;
        let mut reward = self.damage_dealt * r.per_hp_dealt - self.damage_taken * r.per_hp_taken;
        reward -= self.friendly_damage * r.friendly_fire_penalty;
        if self.swept {
            reward -= r.sweep_penalty;
        }
        if self.won {
            reward += r.win;
        }
        if self.lost {
            reward -= r.loss;
        }
        reward
    }
}

impl Arena {
    pub fn new(seed: u64) -> Self {
        let team_size = cfg().team_size.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let terrain = new_terrain(&mut rng);
        let players = (0..2 * team_size)
            .map(|i| Player::new(if i < team_size { 0 } else { 1 }))
            .collect();
        let mut arena = Arena {
            players,
            projectiles: Vec::new(),
            world: BlockWorld::default(),
            time_left: cfg().match_time_seconds,
            terrain,
            team_size,
            rng,
            collision_scratch: Vec::new(),
        };
        arena.randomize_spawns();
        arena
    }

    fn team_range(&self, team: u8) -> std::ops::Range<usize> {
        if team == 0 {
            0..self.team_size
        } else {
            self.team_size..2 * self.team_size
        }
    }

    fn team_alive(&self, team: u8) -> bool {
        self.team_range(team).any(|i| self.players[i].alive())
    }

    /// Fresh terrain, cleared blocks/projectiles, and every player dropped
    /// back on the platform in two facing lines with a full kit - the
    /// "instant reset" the training loop needs on match end.
    fn randomize_spawns(&mut self) {
        self.terrain = new_terrain(&mut self.rng);
        self.projectiles.clear();
        self.world.clear();

        let angle: f32 = self.rng.gen_range(0.0..std::f32::consts::TAU);
        let dist = cfg().arena_radius * 0.6;
        let u = Vec3::new(angle.cos(), 0.0, angle.sin());
        let tangent = Vec3::new(-angle.sin(), 0.0, angle.cos());
        let spacing = 1.5_f32;
        let k = self.team_size as f32;
        let max_r = cfg().arena_radius * 0.9;
        let l = loadout(cfg().kit);

        for team in 0u8..2 {
            let sign = if team == 0 { 1.0 } else { -1.0 };
            let centroid = Vec3::new(sign * dist * u.x, 0.0, sign * dist * u.z);
            let enemy_centroid = Vec3::new(-centroid.x, 0.0, -centroid.z);
            for (m, idx) in self.team_range(team).enumerate() {
                let off = (m as f32 - (k - 1.0) / 2.0) * spacing;
                let mut x = centroid.x + tangent.x * off;
                let mut z = centroid.z + tangent.z * off;
                let r = (x * x + z * z).sqrt();
                if r > max_r {
                    x *= max_r / r;
                    z *= max_r / r;
                }
                let pos = Vec3::new(x, self.terrain.surface_y(x, z), z);
                let yaw = yaw_towards(Vec3::new(x, 0.0, z), enemy_centroid);
                self.players[idx].respawn(pos, yaw, &l);
                // A real connection's latency is roughly stable over a match,
                // not white noise - roll the baseline once here.
                let ping = self.rng.gen_range(cfg().min_ping_ms..=cfg().max_ping_ms);
                self.players[idx].base_ping_ms = ping;
                self.players[idx].ping_ms = ping;
            }
        }
        self.time_left = cfg().match_time_seconds;
    }

    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, actions: &[Action]) -> Vec<Observation> {
        let n = self.players.len();
        assert_eq!(actions.len(), n, "expected {n} actions, got {}", actions.len());
        let cfg_ = cfg();

        // Vanilla netcode is client-authoritative for your own movement: your
        // input has zero delay. Only your *view* of others lags (handled in
        // `observation::build` via each player's snapshot ring). Here we just
        // jitter this tick's reported ping around the match baseline.
        for i in 0..n {
            let jitter = self.rng.gen_range(-cfg_.ping_jitter_ms..=cfg_.ping_jitter_ms);
            self.players[i].ping_ms =
                (self.players[i].base_ping_ms + jitter).clamp(cfg_.min_ping_ms, cfg_.max_ping_ms);
        }
        let eff = actions;

        let mut ev = vec![StepEvents::default(); n];

        // Tick down hurt-invulnerability.
        for p in &mut self.players {
            if p.hurt_time_left > 0.0 {
                p.hurt_time_left = (p.hurt_time_left - DT).max(0.0);
                if p.hurt_time_left == 0.0 {
                    p.last_damage = 0.0;
                }
            }
        }

        // Input + item use + physics + environmental effects for the living.
        for i in 0..n {
            if !self.players[i].alive() {
                continue;
            }
            player::apply_input(&mut self.players[i], &eff[i]);

            if let Some(shot) = self.players[i].pending_shot.take() {
                projectile::spawn(&mut self.projectiles, &mut self.players[i], i, shot, &mut self.rng);
            }
            if let Some(item) = self.players[i].pending_place.take() {
                self.place_from_item(i, item);
            }
            if let Some(item) = self.players[i].pending_throw.take() {
                projectile::spawn_splash(&mut self.projectiles, &mut self.players[i], i, item, &mut self.rng);
            }

            let pre_contact = self.world.player_contact(self.players[i].pos);
            let fall = player::integrate(
                &mut self.players[i],
                &self.terrain,
                &self.world,
                pre_contact,
                &mut self.collision_scratch,
            );
            if fall > 0.0 {
                ev[i].damage_taken += self.players[i].take_damage(fall);
            }

            let block_dmg =
                player::apply_block_effects(&mut self.players[i], &self.world, &self.terrain);
            if block_dmg > 0.0 {
                ev[i].damage_taken += block_dmg;
            }
        }

        // Player-player push-apart, then de-penetrate anyone the push shoved
        // into a wall (collision with the world itself happens inside the
        // per-player move in `integrate`).
        for i in 0..n {
            for j in (i + 1)..n {
                if self.players[i].alive() && self.players[j].alive() {
                    let (a, b) = pair_mut(&mut self.players, i, j);
                    player::resolve_player_collision(a, b);
                }
            }
        }
        for i in 0..n {
            if self.players[i].alive() {
                let p = &mut self.players[i];
                collision::push_out_of_solids(
                    &mut p.pos,
                    &mut p.vel,
                    PLAYER_HEIGHT,
                    &self.world,
                    &self.terrain,
                );
            }
        }

        // Left-click: mine a placed block if the pickaxe is aimed at one,
        // otherwise a melee swing. Not holding `attack` (or eating / mid-swap)
        // clears any mining progress.
        for i in 0..n {
            let p = &self.players[i];
            let can_act = p.alive() && eff[i].attack && !p.eating() && p.swap_lockout <= 0.0;
            if !can_act {
                self.players[i].mining_cell = None;
                self.players[i].mining_progress = 0.0;
                continue;
            }
            if self.mine_step(i) {
                continue; // breaking a block this tick - no swing
            }
            combat::resolve_melee(i, &mut self.players, &self.world, &self.terrain, &mut ev);
        }

        projectile::step_all(
            &mut self.projectiles,
            &mut self.players,
            &self.world,
            &self.terrain,
            &mut ev,
        );

        // Advance the block grid: decay placed blocks / sources, and every
        // few ticks recompute fluid flow + resolve lava/water contacts.
        self.world.tick(&self.terrain);

        self.time_left -= DT;

        self.decide_match_end(&mut ev);

        // Record this tick's public state for laggy observers to read back.
        for p in &mut self.players {
            p.push_snapshot();
        }
        for pr in &mut self.projectiles {
            pr.push_snapshot();
        }

        let obs: Vec<Observation> = (0..n).map(|i| observation::build(self, i, ev[i])).collect();

        if ev[0].done {
            self.randomize_spawns();
        }
        obs
    }

    /// A whole team down, or a timeout on total team HP + absorption: mark
    /// every slot's `done` / `won` / `lost`.
    #[allow(clippy::needless_range_loop)]
    fn decide_match_end(&self, ev: &mut [StepEvents]) {
        let (t0, t1) = (self.team_alive(0), self.team_alive(1));
        let mut winner: Option<u8> = None;
        let done = if !t0 || !t1 {
            winner = match (t0, t1) {
                (true, false) => Some(0),
                (false, true) => Some(1),
                _ => None,
            };
            true
        } else if self.time_left <= 0.0 {
            let team_hp = |team| -> f32 {
                self.team_range(team).map(|i| self.players[i].hp + self.players[i].absorption).sum()
            };
            let (hp0, hp1) = (team_hp(0), team_hp(1));
            winner = if hp0 > hp1 {
                Some(0)
            } else if hp1 > hp0 {
                Some(1)
            } else {
                None
            };
            true
        } else {
            false
        };
        if !done {
            return;
        }
        for i in 0..self.players.len() {
            ev[i].done = true;
            match winner {
                Some(w) if self.players[i].team == w => ev[i].won = true,
                Some(_) => ev[i].lost = true,
                None => {}
            }
        }
    }

    /// One tick of mining for player `i`: if they hold a pickaxe or axe and
    /// their eye ray is on a placed block within `place_reach`, advance the
    /// break progress and remove the block when it hits 1.0. Returns `true`
    /// if a block is being broken this tick (the caller then skips the melee
    /// swing). The floor and the arena wall are never valid targets -
    /// `raycast_mine_target` filters them out. The **wrong tool** for a block
    /// (a pickaxe on planks, an axe on stone) mines at speed 1.0 with no
    /// Efficiency, so it's far slower than the right one.
    fn mine_step(&mut self, i: usize) -> bool {
        let p = &self.players[i];
        if !matches!(p.held, Item::Pickaxe | Item::Axe) {
            return false;
        }
        let eye = Vec3::new(p.pos.x, p.pos.y + EYE_HEIGHT, p.pos.z);
        let look = look_direction(p.yaw, p.pitch);
        let target = self.world.raycast_mine_target(
            eye,
            look,
            cfg().combat.place_reach,
            &self.terrain,
        );
        let Some((cell, block)) = target else {
            let p = &mut self.players[i];
            p.mining_cell = None;
            p.mining_progress = 0.0;
            return false;
        };
        let c = &cfg().combat;
        let speed = if block.mine_tool() == Some(p.held) {
            c.mine_correct_tool_speed + p.efficiency as f32 * c.mine_efficiency_speed_per_level
        } else {
            1.0 // wrong tool: hand speed, Efficiency doesn't help
        };
        let break_time = (block.mine_hardness() * c.mine_seconds_per_hardness / speed).max(DT);
        let p = &mut self.players[i];
        if p.mining_cell != Some(cell) {
            p.mining_cell = Some(cell);
            p.mining_progress = 0.0;
        }
        p.mining_progress += DT / break_time;
        if p.mining_progress >= 1.0 {
            p.mining_cell = None;
            p.mining_progress = 0.0;
            self.world.mine_block(cell);
        }
        true
    }

    /// A queued placeable lands in the empty cell against the face of the
    /// first block the player's eye ray hits within `place_reach` - one
    /// block per click, vanilla. Nothing in range, or the target cell
    /// overlapping a living player, -> nothing placed and the item is *not*
    /// consumed (matches vanilla `isUnobstructed`).
    fn place_from_item(&mut self, owner: usize, item: Item) {
        let slot = item.index();
        {
            let p = &self.players[owner];
            if p.counts[slot] == 0 || p.place_cooldown > 0.0 {
                return;
            }
        }
        let p = &self.players[owner];
        let eye = Vec3::new(p.pos.x, p.pos.y + EYE_HEIGHT, p.pos.z);
        let look = look_direction(p.yaw, p.pitch);
        let is_bucket = matches!(item, Item::WaterBucket | Item::LavaBucket);
        let Some((target, _normal)) = self.world.raycast_place_target(
            eye,
            look,
            cfg().combat.place_reach,
            &self.terrain,
            is_bucket,
        ) else {
            return;
        };
        // A solid block / fluid source can't be placed into a living player.
        if !is_bucket || matches!(item, Item::LavaBucket) {
            let tbox = target.box_of();
            let blocked = self.players.iter().any(|q| {
                q.alive() && crate::physics::aabb_overlap(&Aabb::player_at(q.pos), &tbox)
            });
            if blocked {
                return;
            }
        }
        {
            let p = &mut self.players[owner];
            p.counts[slot] -= 1;
            p.place_cooldown = cfg().combat.place_cooldown_seconds;
        }
        match item {
            Item::Planks => self.world.place_block(target, Block::Planks, &self.terrain),
            Item::Cobweb => self.world.place_block(target, Block::Cobweb, &self.terrain),
            Item::WaterBucket => self.world.place_source(FluidKind::Water, target, &self.terrain),
            Item::LavaBucket => self.world.place_source(FluidKind::Lava, target, &self.terrain),
            _ => {}
        }
    }
}

fn yaw_towards(from: Vec3, to: Vec3) -> f32 {
    let dx = to.x - from.x;
    let dz = to.z - from.z;
    (-dx).atan2(dz)
}

fn new_terrain(rng: &mut impl Rng) -> Terrain {
    if cfg().terrain_flat_only {
        Terrain::flat()
    } else {
        Terrain::random(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kit::{HOTBAR_SLOTS, ITEM_COUNT};
    use crate::physics::ARENA_RADIUS;
    use crate::player::FULLY_CHARGED;

    fn noop() -> Action {
        Action::NOOP
    }
    fn attack_action() -> Action {
        Action { attack: true, ..Action::NOOP }
    }

    /// A test hotbar with every combat item at a known slot, so a test can
    /// say `sel(Item::Axe)` regardless of the process kit's real layout.
    const TEST_HOTBAR: [Item; HOTBAR_SLOTS] = [
        Item::Sword,
        Item::Axe,
        Item::Pickaxe,
        Item::Bow,
        Item::Crossbow,
        Item::Cobweb,
        Item::WaterBucket,
        Item::LavaBucket,
        Item::GoldenApple,
    ];
    /// The `TEST_HOTBAR` slot that holds `it`.
    fn sel(it: Item) -> usize {
        TEST_HOTBAR.iter().position(|&x| x == it).expect("item is in TEST_HOTBAR")
    }
    /// Put a player's selected slot on `it` without going through an action
    /// (models "already holding this, no same-tick switch").
    fn set_held(arena: &mut Arena, i: usize, it: Item) {
        arena.players[i].slot = sel(it);
        arena.players[i].held = it;
        arena.players[i].prev_held = it;
    }

    fn duel(a_hp: f32, b_hp: f32) -> Arena {
        let l = loadout(cfg().kit);
        let mk = |team: u8, hp: f32, z: f32, yaw: f32| {
            let mut p = Player::new(team);
            p.hp = hp;
            p.pos = Vec3::new(0.0, 0.0, z);
            p.yaw = yaw;
            // Tests select items by a fixed known layout, not the kit's.
            p.hotbar = TEST_HOTBAR;
            p.slot = 0;
            p.held = Item::Sword;
            p.prev_held = Item::Sword;
            p.counts = l.counts;
            p.arrows = l.arrows;
            p.has_shield = l.has_shield;
            p.crossbow_loaded = l.crossbow_preloaded;
            p.sharpness = l.sharpness;
            p.power = l.power;
            p.piercing = l.piercing;
            p
        };
        Arena {
            players: vec![mk(0, a_hp, 0.0, 0.0), mk(1, b_hp, 1.0, std::f32::consts::PI)],
            projectiles: Vec::new(),
            world: BlockWorld::default(),
            time_left: cfg().match_time_seconds,
            terrain: Terrain::flat(),
            team_size: 1,
            rng: StdRng::seed_from_u64(42),
            collision_scratch: Vec::new(),
        }
    }

    /// Vanilla melee hit detection is client-side: a laggy attacker's swing
    /// lands where the target *was* on their screen, not where the server has
    /// it now. Same geometry with no latency is a clean miss.
    #[test]
    fn a_laggy_swing_lands_on_the_targets_stale_position() {
        use crate::player::MAX_DELAY_TICKS;
        let stale = Vec3::new(0.0, 0.0, 2.0); // dead ahead of the attacker
        let now = Vec3::new(5.0, 0.0, 2.0); // real position, well off the line

        // Drive `resolve_melee` directly so the swing geometry is fixed and
        // the ping is exactly what we set (no per-tick jitter / clamp).
        let hits_with_ping = |ping: f32| {
            let arena = duel(combat::MAX_HP, combat::MAX_HP);
            let mut players = arena.players;
            players[0].ping_ms = ping;
            players[1].pos = now;
            players[1].history.clear();
            // Older ticks: the target was dead ahead. Newest tick: it has
            // already moved to `now`.
            for i in 0..=MAX_DELAY_TICKS {
                let mut s = players[1].snapshot();
                s.pos = if i == MAX_DELAY_TICKS { now } else { stale };
                players[1].history.push_back(s);
            }
            let mut ev = vec![StepEvents::default(); 2];
            combat::resolve_melee(0, &mut players, &arena.world, &arena.terrain, &mut ev);
            ev[0].damage_dealt > 0.0
        };

        assert!(hits_with_ping(400.0), "a ~8-tick ping should hit the stale position");
        assert!(!hits_with_ping(0.0), "with no latency the same swing misses");
    }

    /// The server-side half of vanilla hit registration: the client may pick
    /// a target off its stale view, but if that target is now well beyond
    /// `server_interact_range` the server rejects the hit outright - the
    /// laggy attacker's screen lied.
    #[test]
    fn a_laggy_swing_whiffs_when_the_target_has_since_fled_out_of_server_range() {
        use crate::player::MAX_DELAY_TICKS;
        let stale = Vec3::new(0.0, 0.0, 2.0); // was dead ahead
        let gone = Vec3::new(10.0, 0.0, 2.0); // now ~10 blocks away, past 6.0

        let arena = duel(combat::MAX_HP, combat::MAX_HP);
        let mut players = arena.players;
        players[0].ping_ms = 400.0;
        players[1].pos = gone;
        players[1].history.clear();
        for i in 0..=MAX_DELAY_TICKS {
            let mut s = players[1].snapshot();
            s.pos = if i == MAX_DELAY_TICKS { gone } else { stale };
            players[1].history.push_back(s);
        }
        let mut ev = vec![StepEvents::default(); 2];
        combat::resolve_melee(0, &mut players, &arena.world, &arena.terrain, &mut ev);
        assert_eq!(ev[0].damage_dealt, 0.0, "server should reject the out-of-range hit");
        assert_eq!(players[1].hp, combat::MAX_HP, "target takes no damage");
        // The swing still consumed the attack charge (it happened client-side).
        assert_eq!(players[0].time_since_last_attack, 0.0);
    }

    #[test]
    fn raycast_hit_dead_ahead_deals_damage() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        let obs = arena.step(&[attack_action(), noop()]);
        assert!(obs[0].damage_dealt > 0.0);
        assert!(arena.players[1].hp < combat::MAX_HP);
    }

    #[test]
    fn raycast_misses_when_facing_away() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].yaw = std::f32::consts::PI;
        assert_eq!(arena.step(&[attack_action(), noop()])[0].damage_dealt, 0.0);
    }

    #[test]
    fn raycast_misses_beyond_reach() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[1].pos.z = cfg().combat.attack_reach + 1.5;
        assert_eq!(arena.step(&[attack_action(), noop()])[0].damage_dealt, 0.0);
    }

    #[test]
    fn iframes_stop_a_weaker_follow_up_hit() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.rng = StdRng::seed_from_u64(7);
        let first = arena.step(&[attack_action(), noop()]);
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        let second = arena.step(&[attack_action(), noop()]);
        assert!(first[0].damage_dealt > 0.0);
        assert_eq!(second[0].damage_dealt, 0.0);
    }

    #[test]
    fn raised_shield_blocks_a_frontal_hit() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[1].has_shield = true; // the sword kit has none; the axe/uhc kits do
        let raise = Action { use_item: true, ..Action::NOOP };
        for _ in 0..6 {
            arena.step(&[noop(), raise]);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
        }
        assert!(arena.players[1].shield_up());
        let hp_before = arena.players[1].hp;
        let obs = arena.step(&[attack_action(), raise]);
        assert_eq!(obs[0].damage_dealt, 0.0, "frontal hit into a raised shield is blocked");
        assert_eq!(arena.players[1].hp, hp_before);
    }

    fn axe_arena() -> Arena {
        // Force the axe kit for these players regardless of process config.
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        for p in &mut arena.players {
            p.counts[Item::Axe.index()] = 1;
            p.counts[Item::Sword.index()] = 1;
            p.has_shield = true;
        }
        arena
    }

    /// Step with the given actions. Input latency is no longer simulated
    /// (vanilla client-authoritative movement), so this is just `step` - kept
    /// as a named helper the older tests still call.
    fn step_now(arena: &mut Arena, a0: Action, a1: Action) -> Vec<Observation> {
        arena.step(&[a0, a1])
    }

    #[test]
    fn an_axe_hit_disables_a_raised_shield() {
        let mut arena = axe_arena();
        let raise = Action { use_item: true, ..Action::NOOP };
        for _ in 0..6 {
            arena.step(&[noop(), raise]);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
        }
        assert!(arena.players[1].shield_up());
        // Already holding the axe (no same-tick switch), then swing.
        set_held(&mut arena, 0, Item::Axe);
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        step_now(&mut arena, Action { attack: true, slot: sel(Item::Axe), ..Action::NOOP }, raise);
        assert!(arena.players[1].shield_disabled_time > 0.0, "axe hit disabled the shield");
        assert!(!arena.players[1].shield_up(), "shield no longer blocks while disabled");
    }

    #[test]
    fn attribute_swap_axe_to_sword_carries_the_disable_trait() {
        let mut arena = axe_arena();
        let raise = Action { use_item: true, ..Action::NOOP };
        for _ in 0..6 {
            arena.step(&[noop(), raise]);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
        }
        assert!(arena.players[1].shield_up());
        // Holding the axe; this tick switch axe -> sword AND attack. The hit
        // should still disable the shield (axe trait carried by MC-28289).
        set_held(&mut arena, 0, Item::Axe);
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        step_now(
            &mut arena,
            Action { attack: true, slot: sel(Item::Sword), ..Action::NOOP },
            raise,
        );
        assert_eq!(arena.players[0].held, Item::Sword, "now holding the sword");
        assert!(arena.players[1].shield_disabled_time > 0.0, "swapped hit carried the axe disable");
    }

    #[test]
    fn a_pending_swap_lockout_suppresses_the_attack_then_clears() {
        // The `modern` input order sets `swap_lockout` on a slot switch; this
        // exercises the `step` gate directly (config is process-wide, so the
        // input-order forcing itself is covered in config.rs's tests).
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        set_held(&mut arena, 0, Item::Sword);
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        // A real swap sets this *after* the per-tick decay, so seed it above
        // one tick's worth to survive this step's decay and still gate.
        arena.players[0].swap_lockout = 2.0 * DT;
        let locked = step_now(&mut arena, attack_action(), noop());
        assert_eq!(locked[0].damage_dealt, 0.0, "attack locked out on the swap tick");
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        let freed = step_now(&mut arena, attack_action(), noop());
        assert!(freed[0].damage_dealt > 0.0, "lockout cleared the next tick");
    }

    #[test]
    fn swapping_slots_drops_a_raised_shield() {
        let mut arena = axe_arena();
        set_held(&mut arena, 0, Item::Sword);
        let raise = Action { use_item: true, slot: sel(Item::Sword), ..Action::NOOP };
        for _ in 0..6 {
            step_now(&mut arena, raise, noop());
        }
        assert!(arena.players[0].shield_up(), "shield raised while holding the sword");
        // Switch to the axe with use still held: the shield must drop.
        step_now(
            &mut arena,
            Action { use_item: true, slot: sel(Item::Axe), ..Action::NOOP },
            noop(),
        );
        assert_eq!(arena.players[0].shield_time, 0.0, "the swap reset the shield raise");
        assert!(!arena.players[0].shield_up(), "shield no longer up after the swap");
    }

    #[test]
    fn selecting_an_empty_slot_gives_an_empty_hand() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].hotbar[5] = Item::Empty;
        // Land on the empty slot first (no same-tick switch), then swing.
        arena.players[0].slot = 5;
        arena.players[0].held = Item::Empty;
        arena.players[0].prev_held = Item::Empty;
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        let obs = step_now(
            &mut arena,
            Action { attack: true, slot: 5, ..Action::NOOP },
            noop(),
        );
        assert_eq!(arena.players[0].held, Item::Empty, "empty slot selected");
        assert!(obs[0].damage_dealt > 0.0, "an empty hand still lands a (weak) hit");
    }

    #[test]
    fn hotbar_observation_matches_the_kit_layout() {
        let mut arena = Arena::new(1);
        let obs = arena.step(&[noop(), noop()]);
        let layout = loadout(cfg().kit).hotbar;
        let seen: Vec<usize> = obs[0].hotbar.iter().map(|&x| x as usize).collect();
        let want: Vec<usize> = layout.iter().map(|it| it.index()).collect();
        assert_eq!(seen, want, "obs hotbar block == the kit's slot->item layout");
        assert_eq!(obs[0].self_slot as usize, loadout(cfg().kit).default_slot);
    }

    /// `slot` action `HOTBAR_SLOTS + item` hotkeys that item into the
    /// selected slot.
    fn hotkey(item: Item) -> usize {
        crate::kit::HOTBAR_SLOTS + item.index()
    }

    #[test]
    fn a_hotkey_brings_an_off_hotbar_item_into_the_selected_slot() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        // give the player a golden head and clear slot 0's item so the hotkey
        // has a clean target
        arena.players[0].counts[Item::GoldenHead.index()] = 2;
        arena.players[0].slot = 0;
        arena.players[0].hotbar[0] = Item::Empty;
        arena.players[0].held = Item::Empty;
        step_now(
            &mut arena,
            Action { slot: hotkey(Item::GoldenHead), ..Action::NOOP },
            noop(),
        );
        assert_eq!(arena.players[0].held, Item::GoldenHead, "hotkey'd into hand");
        assert_eq!(arena.players[0].hotbar[0], Item::GoldenHead);
        assert_eq!(arena.players[0].counts[Item::GoldenHead.index()], 2, "count untouched by a hotkey");
    }

    #[test]
    fn a_hotkey_for_an_unowned_item_is_a_noop() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].counts[Item::GoldenHead.index()] = 0;
        let before = arena.players[0].held;
        step_now(&mut arena, Action { slot: hotkey(Item::GoldenHead), ..Action::NOOP }, noop());
        assert_eq!(arena.players[0].held, before, "no golden head owned -> nothing happens");
    }

    #[test]
    fn a_hotkey_for_an_item_already_on_the_bar_swaps_the_two_slots() {
        let mut arena = axe_arena();
        // TEST_HOTBAR: slot 0 = Sword, slot 1 = Axe. Select slot 0, hotkey the
        // axe -> slots 0 and 1 trade contents.
        set_held(&mut arena, 0, Item::Sword);
        step_now(&mut arena, Action { slot: hotkey(Item::Axe), ..Action::NOOP }, noop());
        assert_eq!(arena.players[0].hotbar[0], Item::Axe);
        assert_eq!(arena.players[0].hotbar[1], Item::Sword);
        assert_eq!(arena.players[0].held, Item::Axe);
    }

    #[test]
    fn a_hotkey_on_the_attack_tick_still_carries_the_swapped_out_attributes() {
        let mut arena = axe_arena();
        let raise = Action { use_item: true, ..Action::NOOP };
        for _ in 0..6 {
            arena.step(&[noop(), raise]);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
        }
        assert!(arena.players[1].shield_up());
        // Holding the axe; hotkey the sword into this slot AND attack. Legacy
        // input order: the hit still carries the axe's shield-disable.
        set_held(&mut arena, 0, Item::Axe);
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        step_now(
            &mut arena,
            Action { attack: true, slot: hotkey(Item::Sword), ..Action::NOOP },
            raise,
        );
        assert_eq!(arena.players[0].held, Item::Sword);
        assert!(arena.players[1].shield_disabled_time > 0.0, "hotkey swap carried the axe disable");
    }

    #[test]
    fn a_drawn_bow_fires_an_arrow_on_release() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        for p in &mut arena.players {
            p.counts[Item::Bow.index()] = 1;
            p.arrows = 6;
        }
        set_held(&mut arena, 0, Item::Bow);
        arena.players[0].pitch = -0.2; // arc it up so it doesn't instantly hit
        arena.players[1].pos.z = 40.0; // and put the target out past the arena
        let draw = Action { use_item: true, slot: sel(Item::Bow), ..Action::NOOP };
        for _ in 0..25 {
            step_now(&mut arena, draw, noop());
        }
        assert!(arena.projectiles.is_empty(), "still drawing, nothing fired");
        step_now(&mut arena, Action { slot: sel(Item::Bow), ..Action::NOOP }, noop()); // release
        assert_eq!(arena.players[0].arrows, 5, "one loose arrow consumed");
        assert_eq!(arena.projectiles.len(), 1, "the arrow is in flight");
    }

    #[test]
    fn eating_a_golden_apple_heals_and_grants_absorption() {
        let mut arena = duel(6.0, combat::MAX_HP);
        arena.players[0].counts[Item::GoldenApple.index()] = 2;
        set_held(&mut arena, 0, Item::GoldenApple);
        let eat = Action { use_item: true, slot: sel(Item::GoldenApple), ..Action::NOOP };
        let ticks = (cfg().combat.golden_apple_eat_seconds / DT).ceil() as usize + 2;
        for _ in 0..ticks {
            step_now(&mut arena, eat, noop());
        }
        assert!(arena.players[0].hp > 6.0, "golden apple healed");
        assert!(arena.players[0].absorption > 0.0, "golden apple gave absorption");
        assert_eq!(arena.players[0].counts[Item::GoldenApple.index()], 1, "one consumed");
    }

    #[test]
    fn a_placed_cobweb_slows_a_player_who_walks_into_it() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].counts[Item::Cobweb.index()] = 8;
        set_held(&mut arena, 0, Item::Cobweb);
        arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
        arena.players[0].yaw = 0.0;
        arena.players[0].pitch = 0.55; // aim at the ground a couple blocks ahead
        step_now(&mut arena, Action { use_item: true, slot: sel(Item::Cobweb), ..Action::NOOP }, noop());
        assert!(!arena.world.is_empty(), "a cobweb block was placed");
        // Sprint into it and check the velocity is crushed.
        set_held(&mut arena, 0, Item::Sword);
        let run = Action { move_z: 1.0, sprint: true, ..Action::NOOP };
        for _ in 0..12 {
            arena.step(&[run, noop()]);
        }
        assert!(arena.players[0].horizontal_speed() < 0.15, "cobweb crushed the player's speed");
    }

    /// Aim player 0's pickaxe/axe at a block one cell ahead and count the
    /// ticks to break it (`None` if it never breaks within `max_ticks`).
    fn ticks_to_mine(item: Item, block: crate::blocks::Block, max_ticks: usize) -> Option<usize> {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[1].pos = Vec3::new(0.0, 0.0, 40.0); // out of the way
        arena.players[0].pos = Vec3::new(0.5, 0.0, 0.5);
        arena.players[0].yaw = 0.0;
        arena.players[0].pitch = 0.62; // look down at the block a couple ahead
        set_held(&mut arena, 0, item);
        arena.players[0].efficiency = 3;
        arena
            .world
            .place_block_for_test(crate::blocks::Cell::new(0, 0, 2), block);
        let mine = Action { attack: true, slot: sel(item), ..Action::NOOP };
        (0..max_ticks).find(|_| {
            arena.step(&[mine, noop()]);
            arena.world.is_empty()
        })
    }

    #[test]
    fn the_pickaxe_mines_stone_fast_but_never_the_terrain_floor() {
        // A straight-down ray finds no placed block, so holding attack at the
        // ground can never dig a hole in the terrain.
        let arena = duel(combat::MAX_HP, combat::MAX_HP);
        assert!(
            arena
                .world
                .raycast_mine_target(
                    Vec3::new(0.5, 1.62, 0.5),
                    Vec3::new(0.0, -1.0, 0.0),
                    4.5,
                    &arena.terrain,
                )
                .is_none(),
            "terrain floor is not mineable"
        );
        let t = ticks_to_mine(Item::Pickaxe, crate::blocks::Block::Stone, 40)
            .expect("Efficiency-3 pickaxe broke the stone");
        assert!(t < 15, "and it broke it quickly ({} ticks)", t + 1);
    }

    #[test]
    fn the_axe_breaks_planks_much_faster_than_the_pickaxe() {
        let axe = ticks_to_mine(Item::Axe, crate::blocks::Block::Planks, 200)
            .expect("the axe (correct tool) broke the planks");
        let pick = ticks_to_mine(Item::Pickaxe, crate::blocks::Block::Planks, 200)
            .expect("the pickaxe eventually broke the planks too");
        assert!(
            pick > axe * 4,
            "pickaxe on wood is the wrong tool: {pick} ticks vs the axe's {axe}"
        );
    }

    #[test]
    fn releasing_attack_resets_mining_progress() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[1].pos = Vec3::new(0.0, 0.0, 40.0);
        arena.players[0].pos = Vec3::new(0.5, 0.0, 0.5);
        arena.players[0].pitch = 0.62;
        set_held(&mut arena, 0, Item::Pickaxe);
        arena.players[0].efficiency = 0; // slow, so one tick can't finish it
        arena
            .world
            .place_block_for_test(crate::blocks::Cell::new(0, 0, 2), crate::blocks::Block::Obsidian);
        let mine = Action { attack: true, slot: sel(Item::Pickaxe), ..Action::NOOP };
        arena.step(&[mine, noop()]);
        assert!(arena.players[0].mining_progress > 0.0, "started mining");
        arena.step(&[Action { slot: sel(Item::Pickaxe), ..Action::NOOP }, noop()]); // let go
        assert_eq!(arena.players[0].mining_progress, 0.0, "progress reset");
        assert!(!arena.world.is_empty(), "block still there");
    }

    #[test]
    fn a_water_bucket_makes_a_spreading_puddle_and_a_lava_bucket_meeting_it_makes_stone() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].pos = Vec3::new(0.0, 0.0, 20.0); // out of the way
        arena.players[1].pos = Vec3::new(0.0, 0.0, 24.0);
        let t = arena.terrain;
        arena.world.place_source(FluidKind::Water, crate::blocks::Cell::new(0, 0, 0), &t);
        for _ in 0..30 {
            arena.step(&[noop(), noop()]);
        }
        let water_cells = arena.world.iter_blocks().filter(|(_, b)| b.is_water()).count();
        assert!(water_cells > 2, "water spread to {water_cells} cells");

        arena.world.place_source(FluidKind::Lava, crate::blocks::Cell::new(3, 0, 0), &t);
        for _ in 0..60 {
            arena.step(&[noop(), noop()]);
        }
        let generated = arena.world.iter_blocks().any(|(_, b)| {
            matches!(
                b,
                crate::blocks::Block::Stone
                    | crate::blocks::Block::Cobblestone
                    | crate::blocks::Block::Obsidian
            )
        });
        assert!(generated, "lava meeting water generated a stone-family block");
    }

    #[test]
    fn one_bucket_click_places_one_source_ahead_not_underfoot() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].counts[Item::WaterBucket.index()] = 4;
        arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
        arena.players[0].yaw = 0.0;
        arena.players[0].pitch = 0.55; // aim at the ground a couple blocks ahead
        set_held(&mut arena, 0, Item::WaterBucket);
        step_now(
            &mut arena,
            Action { use_item: true, slot: sel(Item::WaterBucket), ..Action::NOOP },
            noop(),
        );
        for _ in 0..20 {
            arena.step(&[noop(), noop()]);
        }
        assert!(arena.world.iter_blocks().any(|(_, b)| b.is_water()), "a puddle appeared ahead");
    }

    #[test]
    fn determinism_holds() {
        let mut a1 = Arena::new(1234);
        let mut a2 = Arena::new(1234);
        let strafe = Action { move_x: 0.7, jump: true, ..Action::NOOP };
        for i in 0..40 {
            let acts = if i % 2 == 0 { [attack_action(), strafe] } else { [strafe, attack_action()] };
            let s1 = a1.step(&acts);
            let s2 = a2.step(&acts);
            assert_eq!(s1[0].self_hp, s2[0].self_hp, "step {i}");
        }
        assert_eq!(a1.players[0].pos.x, a2.players[0].pos.x);
    }

    #[test]
    fn fluid_spread_is_deterministic() {
        let mk = || {
            let mut a = duel(combat::MAX_HP, combat::MAX_HP);
            a.players[0].counts[Item::WaterBucket.index()] = 4;
            a.players[0].counts[Item::LavaBucket.index()] = 4;
            a.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
            a.players[0].pitch = 0.55; // aim at the ground ahead
            set_held(&mut a, 0, Item::WaterBucket);
            a
        };
        let (mut a1, mut a2) = (mk(), mk());
        let place_w = Action { use_item: true, slot: sel(Item::WaterBucket), ..Action::NOOP };
        for _ in 0..40 {
            let r1 = a1.step(&[place_w, noop()]);
            let r2 = a2.step(&[place_w, noop()]);
            assert_eq!(r1[0].self_hp, r2[0].self_hp);
        }
        let c1 = a1.world.iter_blocks().count();
        let c2 = a2.world.iter_blocks().count();
        assert_eq!(c1, c2, "fluid grid diverged ({c1} vs {c2})");
        assert!(c1 > 0);
    }

    #[test]
    fn players_cannot_leave_the_platform() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].pos = Vec3::new(ARENA_RADIUS * 5.0, 0.0, 0.0);
        arena.players[0].vel = Vec3::new(50.0, 0.0, 0.0);
        arena.step(&[noop(), noop()]);
        let r = (arena.players[0].pos.x.powi(2) + arena.players[0].pos.z.powi(2)).sqrt();
        assert!(r <= ARENA_RADIUS + 1e-3);
    }

    #[test]
    fn a_long_fall_deals_damage() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].pos.y = 12.0;
        arena.players[0].on_ground = false;
        let mut took = false;
        for _ in 0..120 {
            if arena.step(&[noop(), noop()])[0].damage_taken > 0.0 {
                took = true;
            }
        }
        assert!(took);
    }

    #[test]
    fn fall_damage_bypasses_armor_points_but_not_protection() {
        // Same fall, once with the kit's real Protection EPF and once with
        // none. Armor *points* (20, both cases) must not matter; the EPF run
        // must land strictly less damage, and both must be deterministic.
        let fall_dmg = |epf: f32| -> f32 {
            let mut a = duel(combat::MAX_HP, combat::MAX_HP);
            a.players[0].pos = Vec3::new(0.0, 20.0, 0.0);
            a.players[0].on_ground = false;
            a.players[0].armor.protection_epf = epf;
            let mut taken = 0.0;
            for _ in 0..120 {
                taken += a.step(&[noop(), noop()])[0].damage_taken;
            }
            taken
        };
        let bare = fall_dmg(0.0);
        let protected = fall_dmg(10.0); // 1 - 10/25 = 0.6
        assert!(bare > 0.0);
        assert!((protected - bare * 0.6).abs() < 1e-3, "EPF halves via 1-epf/25: {protected} vs {bare}");
        assert_eq!(fall_dmg(0.0), bare, "deterministic - no armor roll");
    }

    #[test]
    fn a_sprint_hit_ends_the_attackers_sprint() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].sprinting = true;
        arena.players[0].time_since_last_attack = FULLY_CHARGED;
        step_now(&mut arena, attack_action(), noop());
        assert!(!arena.players[0].sprinting, "sprint ends after a sprint-attack");
    }

    #[test]
    fn crouching_slows_you_down() {
        let travel = |sneak: bool| {
            let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
            arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
            arena.players[0].yaw = 0.0;
            arena.players[1].pos = Vec3::new(30.0, 0.0, 0.0); // out of the way
            let a = Action { move_z: 1.0, sneak, ..Action::NOOP };
            for _ in 0..10 {
                arena.step(&[a, noop()]);
            }
            arena.players[0].pos.z
        };
        assert!(travel(true) < travel(false) * 0.6, "sneaking is much slower than walking");
    }

    #[test]
    fn a_crouched_target_is_shorter_and_harder_to_hit_high() {
        let hit_a_target = |crouched: bool| -> f32 {
            let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
            arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
            arena.players[0].yaw = 0.0;
            arena.players[0].pitch = 0.0; // level - grazes the head
            arena.players[1].pos = Vec3::new(0.0, 0.0, 2.0);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
            let defend = Action { sneak: crouched, ..Action::NOOP };
            step_now(&mut arena, attack_action(), defend)[0].damage_dealt
        };
        assert_eq!(hit_a_target(true), 0.0, "the level shot sails over the crouched target");
        assert!(hit_a_target(false) > 0.0, "and connects with them standing");
    }

    #[test]
    fn sneaking_will_not_walk_off_a_placed_block() {
        let end_state = |sneak: bool| -> (f32, f32) {
            let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
            arena.world.place_block_for_test(crate::blocks::Cell::new(0, 0, 0), crate::blocks::Block::Planks);
            arena.players[0].pos = Vec3::new(0.0, 1.0, 0.0); // on top of the block
            arena.players[0].on_ground = true;
            arena.players[0].yaw = 0.0;
            arena.players[1].pos = Vec3::new(30.0, 0.0, 0.0);
            let a = Action { move_z: 1.0, sneak, ..Action::NOOP };
            for _ in 0..30 {
                arena.step(&[a, noop()]);
            }
            (arena.players[0].pos.z, arena.players[0].pos.y)
        };
        let (_sz, sy) = end_state(true);
        let (wz, wy) = end_state(false);
        assert!(sy > 0.9, "sneaking kept the player on top of the block (y={sy})");
        assert!(wz > 1.0 && wy < 0.9, "without sneak they walk off and fall (z={wz} y={wy})");
    }

    #[test]
    fn landing_attacks_drain_saturation_then_food() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
        arena.players[0].yaw = 0.0;
        arena.players[0].saturation = 1.0;
        let food0 = arena.players[0].food;
        for _ in 0..120 {
            arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
            arena.players[1].pos = Vec3::new(0.0, 0.0, 1.5); // stay in reach
            arena.players[1].vel = Vec3::ZERO;
            arena.players[1].hp = combat::MAX_HP; // never dies -> no spawn reset
            arena.players[1].hurt_time_left = 0.0;
            step_now(&mut arena, attack_action(), noop());
        }
        assert_eq!(arena.players[0].saturation, 0.0, "saturation spent first");
        assert!(arena.players[0].food < food0, "then food");
    }

    #[test]
    fn a_starving_player_cannot_start_sprinting() {
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].food = 4.0; // below min_food_to_sprint (6)
        let run = Action { move_z: 1.0, sprint: true, ..Action::NOOP };
        step_now(&mut arena, run, noop());
        assert!(!arena.players[0].sprinting);
    }

    #[test]
    fn no_critical_hit_while_in_water() {
        let dive_attack = |water: bool| -> bool {
            let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
            let t = arena.terrain;
            arena.players[0].pos = Vec3::new(0.0, 0.6, 0.0);
            arena.players[0].yaw = 0.0;
            arena.players[0].vel = Vec3::new(0.0, -0.3, 0.0); // falling
            arena.players[0].on_ground = false;
            arena.players[0].pitch = 0.5; // look down at the target below
            arena.players[0].time_since_last_attack = FULLY_CHARGED;
            arena.players[1].pos = Vec3::new(0.0, 0.0, 1.2);
            let _ = t;
            if water {
                arena.world.place_block_for_test(crate::blocks::Cell::new(0, 0, 0), crate::blocks::Block::Water(0));
            }
            step_now(&mut arena, attack_action(), noop())[0].swept
        };
        // in water: the would-be crit falls back to Sweep (swept flag set);
        // out of water: it's a Critical (swept flag clear).
        assert!(dive_attack(true), "in water -> not a crit");
        assert!(!dive_attack(false), "out of water -> crit");
    }

    #[test]
    fn a_cobweb_negates_fall_damage() {
        let fall_into = |cobweb: bool| -> f32 {
            let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
            let t = arena.terrain;
            arena.players[0].pos = Vec3::new(0.0, 8.0, 0.0);
            arena.players[0].on_ground = false;
            if cobweb {
                for y in 0..5 {
                    arena.world.place_block(crate::blocks::Cell::new(0, y, 0), crate::blocks::Block::Cobweb, &t);
                }
            }
            let mut taken = 0.0;
            for _ in 0..80 {
                taken += arena.step(&[noop(), noop()])[0].damage_taken;
            }
            taken
        };
        assert!(fall_into(false) > 0.0, "a bare fall hurts");
        assert_eq!(fall_into(true), 0.0, "falling through cobweb doesn't");
    }

    #[test]
    fn a_submerged_player_sinks_slowly_and_swims_up_on_jump() {
        let fill = |arena: &mut Arena| {
            for y in 0..8 {
                arena.world.place_block_for_test(crate::blocks::Cell::new(0, y, 0), crate::blocks::Block::Water(0));
            }
        };
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].pos = Vec3::new(0.0, 4.0, 0.0);
        arena.players[0].on_ground = false;
        arena.players[1].pos = Vec3::new(6.0, 0.0, 0.0);
        let free_fall = {
            let mut a2 = duel(combat::MAX_HP, combat::MAX_HP);
            a2.players[0].pos = Vec3::new(0.0, 4.0, 0.0);
            a2.players[0].on_ground = false;
            a2.players[1].pos = Vec3::new(6.0, 0.0, 0.0);
            for _ in 0..4 {
                a2.step(&[noop(), noop()]);
            }
            4.0 - a2.players[0].pos.y
        };
        let y0 = arena.players[0].pos.y;
        for _ in 0..4 {
            fill(&mut arena);
            arena.step(&[noop(), noop()]);
        }
        let sank = y0 - arena.players[0].pos.y;
        assert!(sank > 0.0 && sank < free_fall * 0.5, "sinks far slower than a free fall: {sank} vs {free_fall}");

        let y1 = arena.players[0].pos.y;
        for _ in 0..6 {
            fill(&mut arena);
            arena.step(&[Action { jump: true, ..Action::NOOP }, noop()]);
        }
        assert!(arena.players[0].pos.y > y1, "holding jump swims up");
    }

    #[test]
    fn a_golden_apple_regenerates_over_time_not_instantly() {
        let mut arena = duel(6.0, combat::MAX_HP);
        arena.players[0].counts[Item::GoldenApple.index()] = 2;
        set_held(&mut arena, 0, Item::GoldenApple);
        let eat = Action { use_item: true, slot: sel(Item::GoldenApple), ..Action::NOOP };
        let ticks = (cfg().combat.golden_apple_eat_seconds / DT).ceil() as usize + 1;
        for _ in 0..ticks {
            step_now(&mut arena, eat, noop());
        }
        assert!((arena.players[0].hp - 6.0).abs() < 0.5, "no instant heal on finishing the apple");
        assert!(arena.players[0].absorption > 0.0, "absorption is immediate though");
        for _ in 0..120 {
            step_now(&mut arena, noop(), noop());
        }
        assert!(arena.players[0].hp > 8.0, "the HP came back via regen over time");
    }

    #[test]
    fn timeout_awards_the_higher_total_hp_team() {
        let mut arena = duel(combat::MAX_HP, 5.0);
        arena.time_left = DT * 0.5;
        let obs = arena.step(&[noop(), noop()]);
        assert!(obs[0].won && obs[1].lost);
    }

    #[test]
    fn arena_resets_after_a_match() {
        let mut arena = duel(combat::MAX_HP, 0.5);
        arena.step(&[attack_action(), noop()]);
        assert_eq!(arena.players[0].hp, combat::MAX_HP);
        assert_eq!(arena.time_left, cfg().match_time_seconds);
    }

    #[test]
    fn a_laggy_observer_sees_a_moving_enemy_in_its_past() {
        // Player 1 sprints in a straight line; player 0 has a high ping and
        // player 2 (fresh) a near-zero one. The laggy view must trail behind.
        let mut arena = duel(combat::MAX_HP, combat::MAX_HP);
        arena.players[0].base_ping_ms = 100.0;
        arena.players[1].base_ping_ms = 100.0;
        arena.players[0].pos = Vec3::new(0.0, 0.0, 0.0);
        arena.players[1].pos = Vec3::new(0.0, 0.0, 3.0);
        arena.players[1].yaw = 0.0;
        let run = Action { move_z: 1.0, sprint: true, ..Action::NOOP };
        let mut obs = arena.step(&[noop(), run]);
        for _ in 0..20 {
            obs = arena.step(&[noop(), run]);
        }
        let seen_z = obs[0].enemies[0].rel_z; // enemy z relative to still player 0
        let true_z = arena.players[1].pos.z;
        assert!(
            seen_z + 0.3 < true_z,
            "laggy view (rel_z {seen_z}) trails the true enemy z {true_z}"
        );
    }

    #[test]
    fn observation_slot_counts_match_config() {
        let arena = duel(combat::MAX_HP, combat::MAX_HP);
        let obs = observation::build(&arena, 0, StepEvents::default());
        assert_eq!(obs.enemies.len(), cfg().max_observed_enemies);
        assert_eq!(obs.teammates.len(), cfg().max_observed_teammates);
        assert_eq!(obs.projectiles.len(), cfg().max_observed_projectiles);
        assert_eq!(obs.block_view.len(), cfg().block_view_size * cfg().block_view_size);
        assert_eq!(obs.inventory.len(), ITEM_COUNT - 1);
    }
}
