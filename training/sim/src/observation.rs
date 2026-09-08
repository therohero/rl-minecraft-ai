//! Turns an `Arena`'s post-step state into the wire `Observation` one
//! player-slot sees: everything is expressed relative to "self" (positions
//! and velocities rotated into the observer's yaw frame), the nearest few
//! enemies / teammates / arrows, the inventory + hotbar layout, a
//! yaw-rotated block-grid view, and this step's reward + event flags.
//!
//! `python/features.py` is the mirror image of this on the trainer side -
//! keep the two in lockstep (bump `protocol::WIRE_VERSION` when the layout
//! changes).

use crate::arena::{Arena, StepEvents};
use crate::combat;
use crate::config::cfg;
use crate::kit::ITEM_COUNT;
use crate::protocol::{BlockColumnObs, Observation, OtherPlayer, ProjectileObs};

#[allow(clippy::needless_range_loop)]
pub(crate) fn build(arena: &Arena, me_idx: usize, ev: StepEvents) -> Observation {
    let players = &arena.players;
    let n = players.len();
    let me = &players[me_idx];
    let (sin_y, cos_y) = me.yaw.sin_cos();
    let rot = |dx: f32, dz: f32| (dx * cos_y - dz * sin_y, dx * sin_y + dz * cos_y);
    let c = &cfg().combat;

    // Nearest living enemies / teammates, by squared distance.
    let mut enemies: Vec<(f32, usize)> = Vec::new();
    let mut mates: Vec<(f32, usize)> = Vec::new();
    for j in 0..n {
        if j == me_idx || !players[j].alive() {
            continue;
        }
        let p = &players[j];
        let (dx, dy, dz) = (p.pos.x - me.pos.x, p.pos.y - me.pos.y, p.pos.z - me.pos.z);
        let d2 = dx * dx + dy * dy + dz * dz;
        if p.team == me.team {
            mates.push((d2, j));
        } else {
            enemies.push((d2, j));
        }
    }
    enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    mates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // How stale this observer's view of everyone else is (vanilla: you see
    // others one round-trip in the past). Your own state, below, stays live.
    let obs_ping = me.ping_ms;
    let to_block = |j: usize| -> OtherPlayer {
        let s = players[j].delayed_view(obs_ping);
        let (rx, rz) = rot(s.pos.x - me.pos.x, s.pos.z - me.pos.z);
        OtherPlayer {
            present: true,
            hp: s.hp + s.absorption,
            rel_x: rx,
            rel_y: s.pos.y - me.pos.y,
            rel_z: rz,
            vel_x: s.vel.x,
            vel_y: s.vel.y,
            vel_z: s.vel.z,
            ground_height: arena.world.support_y(
                s.pos.x.floor() as i32,
                s.pos.z.floor() as i32,
                &arena.terrain,
                s.pos.y.floor() as i32 + cfg().combat.block_ceiling as i32,
            ),
            blocking: s.shield,
            eating: s.eating,
            held_ranged: s.held_ranged,
            sneaking: s.sneaking,
        }
    };
    let fill = |src: &[(f32, usize)], count: usize| -> Vec<OtherPlayer> {
        let mut v: Vec<OtherPlayer> = src.iter().take(count).map(|&(_, j)| to_block(j)).collect();
        v.resize(count, OtherPlayer::default());
        v
    };

    // Nearest in-flight projectiles that aren't this player's own.
    let mut projs: Vec<(f32, usize)> = arena
        .projectiles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.owner != me_idx)
        .map(|(k, p)| {
            let (dx, dy, dz) = (p.pos.x - me.pos.x, p.pos.y - me.pos.y, p.pos.z - me.pos.z);
            (dx * dx + dy * dy + dz * dz, k)
        })
        .collect();
    projs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut proj_blocks: Vec<ProjectileObs> = projs
        .iter()
        .take(cfg().max_observed_projectiles)
        .map(|&(_, k)| {
            let (ppos, pvel) = arena.projectiles[k].delayed_view(obs_ping);
            let (rx, rz) = rot(ppos.x - me.pos.x, ppos.z - me.pos.z);
            let (vrx, vrz) = rot(pvel.x, pvel.z);
            ProjectileObs {
                present: true,
                rel_x: rx,
                rel_y: ppos.y - me.pos.y,
                rel_z: rz,
                vel_x: vrx,
                vel_y: pvel.y,
                vel_z: vrz,
            }
        })
        .collect();
    proj_blocks.resize(cfg().max_observed_projectiles, ProjectileObs::default());

    // Yaw-rotated block-grid view around the player.
    let bv = cfg().block_view_size;
    let block_view: Vec<BlockColumnObs> = if bv > 0 {
        let inv_amp = 1.0 / cfg().terrain_max_amplitude.max(1e-3);
        arena
            .world
            .column_view(me.pos, me.yaw, &arena.terrain, bv)
            .into_iter()
            .map(|[h, w, l, cw]| BlockColumnObs { top_rel: h * inv_amp, water: w, lava: l, cobweb: cw })
            .collect()
    } else {
        Vec::new()
    };

    // Per-item inventory counts (Sword..GoldenHead) + the kit's slot->item
    // hotbar layout, so the policy can learn what it can switch to.
    let inventory: Vec<f32> = (1..ITEM_COUNT).map(|k| me.counts[k] as f32).collect();
    let hotbar: Vec<f32> = me.hotbar.iter().map(|it| it.index() as f32).collect();
    let swap_lockout_max = cfg().combat.swap_lockout_seconds.max(1e-3);

    let fwd = me.forward();
    let right = me.right();
    let slope_d = cfg().slope_sample_distance;
    // Voxel world: the ground under a point is the top of the highest solid
    // cell (terrain or placed block) in that column, up to the build ceiling.
    let ceil = |y: f32| y.floor() as i32 + cfg().combat.block_ceiling as i32;
    let ground_at = |x: f32, z: f32, y: f32| {
        arena
            .world
            .support_y(x.floor() as i32, z.floor() as i32, &arena.terrain, ceil(y))
    };
    let self_ground = ground_at(me.pos.x, me.pos.z, me.pos.y);
    let ahead = ground_at(me.pos.x + fwd.x * slope_d, me.pos.z + fwd.z * slope_d, me.pos.y);
    let right_h = ground_at(me.pos.x + right.x * slope_d, me.pos.z + right.z * slope_d, me.pos.y);
    let arena_radius = cfg().arena_radius;

    Observation {
        self_hp: me.hp,
        self_vel_x: me.vel.x,
        self_vel_y: me.vel.y,
        self_vel_z: me.vel.z,
        self_yaw: me.yaw,
        self_pitch: me.pitch,
        self_on_ground: me.on_ground,
        self_attack_cooldown: combat::strength_scale(me.time_since_last_attack, me.held_weapon_recharge()),
        self_ping_ms: me.ping_ms,
        self_shield: me.shield_fraction(),
        self_dist_from_center: (me.pos.x * me.pos.x + me.pos.z * me.pos.z).sqrt() / arena_radius,
        self_ground_height: self_ground,
        self_slope_forward: ahead - self_ground,
        self_slope_right: right_h - self_ground,
        self_hurt: me.hurt_fraction(),
        self_held: me.held.index() as f32,
        self_food: me.food,
        self_sneaking: if me.sneaking { 1.0 } else { 0.0 },
        self_absorption: me.absorption,
        self_eating: me.eat_fraction(),
        self_bow_draw: (me.bow_draw / c.bow_max_draw_seconds).clamp(0.0, 1.0),
        self_burning: if me.burn_time_left > 0.0 { 1.0 } else { 0.0 },
        self_shield_disabled: (me.shield_disabled_time / c.axe_shield_disable_seconds).clamp(0.0, 1.0),
        self_arrows: me.arrows as f32,
        self_slot: me.slot as f32,
        self_swap_lockout: (me.swap_lockout / swap_lockout_max).clamp(0.0, 1.0),
        self_mining: me.mining_progress.clamp(0.0, 1.0),
        self_effects: me.effects.levels().to_vec(),
        inventory,
        hotbar,
        enemies: fill(&enemies, cfg().max_observed_enemies),
        teammates: fill(&mates, cfg().max_observed_teammates),
        projectiles: proj_blocks,
        block_view,
        time_left: arena.time_left,
        enemies_alive: enemies.len() as f32,
        teammates_alive: mates.len() as f32,
        reward: ev.reward(),
        damage_dealt: ev.damage_dealt,
        damage_taken: ev.damage_taken,
        swept: ev.swept,
        won: ev.won,
        lost: ev.lost,
        done: ev.done,
    }
}
