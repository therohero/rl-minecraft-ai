# `training/sim/` - high-speed PvP simulation backend

A headless Rust reimplementation of just enough **vanilla** Minecraft PvP
to train a policy against:

- vanilla per-tick movement (gravity/drag/accel equilibrium, sprint
  speed, sprint-jump) resolved by a **vanilla per-axis AABB collision**
  against a **voxel world** (`collision.rs`): the terrain heightfield is
  quantized to integer block columns, the move is clipped Y-then-larger-
  horizontal-then-smaller against real block boxes, and the 0.6-block
  `maxUpStep` auto step-up applies (so a *full* block still needs a jump).
  **Crouching** (`sneak` action: ~0.3x speed, a 1.5-block hitbox, and it
  won't let you walk off a block/ledge edge), and vanilla **swimming**
  (buoyant fall, 0.8 drag on all axes, hold jump to swim up) plus
  **flowing-fluid push** (0.014 water / 0.007 lava per tick along the flow),
- a **hunger + saturation** economy: sprinting / jumping / landing an
  attack drain exhaustion, which spends saturation then food; you can't
  start sprinting below food 6; optional hunger-gated natural regen
  (`--natural-regen`),
- **raycast** melee hit detection - the attacker's eye ray is intersected
  against each candidate's hitbox and the nearest within reach is struck
  (`combat.rs` / `physics::Aabb::ray_intersect`), not a facing cone;
  **terrain and placed blocks break the line of sight** (a voxel DDA march,
  `blocks::ray_blocked`). This is the **two-step exchange vanilla runs**:
  the client picks the target off its latency-stale view (that raycast,
  against `delayed_hitbox`, bounded by `attack_reach` = 3.0), then the
  **server re-validates reach** against where the target actually is now
  (`server_interact_range` = 6.0, `AABB.distanceToSqr` in
  `handleInteract`). A target that dodges out of range during the
  round-trip takes nothing, even though the laggy attacker's screen showed
  a clean hit - only the swing (and its attack-cooldown reset) happened,
- vanilla's attack-charge damage scaling, critical hits (falling, not in
  water, not sprinting), sprint-hit knockback - **a landed sprint-hit
  ends your sprint**, so you re-press it for the next one (why W-tapping
  works) - and the modern **deterministic armour + Protection** formula
  (the pre-1.9 `nextFloat()*0.5+0.5` EPF roll is gone; fall / magic damage
  bypasses armour points but Protection's EPF still applies),
- the **vanilla 20-tick (1.0 s) hurt-invulnerability window**: only in its
  *first half* does a bigger follow-up land (and just the difference,
  without refreshing the timer); past halfway the next hit lands in full
  and resets the window - so focus-firing one target has sharply
  diminishing returns,
- **sweep attacks** with a real area of effect - **swords only** (vanilla),
  and only when near-stationary,
- **shields** (off-hand, held via `use_item`): a raised shield blocks all
  damage - melee *and* arrows - from within a 180 deg frontal arc and
  negates knockback, at the cost of sneak-speed movement. It does **not**
  interrupt the sword's attack-charge. An **axe hit disables a shield for
  5 s**.
- **kits** (`--kit sword|axe|uhc`): the classic sword loadout (diamond
  sword + full diamond armour, **no shield**); an axe kit adding a shield,
  a diamond axe, a bow, a crossbow and 6 arrows; and the full Ultra
  Hardcore kit - a shield, Sharpness/Power/Piercing gear, an **Efficiency 3
  diamond pickaxe** (mines placed blocks - see below), golden apples &
  heads, placeable planks/cobweb/water/lava, and (inherently) no natural
  regen. Weapons, enchants and armour come from `kit.rs`.
- **splash potions** (`effects.rs`, `--config splash_potions`): a plain
  `use_item` **throws** one - it arcs under gravity and breaks on the first
  solid/entity contact, applying a status effect in a `splash_radius` cloud
  with linear distance falloff. **Sneak + hold `use_item`** for
  `potion_drink_seconds` (~1.6 s) instead **drinks** it (self-only, full
  strength, `drink_duration_multiplier`x the splash duration; movement slows
  and `self_eating` fills, like the golden apple). Five types - Healing,
  Harming, Poison, Speed, Strength - driving the vanilla effect table
  (Speed/Slowness move ±%, Strength/Weakness flat melee ±, Regeneration/Poison
  HP-over-time, Instant Health/Damage, Fire Resistance). **No kit carries
  potions by default** - a config's `splash_potions: { poison: 2, ... }`
  hands them to both teams.
- **a 9-slot physical hotbar** (`kit.rs::HOTBAR_SLOTS`): the held-item
  action (`HOTBAR_ACTION_DIM` = 9 + `ITEM_COUNT` classes) either selects a
  *slot* (key 1-9) or **hotkeys** an owned item into the selected slot (the
  vanilla number-key swap - so every kit item is reachable past 9 slots;
  a hotkey'd item that's displaced stays in the inventory). The policy
  observes the layout + its selected slot. Selecting an empty slot is a
  real empty-hand state, and any hotbar op cancels a raised shield / bow
  draw / crossbow load / eat.
- **attribute swapping** (the vanilla MC-28289 bug, on by default under
  `--input-order legacy` = the 1.21.11 `hotbar->attack->use` order):
  switching the held slot on the same tick as an attack resolves the hit
  with the *previous* item's damage / attack-speed / on-hit traits - so an
  axe->sword swap lands a full-damage sword hit that also disables the
  shield. `--no-attribute-swapping` for the patched behaviour.
- **the hotbar "tick lockout"** (`--input-order modern` = the 26.2-pre-2
  `attack->use->hotbar` order): attribute swapping is impossible and every
  hotbar swap costs a one-tick (`combat.swap_lockout_seconds`) attack/use
  lockout - the swap-heavy techniques Marlowww's hotbar-order video covers.
- **bow & crossbow projectiles**: real arrow/bolt entities with
  gravity + drag, per-tick swept-ray collision, and vanilla launch -
  the aim gets a per-axis `triangle(0, 0.0172275)` deviation, the shooter's
  own velocity is added (vertical only while airborne), the bow uses
  vanilla's convex `(t^2+2t)/3` charge curve, and impact damage is
  `ceil(speed_at_impact * 2)` (so a drag-slowed long shot hits softer),
  plus Power and Piercing. Nearby in-flight projectiles are in the
  observation, aged by the observer's ping.
- **a sparse block grid** (`blocks.rs`) sharing one voxel space with the
  terrain. Two things go in it:

  - **placed planks and cobweb** - **one block per click**, into the empty
    cell against the face the eye-ray voxel march hits within `place_reach`
    4.5. Looking at nothing places nothing (the item is kept); you can't
    place into a player.
  - **flowing water and lava** poured from a bucket. Water spreads 7
    blocks, overworld lava 3 (level-step 2). Each fluid re-floods on its
    own vanilla schedule - water every 5 ticks, lava every 30. Destroy a
    source and the flow **drains from the middle outward**: the level-1
    ring next to the source empties on the first fluid tick, level 2 on the
    next, and so on - vanilla, not edge-first.

  A flowing fluid **pushes** an overlapping player along its flow
  direction. Lava meeting water generates **obsidian** (from a lava
  source), **stone** (water directly above/below a lava flow) or
  **cobblestone** (otherwise). Solid blocks - terrain *and* placed - block
  movement, bridging and attack/arrow rays; fluid and cobweb effects apply
  on **any hitbox overlap**. The policy sees a yaw-rotated
  `block_view_size` x `block_view_size` column view (integer surface top +
  water / lava / cobweb flags), plus its inventory counts and the kit's
  slot->item hotbar layout.

  The fluid reflood is tuned to be cheap: it re-floods only the kind whose
  schedule fires that tick (the other kind's cells act as flood barriers),
  reuses its BFS buffers across ticks so a steady state allocates nothing,
  and skips the whole pass when no source is live.
- **mining** (`uhc` only) - hold `attack` with a **pickaxe or axe** in hand
  and, while the eye ray is on a **placed** block within `place_reach`, a
  break timer fills; at 100% the block is removed. Break time is
  `block_hardness * mine_seconds_per_hardness / speed`, where hardness is a
  fixed table (planks 2, stone 1.5, cobblestone 2, obsidian 50, cobweb 0.8)
  and `speed` is:
  - the **correct tool** (axe on planks, pickaxe on stone / cobblestone /
    obsidian): `mine_correct_tool_speed + efficiency * mine_efficiency_speed_per_level`
    (`uhc` Efficiency 3 -> speed 17: a plank falls to the axe in ~3 ticks,
    stone to the pickaxe in ~2);
  - the **wrong tool** (a pickaxe on planks, an axe on stone): `1.0`, and
    Efficiency doesn't apply - so it's ~15x slower (a plank takes the
    pickaxe ~2.5 s).

  Aiming elsewhere or releasing `attack` resets the progress. The **terrain
  floor and the arena wall are never mineable** - `raycast_mine_target`
  stops the ray at them and returns nothing. A swing at anything that isn't
  a block is still a normal melee hit. The policy sees the break progress
  as `self_mining` (0..1).
- **golden apple / head**: a timed `use_item` "eat" (slowed, can't attack).
  Vanilla-like - **no instant heal**: it grants absorption and food
  immediately and the HP comes back over time via Regeneration (the golden
  head heals at twice the apple's rate).
- fall damage - a **cobweb** arrests the fall and negates it.
- **enchants** (`--config enchants`, all off by default): **Fire Aspect**
  ignites a melee target, **Flame** ignites an arrow target, **Punch** adds
  arrow knockback, **Knockback** adds melee knockback, and **Knockback
  Resistance** (an armour attribute, 0..1) scales *all* incoming knockback
  down. Sharpness / Power / Piercing are on the `uhc` kit already.
- a **circular arena** with optional rolling terrain and a velocity-killing
  rim (no edge to fall off).
- **UHC perimeter wall** - the `uhc` kit rings the arena rim with a solid
  wall `arena_wall_height` (3) blocks tall (`arena_walls`: forced on for
  `uhc`, off for every other kit). It gives the policy a surface to pin an
  opponent against and web their head, and a hard stop on a jump-out. It's
  implicit static geometry like the terrain ground - no block-map cost -
  and collision, line-of-sight and placement all see it.
- **team fights**: `--team-size N` puts two teams of N on the platform
  (1v1 is just `N = 1`); friendly fire is on by default.

It is **fully uncoupled from wall-clock time**: there is no tick sleep
anywhere, so it steps every arena as fast as the trainer can send actions
and consume states.

## Build & run

```bash
cargo build --release
./target/release/mc_pvp_sim --port 9999 --arenas 64
```

| flag | default | meaning |
|---|---|---|
| `-p, --port <PORT>` | 9999 | UDP port to listen on |
| `-n, --arenas <N>` | 64 | parallel arenas (each has 2 players) |
| `-s, --seed <SEED>` | random | base RNG seed; logged on startup so any run is reproducible |
| `-t, --team-size <N>` | 1 | players per team (1 = 1v1 duel, 2 = 2v2, ...) |
| `-k, --kit <KIT>` | sword | combat kit: `sword` \| `axe` \| `uhc` |
| `-c, --config <FILE>` | built-in | JSON file of `SimConfig` overrides (partial files are fine) |
| `--dump-config` | | print the fully-resolved config as JSON and exit |

The bare positional form `mc_pvp_sim 9999 64 <seed>` still works. Normally
you don't run this by hand - `training/python/train.py` (or `run.sh`) launches it
for you and forwards `--port` / `--arenas` / `--seed` / `--config` (with
`--team-size` / `--kit` folded into the config file it generates).

`cargo test` runs the unit-test suite. It covers, among others:

- **combat** - raycast hit detection, line-of-sight blocked by a hill, the
  1.0 s i-frame window's two phases, deterministic armour / EPF-only fall
  damage, sword-only sweep, shield blocking + axe shield-disable,
  attribute-swap trait carry-over, attack-charge, a laggy swing landing on
  a stale position but whiffing when the target has since fled past
  `server_interact_range`.
- **projectiles** - vanilla arrow launch (spread, velocity inheritance,
  impact-speed damage).
- **movement / world** - voxel collision (step-up, head-bonk, no tunnelling
  at terminal velocity, de-penetration), sprint speed, gravity, every
  random terrain is walkable, wall clamping.
- **blocks** - water/lava spread, source-first drain, lava-meets-water
  stone generation, sliver-overlap fluid contact, one-block placement, the
  UHC rim wall (a 3-tall ring riding the terrain surface), the pickaxe
  mining stone but never the terrain floor, the axe breaking planks far
  faster than the pickaxe (right tool vs wrong), mining progress resetting
  when `attack` is released.
- **misc** - config parsing, golden-apple heal/absorption, cobweb slow,
  laggy observation of others, 1v1 and 2v2 wins/losses/draws, friendly-fire
  penalty, instant reset, determinism.

## Transport: binary over UDP

The sim talks to exactly one trainer at a time over UDP with a small
binary framing (full spec in `src/protocol.rs`). Every env step is one
request/response:

```
trainer -> sim :  MSG_ACTION   payload = num_slots * 10                   f32  (LE)
sim -> trainer :  MSG_STATE    payload = num_slots * obs_floats_per_slot  f32  (LE)
```

An action row is `[move_x, move_z, yaw_delta, pitch_delta, jump, attack,
sprint, use_item, sneak, held_slot]`. `held_slot` `0..HOTBAR_ACTION_DIM`: `0..9`
selects a physical hotbar slot, `9..21` hotkeys `kit::Item` id
`(held_slot - 9)` into the selected slot. `num_slots = num_arenas *
players_per_arena` (`players_per_arena =
2 * team_size`), and `obs_floats_per_slot` depends on how many
enemies / teammates / projectiles each observation describes and the
`block_view_size` - all sent in the `Hello` so the trainer never hardcodes
them. Slots come out in team order per arena: team 0's members, then team 1's.

- **Why UDP, not TCP/JSON.** At a few hundred arenas, JSON-encoding a
  batch of thousands of small floats every step was a measurable slice of
  the training loop, and TCP's ack/Nagle/head-of-line handling buys
  nothing on loopback. The payloads are now flat `f32` arrays that each
  side reads directly into a numpy buffer / `&[f32]` with no per-element
  parsing.
- **Framing.** A 10-byte header (`type`, `wire version`, `u32 seq`,
  `u16 frag_idx`, `u16 frag_count`) then a payload fragment. A logical
  message over 60 KB (state batches past ~280 arenas) is split across
  several datagrams sharing one `seq` and reassembled by index.
- **Loss tolerance.** The exchange is strictly lockstep. The sim caches
  the datagrams of its last reply; if it sees the same `seq` again (client
  retransmit after a dropped datagram) it **replays the cached reply
  instead of stepping the arenas again**, so a rare loopback drop can't
  desync a run. Out-of-order datagrams from an older step are dropped.
  Both ends request a 16 MB socket buffer (`socket2` on the sim side) so a
  multi-fragment batch doesn't overflow it; the kernel clamps to
  `net.core.rmem_max` / `wmem_max`, and at very high arena counts on a box
  where those are small you may still see `retransmitting step N` in the
  trainer log - raise them with `sudo sysctl -w net.core.rmem_max=16777216`.
- **Handshake.** On the first `MSG_HELLO_REQ` the sim (re)initializes its
  arenas and replies with `MSG_HELLO_RESP` - a JSON `Hello` carrying
  `num_arenas`, `players_per_arena`, `team_size`, `kit`, `item_count`
  (the `inventory` block width), `hotbar_slots` (the `hotbar` obs block
  width), `hotbar_action_dim` (the held-slot action head width),
  `effect_count` (the `self_effects` block width),
  `obs_floats_per_slot`, `tick_dt`, every observation-normalization
  constant, and the full resolved `SimConfig`, so the Python side never
  hand-copies a value.

Changing any field of `Observation` / `Action` means: bump `WIRE_VERSION`
in `src/protocol.rs` **and** `training/python/env.py`, and update
`Observation::write_wire` alongside `training/python/features.py::wire_batch_to_obs`
(and `obs_floats_per_slot` / `features.wire_floats_per_slot()` if the row
width formula changed).

## One tick, in order (`arena.rs::step`)

Every arena advances by calling into the other modules in a fixed order.
Getting the order right is what makes the vanilla mechanics compose
correctly:

1. **Ping jitter** - roll this tick's reported latency around the
   per-match `base_ping_ms` baseline. Your own input has *zero* delay
   (vanilla client-authoritative movement); only your *view* of others lags.
2. **I-frame decay** - tick down every player's `hurt_time_left`.
3. **Per-player input + physics** (living players only):
   `player::apply_input` (look clamp, hotbar select/hotkey, item use -
   shield / bow / crossbow / eat / place - then movement intent), then any
   queued shot/placement is spawned, then `player::integrate` (gravity or
   buoyant swim → `collision::move_with_collision` against the voxel world
   with the 0.6 step-up and sneak edge back-off → `on_ground` / velocity /
   fall-damage derivation → arena-rim clamp → sprint hunger cost), then
   `apply_block_effects` (cobweb/water/lava, fluid push, golden-apple regen,
   hunger-gated natural regen or starvation).
4. **Player-player push-apart**, then de-penetrate anyone the push shoved
   into a wall.
5. **Left-click resolution** - for each player holding `attack` and not
   eating / mid-swap: if a pickaxe/axe is aimed at a placed block, advance
   mining (`mine_step`) and skip the swing; otherwise `combat::resolve_melee`
   (eye-ray target pick against each candidate's *ping-delayed* hitbox →
   server-side reach re-check against their *current* position → damage +
   knockback + sprint-reset + attack exhaustion + sweep AoE).
6. **Projectiles** - `projectile::step_all`: ballistic flight, swept-ray
   collision against players and blocks, damage through armour/shield/
   i-frames, Piercing.
7. **Block grid tick** - decay placed blocks/sources; every few ticks
   recompute fluid flow and resolve lava/water contacts.
8. **Match-end decision** - a whole team down, or a timeout on total team
   HP + absorption → set every slot's `done` / `won` / `lost`.
9. **Snapshot** this tick's public state into every player's / projectile's
   history ring (so laggy observers can read it back next tick).
10. **Build observations** (`observation::build` per slot), then if the
    match ended, `randomize_spawns` re-rolls terrain + spawns *in place* -
    the "instant reset" the training loop needs.

## Observation layout (wire row, `WIRE_VERSION 9`)

`Observation::write_wire` emits a flat `f32` row in exactly this order;
`python/features.py::wire_batch_to_obs` decodes it 1:1. Widths marked `×k`
repeat for the nearest `k` (short lists zero-padded; `present` flag first).

| block | floats | contents |
|---|---|---|
| self | 27 | hp, vel xyz, yaw, pitch, on_ground, attack_cooldown, ping_ms, shield, dist_from_center, ground_height, slope_forward, slope_right, hurt, held, absorption, eating, bow_draw, burning, shield_disabled, arrows, slot, swap_lockout, food, sneaking, mining |
| self_effects | `EFFECT_COUNT` = 9 | `amplifier + 1` while active else 0, per `effects.rs::Effect`: speed, slowness, strength, weakness, regeneration, poison, instant_health, instant_damage, fire_resistance (the two instants never persist ⇒ always 0) |
| inventory | `ITEM_COUNT-1` = 16 | per-item counts (sword…golden_head, then the 5 splash potions), clamped + normalized |
| hotbar | `HOTBAR_SLOTS` = 9 | the kit's slot→item id layout |
| enemies | 13 × `max_observed_enemies` | present, hp+absorption, rel xyz (yaw-rotated), vel xyz, ground_height, blocking, eating, held_ranged, sneaking |
| teammates | 13 × `max_observed_teammates` | same shape as an enemy block |
| projectiles | 7 × `max_observed_projectiles` | present, rel xyz, vel xyz |
| block view | 4 × `block_view_size²` | per column: top_rel (normalized), water, lava, cobweb - row-major, front-left first |
| globals | 3 | time_left, enemies_alive, teammates_alive |
| events | 7 | reward, damage_dealt, damage_taken, swept, won, lost, done - **dropped** before the policy sees it; the trainer reads reward/done/won/lost off these columns directly |

The policy input is slightly wider than the wire row: `features.py` expands
`self_yaw` / `self_pitch` each into a `(sin, cos)` pair (+2) and drops the 7
event floats.

## Where things live

| file | responsibility |
|---|---|
| `src/main.rs` | CLI parsing, config load, logging setup |
| `src/server.rs` | the UDP server: framing, fragment reassembly, seq-dedup, rayon-parallel arena stepping |
| `src/protocol.rs` | wire types, framing constants, `Action::from_wire` / `Observation::write_wire`, `obs_floats_per_slot` |
| `src/arena.rs` | one match: holds the players / projectiles / block grid, runs the tick (`step`) by calling into the modules below, decides match end + instant reset, dispatches block placement and pickaxe mining (`mine_step`) |
| `src/player.rs` | the `Player` entity and its whole per-tick self-update: input (look / hotbar / item use), physics integration (`integrate` - gravity/swim, then `collision::move_with_collision`; the pre-move block contact is sampled once by `arena` and passed in), the hunger / saturation / exhaustion economy, environmental effects (`apply_block_effects`, incl. fluid push), consumables, damage application, spawn / respawn, the per-tick public-state snapshot ring that laggy observers read |
| `src/collision.rs` | vanilla per-axis AABB move against the voxel world: swept-box candidate gather (`solid_boxes`), Y/X/Z clip, the 0.6 `maxUpStep` step-up, the sneak edge back-off, and `push_out_of_solids` |
| `src/combat.rs` | melee: `resolve_melee` (eye-ray target pick against the attacker's ping-delayed view of others, then a server-side reach re-check against the target's current position, damage + knockback, sprint-reset, attack exhaustion, sweep AoE), per-weapon stats, attack-charge scaling, crit (falling / not-in-water) / sweep (near-stationary) / sprint classification, shield block (arc), axe disable, i-frame / last-damage rule |
| `src/projectile.rs` | in-flight arrows / bolts **and thrown splash potions**: spawning, ballistic flight, swept-ray collision; arrows deal damage through armour / shield / i-frames + Piercing + knockback, potions break and apply their effect in a `splash_radius` cloud with distance falloff |
| `src/effects.rs` | the status-effect table on each `Player` (`StatusEffects`): the 9 `Effect`s, vanilla stacking rules, per-tick Regen/Poison, the move-speed and melee-damage modifiers, and the splash-potion → effect mapping |
| `src/observation.rs` | `Arena` state -> the wire `Observation` one slot sees (self-relative, nearest few others / arrows, inventory + hotbar, block-grid view, reward + event flags) - the mirror of `training/python/features.py` |
| `src/kit.rs` | the `Item` enum, `HOTBAR_SLOTS` / `HOTBAR_ACTION_DIM`, and the per-kit loadout (items + counts, the 9-slot hotbar layout, enchant levels, armour) |
| `src/blocks.rs` | the sparse block grid over the voxel terrain: the unified `is_solid_cell` predicate (terrain ground + placed solids + the implicit `uhc` rim wall), `Cell` (stored `(x,z,y)`-ordered), voxel-DDA ray-blocking + face-offset placement (`raycast_place_target`) + pickaxe mine targeting (`raycast_mine_target` - placed blocks only, never floor/wall), water/lava flow on per-fluid schedules (per-kind selective reflood, reused scratch, skipped when sourceless) + **source-first** drain, flow-vector push, lava+water -> stone/cobblestone/obsidian, fluid/cobweb overlap, `support_y`, the column-view observation |
| `src/physics.rs` | vanilla per-tick movement constants (`MAX_UP_STEP`, `COLLISION_EPSILON`, ...), `Vec3` / `Aabb` (ray/box + segment intersection, `union` / `translated`), `pair_mut` |
| `src/terrain.rs` | rolling terrain quantized to integer block columns (`column_top`) - 5 shapes x random params (capped so no neighbour column steps >1 block), seeded per arena |
| `src/config.rs` | `SimConfig` / `Kit` - every runtime-tunable knob, with defaults matching vanilla where it matters |

## Configuration

`SimConfig` (see `src/config.rs`) is the one place every runtime knob
lives. Pass a **partial** JSON file with `--config` (omitted keys keep
their default), and run `--dump-config` or read `config.example.json` for
the full list. Top-level fields, by area:

| area | fields |
|---|---|
| **match setup** | `kit`, `team_size`, `friendly_fire`, `match_time_seconds`, `natural_regen` |
| **input model** | `attribute_swapping`; `input_order` - `legacy` (1.21.11 `hotbar->attack->use`, the default) or `modern` (26.2-pre-2 `attack->use->hotbar`, which forces `attribute_swapping` off and adds a `combat.swap_lockout_seconds` cost per hotbar switch) |
| **arena** | `arena_radius`, `terrain_max_amplitude`, `terrain_flat_only`, and the rim wall `arena_walls` / `arena_wall_height` (forced on for the `uhc` kit) |
| **observation** | `max_observed_enemies` / `_teammates` / `_projectiles`, `block_view_size` |
| **splash potions** | `splash_potions: { healing, harming, poison, speed, strength }` - per-player starting counts, all 0 by default (no kit carries potions) |
| **enchants** | `enchants: { fire_aspect, flame, punch, knockback, knockback_resistance }` - per-player levels (0..1 for resistance), all 0 by default |
| **latency (domain randomization)** | `min_ping_ms` / `max_ping_ms`, `ping_jitter_ms` |
| **regularizer** | `max_look_delta` (the per-tick crosshair clamp) |
| **reward weights** | the `reward` block: `per_hp_dealt` / `per_hp_taken` / `win` / `loss` / `sweep_penalty` / `friendly_fire_penalty`, plus opt-in shaping (all 0 = off): `approach_per_block` and `aim_bonus` (potential-based - reward is the per-tick change of `-approach·dist + aim·cos(aim error)` vs the nearest living enemy, so they telescope and can't be farmed) and `draw_penalty` (subtracted when a match ends with equal team HP). Shaping changes only the scalar `reward`; the observation layout and `WIRE_VERSION` are unchanged |

The `combat` block then holds the fine-grained numbers, grouped roughly as:

- **melee** - reach, hitbox inflation, the i-frame window, attack-charge
  timing, crit / sweep / sprint-knockback numbers.
- **shield** - block amount + arc, the axe shield-disable, `swap_lockout_seconds`.
- **weapons** - axe / bow / crossbow damage and timing, Sharpness / Power
  scaling, and mining (`mine_seconds_per_hardness`, `mine_correct_tool_speed`,
  `mine_efficiency_speed_per_level`).
- **consumables** - golden-apple / head absorption, regen and food.
- **splash potions** - `splash_potion_speed` / `_gravity` / `_drag`,
  `splash_radius`, the per-effect `potion_*_seconds` / `potion_*_amplifier`,
  `instant_health_hp` / `instant_damage_hp`, and the drink knobs
  `potion_drink_seconds` / `drink_duration_multiplier`.
- **enchants** - `fire_aspect_seconds_per_level`, `flame_seconds`,
  `punch_knockback_per_level`, `knockback_enchant_per_level`.
- **movement economy** - crouch (`sneak_*`), hunger (`*_exhaustion`,
  `min_food_to_sprint`, `*_regen_seconds`), swimming (`swim_*`).
- **block grid** - `place_reach`, `water_tick_ticks` / `lava_tick_ticks`,
  the `*_max_level` / `lava_level_step` spread, `*_push_per_tick`,
  `block_ceiling`, `max_blocks`, block/source lifetimes, cobweb/water slow
  factors, lava damage.

The vanilla per-tick physics constants and the armour-reduction formula
are deliberately **not** configurable.

## Deliberately not vanilla

A few things stay a conscious simplification or a training-only choice:

- **`max_look_delta` (3.0 rad/tick)** - vanilla has no per-tick limit on
  the crosshair. This cap keeps the learned aim humanly reproducible
  against a live server; it also rescales the yaw/pitch action head, so a
  checkpoint doesn't transfer across values.
- **Self-input has zero latency** - your ping ages your *view of others*
  and the *target pick* for your melee swing (against the target's
  round-trip-old position, like vanilla client-side hit registration), but
  your own movement / aim apply instantly (vanilla client-authoritative
  movement). The server *does* re-check melee reach against everyone's
  current position (`server_interact_range`), matching vanilla's
  `handleInteract`, but the sim still doesn't model the server rejecting a
  laggy player's *movement*.
- **Circular arena with a velocity-killing rim** - there is no edge to fall
  off, and the match resets instantly on a win.
- **A voxel world of full blocks only** - no slabs / stairs, so the 0.6
  step-up never fires on terrain (every 1-block step needs a jump), and the
  terrain observation fields are integer block tops.
- **Splash potions (thrown *and* drunk) and the Fire Aspect / Flame /
  Punch / Knockback / Knockback Resistance enchants are modelled; there's
  no *separate* drink-potion item and no rest of the enchant table
  (Sharpness/Power/Piercing aside)** - and no kit carries potions or these
  enchants unless a config opts in (`splash_potions`, `enchants`).
- **The golden head** is a UHC-server item, not vanilla; it lives only in
  the `uhc` kit.
