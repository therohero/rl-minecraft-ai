# `mod/` - Fabric client mod (`/fight`)

A client-side Fabric mod (Minecraft 1.21.11, Java 21) that lets you drive the
trained RL policy from your own Minecraft client with a chat command, and
record live fights as a training dataset.

It is the client-mod counterpart of [`azalea-bot/`](../azalea-bot/README.md):
same trained policy, same `inference_server.py` HTTP seam, but running inside
a real vanilla client instead of a headless `azalea` bot. Use it only on
servers you are authorised to run bots on - it takes over movement, aim and
attacks while active.

## Commands

| command | what it does |
|---|---|
| `/fight` | Take over, driven by the trained policy served at `inference_url` (`azalea-bot/inference_server.py`). **Passive by default** - it won't attack anyone until they hit you or you `/fight target` them. |
| `/fight train` | Same, but also **records `(observation, action)` per tick** to a JSONL dataset (one file per fight), tagged with the **kit** auto-detected from your inventory. `/fight train practice` tags the data as practice rather than a real match. |
| `/fight stop` | Hand control back to you (also happens automatically on disconnect). |
| `/fight status` | Show mode / kit / target lock / pause state / episode number / inference server. |
| `/fight stopfightingtoggle` | Toggle whether dying hands control back (default) or just rolls the recorder to the next episode and keeps going. |
| `/fight target <spec>` | Set who the bot may fight. `<spec>` is one of: `passive` / `clear` (fight back only), `<player name>`, `look` (whoever's under your crosshair), `nearest` (pin the closest player now), `last` (whoever last hit you), `region` (run twice at opposite corners - engage anyone inside the box), `auto` (continuous nearest - **only** where you're allowed to). Named / pinned / crosshair locks never auto-reacquire when the target dies or leaves. |

### Server safety

`/fight` on a real server defaults to **passive**: it drives movement but
attacks no one until a player damages you (then it locks that player) or you
name a target. It also **auto-pauses** - stops sending inputs *and* recording
- whenever a GUI is open or you're a spectator, resets on a dimension change,
and cuts a fresh recorded episode on every death / kill so the offline
trainer sees clean per-fight returns. Death detection covers the case where a
server plugin cancels the vanilla death (HP restored + a teleport / dimension
swap / forced spectator). Still: only use this where you are authorised to.

## HUD overlay

While a fight is active a small panel is drawn top-left: mode (`RL FIGHT` /
`RL TRAIN`), the detected kit, the current target (`name <dist>m`), the
inference-server round-trip (`<n>ms`, colour-coded green/yellow/red; or
`connecting…` / `unreachable`), and - in `train` mode - the recorded tick
count. It reads live `FightController` state and drives nothing. Hidden when
idle, when the vanilla HUD is off (F1), or while the F3 debug screen is up.
Turn it off with `hud_enabled = false`.
`InferenceClient` tracks the `POST /act` wall-clock round-trip as an EWMA
(`latencyMs()`), which both the HUD and `/fight status` read.

## Kit detection

`/fight` and `/fight train` classify your inventory as the **nearest** of the
trainer's kits and use that kit's policy behaviour / dataset label:

- **`uhc`** - pickaxe / planks / cobweb / water bucket / golden apple present
- **`axe`** - sword + axe + bow/crossbow, without the UHC utility items
- **`sword`** - basically just a sword

Nothing recognisable in the inventory ⇒ **`sword`** (the default). The scoring
is in [`Kit.java`](src/client/java/rl/minecraft/ai/client/combat/Kit.java) and
mirrors `training/sim/src/kit.rs` / `training/python/export_model.py`.

## What `/fight train` writes

```
<dataset dir>/<kit>/session-<timestamp>-e<N>.jsonl   # one fight: one JSON object per tick {t, kit, target, obs, action}
<dataset dir>/manifest.jsonl                          # one line per finished episode
```

One `/fight train` session rotates through `-e0`, `-e1`, ... - a new file
each death or kill - so every file is exactly one fight. `obs` is the
**raw, un-normalised** observation dict, the same shape
`training/python/features.py::observation_to_row` consumes. The dataset dir
defaults to `<game dir>/rl-datasets` (override in config).

The **last** line of each file is an outcome record with no `obs` key:
`{t, outcome, reason, self_hp_end, enemy_hp_end, opponent, server, match}`.
`outcome` is `win` (target down while we were alive), `loss` (we died), or
`unknown` (ended by hand / disengaged / dimension change).
`train_from_episodes.py` uses it for the terminal win/loss reward and the
final-tick HP deltas instead of guessing from the observation stream; the
same fields (plus `episode`, `ticks`, timestamps) are copied into the
`manifest.jsonl` line. Older datasets without the record still fall back to
the guess.

`/fight train` only **collects data** - it does not change the model while
you play. The message it prints on start/stop points at the file and the
command below.

### Closing the loop (offline fine-tune)

```bash
./run_train_mod.sh [dataset_dir]        # -> updates training/checkpoints/latest.pt
./run_bot_mod.sh                        # re-export + serve the updated model
```

`run_train_mod.sh` runs `training/python/train_from_episodes.py`: it
reconstructs a per-tick reward from the observation stream (damage dealt −
damage taken, ±win/loss, mirroring `training/sim/src/arena.rs`), then does a
few epochs of **advantage-weighted regression** on top of the current
checkpoint - a gentle nudge toward what worked in *real* fights, on
*real*-server observations. The previous checkpoint is kept as
`latest.pt.pre_finetune`. Pass `-- --dry-run` first to see the episode
stats. Real-server data is scarce and off the sim's distribution, so this
is domain adaptation, not a replacement for `./run.sh` self-play.

`run_train_mod.sh` defaults `dataset_dir` to `training/datasets/`; set the
mod's `dataset_dir` config to `<repo>/training/datasets` (or pass the path)
so it lines up with where `/fight train` writes.

## How it works internally

`RlMinecraftAiClient` registers the `/fight` command family and a
`START_CLIENT_TICK` handler that runs `FightController.onClientTick` right
before vanilla polls input, so the keybinding state the mod sets is picked
up the same tick.

Each tick, while a fight is active (and not paused - a GUI open / spectator
mode / a dimension change all pause the loop and stop recording):

1. `TargetLock.resolve` picks the target: `PASSIVE` (default) returns only
   whoever last damaged us (`FightController` tracks that off `hurtTime` +
   `getAttacker()`), the other kinds resolve a named / pinned / in-region /
   nearest player. `DeathWatch` checks for a death - vanilla (HP 0 / respawn)
   or plugin-cancelled (HP restored the same tick as a teleport / dimension
   swap / forced spectator) - and a target that just went untargetable
   because it died is a kill; either one closes the current recorded episode
   and opens the next.
2. The reconstructed self-timers advance from the state about to be
   observed: `bowDrawTicks` (ticks holding right-click with a bow),
   `ticksSinceSwap` (0 the tick the selected slot changed), which become
   the `self_bow_draw` / `self_swap_lockout` observation fields the
   1.21.11 client can't report directly.
3. `ObservationBuilder.build` produces the **raw, un-normalised**
   observation `JsonObject` - the same shape
   `features.observation_to_row` consumes - normalising `top_rel` and the
   handful of fields the sim pre-normalises, and nothing else. It's a Java
   port of `azalea_bot`'s `build_observation`; keep the two in sync.
4. `InferenceClient.requestAsync` POSTs it off-thread; the freshest
   `Action` already returned is applied by `ActionApplier.apply`.
5. In `train` mode, `EpisodeRecorder` streams `{t, kit, target, obs,
   action}` to the current `session-*-e<N>.jsonl`; each death / kill /
   disengage / dimension change closes that file with a
   `{t, outcome, reason, self_hp_end, enemy_hp_end, opponent, server, match}`
   record and rolls to `-e<N+1>`.

`ActionApplier` is deliberately lighter than `azalea_bot`'s guard in the
places a real vanilla client already covers (jump-on-ground, the hunger cost
of sprinting, the attack cooldown). What it adds is a **virtual mouse** in
`ClientGuard`: the policy's per-tick yaw/pitch delta is treated as a desired
turn *rate*, delayed by `aim_latency_ticks` (reaction lag), low-pass
smoothed, chased by a modelled mouse velocity under an acceleration cap
(`max_yaw_accel_deg`) and a top-speed cap, given a sub-degree tremor, then
converted to a whole number of **mouse counts** through this client's exact
vanilla sensitivity curve (`(s*0.6+0.2)³·8`, then `·0.15`; `s` read from
your in-game setting or `mouse_sensitivity`) with the leftover fraction
carried over, and applied through `changeLookDirection` - the same path a
real mouse takes. So every rotation the server sees is an integer multiple
of the mouse-count quantum with bounded velocity/acceleration and per-tick
noise. On top of the real attack cooldown, `tryAttack` adds a jittered
minimum click gap and a `max_cps` ceiling. Hit legality is unchanged (reach,
a 14° facing cone, block line-of-sight, aim-settle). A buried kit item is
hotkeyed into the hotbar via the player's own always-open (`syncId 0`)
screen handler - the same `SWAP` click a vanilla client sends when you drag
an item onto a number key - rate-limited by `hotkey_swap_min_gap_ticks`.

`Spec` (max_hp, arena_radius, the combat-timing constants) is fetched from
`GET /spec` on a daemon thread at fight start and falls back to
`Spec.DEFAULT` (the sim defaults) until it arrives. A respawn hands the
client a fresh player entity, so the guard's low-pass and the
reconstructed timers are reset when the entity instance changes (and that
same instance swap is itself read as a death).

## Configuration

Optional `<game dir>/config/rl-minecraft-ai.properties` (or `-Drl.minecraft.ai.<key>=…`):

```properties
inference_url             = http://127.0.0.1:8800/act
dataset_dir               =                       # blank -> <game dir>/rl-datasets
max_yaw_deg_per_tick      = 80
max_pitch_deg_per_tick    = 60
reach                     = 3.0
rotation_smoothing        = 0.7    # 1.0 = no smoothing, lower = softer accel onto target
aim_jitter_deg            = 0.4    # per-tick gaussian "hand tremor" mixed into rotation
aim_settle_deg            = 50     # hold the attack one tick after a turn bigger than this
min_sneak_hold_ticks      = 3      # debounce: min ticks a sneak state is held before flipping
require_line_of_sight     = true   # don't attack through a wall even if in reach + facing cone
hotkey_swap_min_gap_ticks = 10     # min ticks between buried-item inventory swaps
hud_enabled               = true   # small on-screen mode/kit/target/latency readout while fighting
fight_through_death       = false  # keep fighting after death (also /fight stopfightingtoggle)
pause_on_screen           = true   # stop driving inputs + recording while a GUI is open
engage_range              = 0      # /fight target auto max distance (blocks); 0 = unlimited
death_teleport_blocks     = 8      # a 1-tick position jump this far reads as a death
death_hp_floor            = 4      # HP at/below this then instantly restored reads as a death
disengage_ticks           = 100    # end the episode after the target has been gone this many ticks
```

## Build & run

```bash
cd mod
./gradlew build          # -> build/libs/rl-minecraft-ai-<version>.jar
./gradlew runClient      # dev client with the mod loaded
```

Drop the jar (plus Fabric API and Fabric Loader) into `mods/` for a normal
install.

Before `/fight`, the policy server has to be running. From the repo root
(after training at least once with `./run.sh`):

```bash
./run_bot_mod.sh          # exports the latest checkpoint + serves on 127.0.0.1:8800
.\run_bot_mod.ps1         # native Windows
./run_bot_mod.sh 8801     # different port (set inference_url in the mod config to match)
```

(or by hand: `cd azalea-bot && python inference_server.py --model-dir ./model --port 8800`,
after exporting a checkpoint with `training/python/export_model.py`).

## Fidelity & limitations

The observation is a best-effort port of
`azalea-bot/azalea_bot/src/main.rs::build_observation` (itself a mirror of
`training/sim/src/observation.rs`) - keep the three in sync when the wire
format changes. A live server has no equivalent for some sim-only fields:

- `time_left` is pinned to the match length, `self_dist_from_center` uses the
  spec's arena radius - so the normalised feature at least lands in range.
- `self_ground_height` is the absolute surface Y, exactly as `azalea_bot`
  reports it - off-distribution on tall terrain but consistent between the two
  clients.

Everything else is read straight off live client state, including
`self_shield_disabled` (the real item-cooldown manager), `self_mining`
(the real interaction manager's block-break flag) and `self_effects` (the
player's live `StatusEffect` amplifiers) - no packet tracking needed here,
unlike `azalea_bot`'s headless `tracker.rs`. `KitItem` maps a live splash
potion to one of the 5 potion ids by its primary effect; the policy can't
actually throw one yet (see `TODO.md`).

This mod runs an actual vanilla client, so a lot of what `azalea_bot`'s
`guard.rs` has to fake (physics, hunger cost, attack cooldown) is simply
real here. [`ClientGuard`](src/client/java/rl/minecraft/ai/client/combat/ClientGuard.java)
covers what a real client can still give away: rotation goes through a
**virtual mouse** (reaction lag, low-pass, an acceleration + top-speed cap,
tremor, then quantised to whole mouse counts on this client's real
sensitivity curve and applied via `changeLookDirection`), clicks get a
jittered minimum gap and a `max_cps` ceiling on top of the real cooldown, an
attack requires a clear line of sight and holds fire for one tick right
after a big turn, and sneak can't be toggled faster than
`min_sneak_hold_ticks`. A buried kit item is juggled into the hotbar via a
`SWAP` click on the player's own always-open inventory screen handler,
rate-limited by `hotkey_swap_min_gap_ticks` - mirroring `azalea_bot`'s
open/click/close.
