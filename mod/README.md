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
| `/fight` | Take over and fight the **nearest player**, driven by the trained policy served at `inference_url` (`azalea-bot/inference_server.py`). |
| `/fight train` | Same, but also **records every `(observation, action)` pair** to a JSONL dataset, tagged with the **kit** auto-detected from your current inventory. |
| `/fight stop` | Hand control back to you (also happens automatically on disconnect). |
| `/fight status` | Show mode / detected kit / current target / whether the inference server is reachable (with its round-trip latency). |

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
<dataset dir>/<kit>/session-<timestamp>.jsonl   # one JSON object per tick: {t, kit, target, obs, action}
<dataset dir>/manifest.jsonl                     # one line per finished session
```

`obs` is the **raw, un-normalised** observation dict - the same shape
`training/python/features.py::observation_to_row` consumes. The dataset dir
defaults to `<game dir>/rl-datasets` (override in config).

The **last** line of each session file is instead an outcome record -
`{t, outcome, self_hp_end, enemy_hp_end, reason}` with no `obs` key, so the
trainer can tell it apart. `outcome` is `win` (the target went down while we
were alive), `loss` (we died), or `unknown` (the fight was ended by hand or
the target left range). `train_from_episodes.py` uses it for the terminal
win/loss reward and the final-tick HP deltas instead of guessing from the
observation stream; the same fields are copied into the `manifest.jsonl`
line. Older datasets without the record still fall back to the guess.

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

Each tick, while a fight is active:

1. `TargetSelector.nearest` picks the nearest other player.
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
   action}` as one JSONL line per tick, and on stop appends the
   `{t, outcome, self_hp_end, enemy_hp_end, reason}` record -
   `FightController` classifies the outcome from live target / self state.

`ActionApplier` is deliberately lighter than `azalea_bot`'s guard: a real
vanilla client already enforces jump-on-ground, the real hunger cost of
sprinting, and the real attack cooldown, so the mod only adds what a real
client can still fake - `ClientGuard` rotation naturalism (smoothing /
rate-clamp / sub-degree jitter / 0.15° grid snap) and hit legality (reach,
a 14° facing cone, block line-of-sight, aim-settle). A buried kit item is
hotkeyed into the hotbar via the player's own always-open (`syncId 0`)
screen handler - the same `SWAP` click a vanilla client sends when you drag
an item onto a number key - rate-limited by `hotkey_swap_min_gap_ticks`.

`Spec` (max_hp, arena_radius, the combat-timing constants) is fetched from
`GET /spec` on a daemon thread at fight start and falls back to
`Spec.DEFAULT` (the sim defaults) until it arrives. A respawn hands the
client a fresh player entity, so the guard's low-pass and the
reconstructed timers are reset when the entity instance changes.

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
covers what a real client can still give away: rotation is low-pass
smoothed, rate-clamped, given a small per-tick jitter and snapped to the
vanilla 0.15° mouse-sensitivity grid; an attack additionally requires a
clear line of sight to the target (not just reach + facing cone) and holds
fire for one tick right after a big turn (no "spun and hit same tick");
sneak can't be toggled faster than `min_sneak_hold_ticks`. A buried kit item
*is* juggled into the hotbar now too - via a `SWAP` click on the player's
own always-open inventory screen handler, rate-limited by
`hotkey_swap_min_gap_ticks` - mirroring `azalea_bot`'s open/click/close.
