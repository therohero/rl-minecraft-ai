# Bot Bridge: plugging the trained model into a real Minecraft client/server

This directory is the seam between the trained PyTorch policy and an
actual live Minecraft game. Nothing in the training pipeline (Rust sim,
PPO, self-play) needs to know this exists; nothing here needs to know
about PyTorch internals beyond loading a TorchScript file.

## Pieces

- `model/policy.pt`, `model/spec.json` - produced by
  `training/python/export_model.py` from a training checkpoint. Not committed;
  regenerate whenever you export a new checkpoint.
- `inference_server.py` - loads `model/policy.pt` and serves a tiny HTTP
  API (`POST /act`) that takes a JSON observation and returns a JSON
  action, using the exact same feature engineering (`training/python/features.py`)
  the policy was trained on - the repo's `pytest` suite cross-checks the
  bridge's per-observation decode against training's vectorized one so the
  two never drift. It also serves the whole `spec.json` at
  `GET /spec` so a live bot can fetch the sim constants + hotbar layout the
  policy trained against instead of hardcoding them.
- `azalea_bot/` - a ready-made reference client instead of the "write your
  own integration" path below: a real Minecraft client (via the `azalea`
  crate) that connects to an actual server, builds the observation from
  live game state every tick, calls `inference_server.py`, and applies the
  returned action. Inference runs **off the tick loop** (a slow model never
  freezes the client) and every action passes through a **client-side
  legality guard** before it hits the wire. See its own doc comment
  (`azalea_bot/src/main.rs`) and the section below for details, plus the
  auth / ViaProxy notes.

There is also a second consumer of this same seam: [`mod/`](../mod/README.md)
is a Fabric **client** mod that adds a `/fight` command driving the policy
from inside a real vanilla client (and `/fight train` to record live
episodes as a dataset). Its `ObservationBuilder.java` is a Java port of
`azalea_bot`'s `build_observation` - keep the two in sync.

## Why HTTP instead of embedding PyTorch in the bot?

Real Minecraft bot frameworks are usually not Python:
[mineflayer](https://github.com/PrismarineJS/mineflayer) is Node.js,
[Baritone](https://github.com/cabaletta/baritone) and Fabric/Forge mods
are Java/Kotlin. Rather than forcing one of those ecosystems to embed
libtorch, the trained model is served over localhost HTTP+JSON - any
client that can make an HTTP request can drive the bot.

Note this is a *different* seam from the training transport. The
sim↔trainer link (`training/sim/` ↔ `training/python/`) is a hot loop that runs millions of
steps with one fixed peer, so it uses binary batches over UDP for speed.
This bridge runs at ~20 Hz against an arbitrary-language client, so
correctness and reach matter far more than microseconds - plain HTTP+JSON
stays. Measured localhost `/act` round-trip (full observation, pooled
keep-alive connection) is ~1 ms median / low-single-digit-ms mean, well
inside the 50 ms tick, and `azalea_bot` runs inference **off** the tick
loop anyway - so an occasional slow response costs nothing. The bot logs a
rolling `inference /act round-trip` summary (mean / max / count over
budget) every 30 s; if that ever shows a real problem on your hardware,
the fix that keeps the language-agnostic seam is an optional binary
(flat-`f32`) body on the same endpoint rather than a whole new transport.

`inference_server.py` runs the policy on CPU by default (a single tiny
observation per request is usually faster there than paying a host↔GPU
copy). Pass `--device auto` / `cuda` / `mps` / `xpu` / `directml` to use
an accelerator (see `training/python/device.py`).

## The one-command path

`./run_bot.sh` (bash) or `.\run_bot.ps1` (native Windows PowerShell) does
the whole sequence for you: export `training/checkpoints/latest.pt` if no
`model/policy.pt` exists yet, start `inference_server.py` in the background
(reusing one already on port 8800), then build and run `azalea_bot`
against the server, stopping the background server again on exit. All args
are optional:

(For the [`mod/`](../mod/README.md) Fabric client instead of the headless
bot, use `./run_bot_mod.sh` - it exports + serves the model and nothing
else.)

```
./run_bot.sh  [ip] [port] [username] [inference_url] [mc_version]
.\run_bot.ps1 [ip] [port] [username] [inference_url] [mc_version] [auth]
# defaults: localhost 25565 TrainedBot http://127.0.0.1:8800/act
```

Both scripts use the repo-local `.venv/` Python (the torch the model was
exported with) if `run.sh`/`run.ps1` has created it, else a system Python.

**Authentication** (`AUTH=` env for bash, 6th arg for PowerShell, or
`--auth` straight on the bot): `offline` (default) connects unauthenticated;
`microsoft` runs azalea's device-code login (a code + URL is printed the
first time) and caches the token under `~/.minecraft/azalea-auth.json`,
keyed by the username argument.

**Other Minecraft versions** (`MC_VERSION=` env / the 5th arg): the bot
speaks exactly the protocol `azalea` is pinned to (see `Cargo.toml`'s
`+mcX.Y.Z`). To reach a server on any other version, set `mc_version` to the
server's version - the scripts then download [ViaProxy](https://github.com/ViaVersion/ViaProxy)
once into `azalea-bot/viaproxy/`, run it locally (needs a JRE 17+ on PATH),
and point the bot at it. ViaProxy translates between the two versions;
`azalea-bot/viaproxy/viaproxy.yml` is written with `auth-method: NONE`
(offline upstream) - edit it for an online-mode target (see ViaProxy's
docs / account manager).

**Troubleshooting:**
- `disconnected by the server: <reason> - auto-reconnecting in ~5s` on a
  loop: the server is refusing the bot. Common causes: it's whitelisted /
  online-mode (use `--auth microsoft`), a duplicate login (a previous
  `azalea_bot` still connected), or a Minecraft-version mismatch (set
  `mc_version` to run through ViaProxy).
- The guard log shows nearly everything being dropped ("300 illegal
  attacks dropped, 280 mid-air jumps denied ..."): the policy isn't trained
  yet - a random policy emits mostly illegal actions and the guard
  correctly suppresses them. Train longer, re-export, restart.

## The `azalea_bot` reference client

`azalea_bot` is not just a thin "read state → POST → apply" loop. Two
layers sit between the policy and the server:

### How one tick flows

`main.rs::handle` is an `azalea` event handler. The interesting events:

- **`Event::Spawn`** - learn the bot's own network entity id (needed to
  tell its own hurt animation from everyone else's).
- **`Event::Packet`** - fed straight to `mod tracker`, which rebuilds state
  the 1.21.11 protocol doesn't hand a client as one field: scoreboard
  teams (`SetPlayerTeam`), other players' main-hand item (`SetEquipment`),
  the bot's own hurt window (`HurtAnimation` / `DamageEvent`) and shield
  lock-out (`Cooldown` naming the shield).
- **`Event::Tick`** (every 50 ms): (1) `build_observation` reads live client
  state into the same `Observation` struct `features.py` expects and
  `send_replace`s it into a `watch` channel for the async worker; (2) the
  freshest `Decision` the worker has produced is taken, run through
  `guard::Guard::sanitize`, and applied via `apply_action` (which only
  *performs* - every choice was already made in the guard).
- **`Event::Death`** - reset the tracked look direction and the guard's
  rotation low-pass (the server snaps orientation on respawn).
- **`Event::Disconnect`** - log the kick reason (azalea's bundled
  `AutoReconnectPlugin` retries in ~5 s), so a version mismatch shows up as
  a logged loop rather than a silently frozen bot.

`build_observation` maps live Minecraft items onto the training kit's 17
`Item` ids (`Item::from_kind` - diamond/netherite sword → `Sword`,
enchanted golden apple → `GoldenHead`, …), scans the loaded world for the
block-top heights and the yaw-rotated `block_view` grid
(`block_column_view`, a port of `sim/src/blocks.rs::column_view`), reads
its own tab-list latency for `self_ping_ms`, fills `self_effects` from the
`ActiveEffects` component, and asks `mod tracker` for the four
reconstructed timers. `self_mining` is always 0 (this bot doesn't mine) -
the field just keeps the wire row the right width. Splash potions are not
yet distinguished by contents or thrown (see `TODO.md`).

The observation constants (`max_hp`, `arena_radius`, the combat-timing
numbers the tracker needs, `input_order`, …) are fetched once from
`GET /spec` at startup and fall back to `SimConfig::default()` if that
fails - a warning tells you the normalization may be wrong if you trained
with a non-default config.

### Off-the-tick-loop inference (`azalea_bot/src/inference.rs`)

The HTTP round-trip to `inference_server.py` runs on its own Tokio task.
Each game tick the handler drops the freshest observation into a channel
(overwriting one the worker hasn't taken yet) and immediately applies the
**most recent action already available**. So the client acts on every
50 ms tick regardless of model latency; the action is at most one
round-trip stale, and the handler logs a warning if inference falls more
than ~8 ticks behind. The old bridge `await`ed the request inside the
tick, so a slow model or a network hiccup froze the whole client.

### Client-side legality guard (`azalea_bot/src/guard.rs`)

The policy trained in a sim that is vanilla-*shaped*, not vanilla-*exact*,
so its raw output can be something a real client physically cannot do -
and a modern server anticheat (GrimAC, Vulcan, Themis, NCP, ...) will kick
the bot for exactly that. Before any action is applied, the guard rewrites
it to stay inside legit-client bounds:

| check | what it does | default |
|---|---|---|
| rotation rate | clamps yaw/pitch change per tick, with a low-pass so the aim is never a perfectly linear ramp | 80 deg/tick yaw, 60 deg/tick pitch, smoothing 0.7 |
| rotation realism | adds a sub-degree per-tick gaussian "hand tremor" so no two deltas are identical, then snaps the absolute rotation to the **vanilla 0.15 deg mouse-sensitivity grid** - the "GCD" every rotation-analysis check locks onto (`azalea` snaps to the same grid; the guard mirrors it so its aim math and the wire never drift) | tremor sigma 0.4 deg |
| attack target | only swings when a **player hitbox is genuinely under the crosshair**, within reach, with a clear line of sight (voxel raycast against the loaded world) - and attacks *that* entity, not just the nearest one | reach 3.0, hitbox +0.1, LOS on |
| aim settle | holds fire for one tick right after a big turn (no "spun 50+ deg and hit on the same tick") | 50 deg |
| click cadence | enforces a **randomised** minimum gap between clicks (jittered so the inter-click series isn't a metronome) plus a hard 1 s-window ceiling | 12 CPS avg, 25 ms jitter |
| jump | dropped unless the client is actually on the ground (a mid-air jump is a textbook Fly / NoFall signature) | - |
| sprint | drops sprint when hunger is too low, an item is being used, sneaking, or not moving forward | food > 6 |
| sneak | can't toggle crouch faster than a minimum hold (rapid crouch spam is its own flag) | 3 ticks |
| attack vs. use | never sends an attack and a `use_item` on the same tick | - |

It is **geometry, rate limiting and humanisation only** - it never invents
inputs. Every limit has an env override; it logs a periodic summary of what
it corrected:

```
AZALEA_GUARD_MAX_YAW_DEG,  AZALEA_GUARD_MAX_PITCH_DEG   per-tick rotation caps (degrees)
AZALEA_GUARD_SMOOTHING        rotation low-pass factor 0.05..1.0 (1.0 = off)
AZALEA_GUARD_AIM_JITTER_DEG   per-tick rotation tremor sigma (degrees; 0 = off)
AZALEA_GUARD_AIM_SETTLE_DEG   turn size above which the attack waits a tick
AZALEA_GUARD_REACH           melee reach in blocks
AZALEA_GUARD_HITBOX_EXPANSION crosshair hitbox inflation in blocks
AZALEA_GUARD_MAX_CPS         average attacks/second (min click gap = 1000/this ms)
AZALEA_GUARD_CLICK_JITTER_MS  sigma of the random jitter on that gap
AZALEA_GUARD_MIN_FOOD_SPRINT  hunger floor for sprinting
AZALEA_GUARD_MIN_SNEAK_HOLD_TICKS  minimum ticks a crouch state is held
AZALEA_GUARD_LOS=0/1       require line of sight to attack
AZALEA_GUARD_DISABLE=1     pass the raw policy action straight through
```

No client-side guard makes a bot undetectable against a well-tuned modern
anticheat - a policy that plays visibly better than any human still stands
out. This keeps an *honest* policy from being kicked for mechanically
impossible inputs on servers you are authorized to run.

This keeps an honest RL policy from *looking* like a cheat on servers you
are authorized to run (your own test servers, research, CTF events). It is
not a tool for hiding one - it only ever makes the bot *more* vanilla.

The [`mod/`](../mod/README.md) Fabric client has a Java counterpart,
`ClientGuard`, that goes further on rotation - it drives a modelled mouse
(reaction lag, acceleration cap, whole-mouse-count quantisation on the
client's real sensitivity curve, applied via `changeLookDirection`) rather
than snapping the absolute angle - since a client-side mod can, where the
headless `azalea` client only exposes an absolute set-rotation.

## Steps to test the bot live (by hand)

1. Train a while, then export a checkpoint:
   ```bash
   cd training/python
   python export_model.py --checkpoint ../checkpoints/latest.pt --out-dir ../../azalea-bot/model
   ```
2. Start the inference server:
   ```bash
   cd ../../azalea-bot
   python inference_server.py --model-dir ./model --port 8800
   ```
3. In your Minecraft bot integration (if you're not using the `azalea_bot`
   reference client above - this is the part you write for your specific
   client/bot framework, since it's otherwise game-specific), on every
   decision tick:
   - Read the bot's own position/velocity/look/HP from the game, and the
     nearest few other players' position/velocity/HP.
   - Report the bot's own real connection latency (ms) as `self_ping_ms` -
     the policy was trained with a randomized simulated latency in this
     field (see the main `README.md`) and expects a real value here, not a
     constant. In training this ping also delays the policy's own melee
     hit detection (it swings at where a target was a round-trip ago, like
     vanilla client-side hit registration), so an accurate `self_ping_ms`
     matters for more than just the observation. `azalea_bot` reads it
     straight from the client's tab-list entry for itself, which is the
     same value vanilla displays.
   - Build the observation dict per `model/spec.json` (`obs_field_order`):
     the `self_*` scalars (HP/vel/look/`self_held`/`self_absorption`/
     `self_eating`/`self_bow_draw`/`self_burning`/`self_shield_disabled`/
     `self_arrows`/`self_slot`/`self_swap_lockout`/`self_food` (0-20)/
     `self_sneaking`/`self_mining` (0-1 block-break progress, `uhc` pickaxe;
     `azalea_bot` doesn't mine, so it reports 0)/`self_effects` (9 floats,
     `amplifier + 1` per status effect) ...), an `inventory` list
     (per-item counts, order = `spec.json`'s `inventory_items`), a `hotbar`
     list (the `kit::Item` id in each of the 9 physical slots - see
     `spec.json`'s `hotbar_layout`), then `enemies`, `teammates`,
     `projectiles` - lists of blocks with the nearest first, zero-padded
     server-side - and `block_view`, a row-major list of
     `[top_rel, water, lava, cobweb]` columns for the yaw-rotated
     `block_view_size` x `block_view_size` grid around the bot. An
     other-player block is `{present, hp, rel_x, rel_y, rel_z, vel_x,
     vel_y, vel_z, ground_height, blocking, eating, held_ranged, sneaking}`;
     `rel_x/y/z` are relative to the bot, rotated into its own yaw frame
     (see `training/sim/src/observation.rs`). `ground_height` and `top_rel` are the
     **y of the topmost full-collision block** under that column (integer),
     matching the sim's voxel terrain - `azalea_bot` reads them from the
     loaded world; report 0 only if you have no world data. (The sim's
     `uhc` kit also rings the arena rim with a 3-block wall, which surfaces
     through exactly these `block_view` / `ground_height` fields - a live
     bridge just reports whatever real geometry the server has, no special
     handling.) A `sword`-kit
     bridge can
     leave `inventory`, `projectiles`, `block_view` and the kit-only
     `self_*` fields at their defaults; a bridge that can't tell allies
     from enemies can report everyone as an enemy (`azalea_bot` reads
     scoreboard teams and only falls back to that when a server sets none).
   - `POST` that as JSON to `http://127.0.0.1:8800/act`.
   - Apply the returned `move_x`/`move_z` as strafe/forward input,
     `yaw_delta`/`pitch_delta` as look-direction changes, trigger
     jump/attack when `jump`/`attack` are `true`, act on `held_slot`
     (`0-8` = press that number key; `9-20` = a vanilla number-key
     **hotkey** - drop `spec.json`'s `item_ids[held_slot - 9]` into the
     selected slot; `azalea_bot` just selects the slot if the item is
     already on the hotbar, and otherwise opens the inventory screen, sends
     the swap and closes it - rate-limited), fire `use_item` for the held
     item (shield / bow / eat / place - placement uses an eye raycast at
     ~4.5 blocks, so look where you want it), toggle crouch on `sneak`, and
     actually toggle your client's sprint state (not just fast forward
     movement) when `sprint` is `true` - vanilla's sprint-attack knockback
     bonus and crit-cancelling only kick in when the client is really
     flagged as sprinting, and a landed sprint-hit clears the flag.
4. Re-export periodically as training improves, and restart
   `inference_server.py` to pick up the new weights (or extend it to
   hot-reload - not implemented here to keep the seam simple).

## A minimal example request/response

```
POST /act
{
  "self_hp": 17.0, "self_vel_x": 0.0, "self_vel_y": -1.2, "self_vel_z": 0.4,
  "self_yaw": 1.57, "self_pitch": 0.0, "self_on_ground": true,
  "self_attack_cooldown": 0.0, "self_ping_ms": 34.0,
  "self_shield": 0.0, "self_hurt": 0.0,
  "self_dist_from_center": 0.3, "self_ground_height": 0.0,
  "self_slope_forward": 0.0, "self_slope_right": 0.0,
  "self_held": 1, "self_absorption": 0.0, "self_eating": 0.0,
  "self_bow_draw": 0.0, "self_burning": 0.0, "self_shield_disabled": 0.0,
  "self_arrows": 6, "self_slot": 0, "self_swap_lockout": 0.0,
  "self_food": 20, "self_sneaking": 0,
  "self_effects": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "time_left": 18.0, "enemies_alive": 1, "teammates_alive": 0,
  "inventory": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "hotbar": [1, 2, 4, 5, 0, 0, 0, 0, 0],
  "enemies": [
    { "present": true, "hp": 12.0,
      "rel_x": 1.1, "rel_y": 0.0, "rel_z": 2.4,
      "vel_x": -0.5, "vel_y": 0.0, "vel_z": 0.2,
      "ground_height": 0.0, "blocking": 1.0, "eating": 0.0,
      "held_ranged": 0.0, "sneaking": 0.0 }
  ],
  "teammates": [],
  "projectiles": [],
  "block_view": []
}

200 OK
{ "move_x": 0.12, "move_z": 0.98, "yaw_delta": -0.07, "pitch_delta": 0.0,
  "jump": false, "attack": true, "sprint": false, "use_item": false,
  "sneak": false, "held_slot": 1 }
```

`held_slot` in the response is `0..spec.json["hotbar_action_dim"]` (26 for
the current kits): `0-8` selects that physical hotbar slot, `9-25` hotkeys
`item_ids[held_slot - 9]` into the selected slot (`9` = empty hand).
`spec.json`'s `hotbar_layout` is the kit's starting slot -> item.
Under `--input-order legacy` (the default), a slot change / hotkey on the
same tick the bot attacks triggers the sim's attribute swap; under
`modern`, it costs a one-tick lockout. Missing optional
observation fields default to 0 / absent, so a simpler `sword`-kit bridge
can omit `inventory`, `hotbar`, `projectiles`, `block_view` and the
kit-only `self_*` fields and just send the melee-relevant state.

The `azalea_bot` reference client fills all of this from live game state -
real inventory / hotbar / arrow counts, real nearby arrows, a real block
view scanned from the loaded world, the real block-top height under and
around the bot's feet (`self_ground_height` / `self_slope_forward` /
`self_slope_right`), real `self_food` (hunger bar) and `self_sneaking`
(crouch state), best-effort enemy HP, each nearby player's `sneaking` bit
and (from `SetEquipment`) their main-hand item, so `blocking` / `eating` /
`held_ranged` are real rather than a bare "hand active" bit. The
`enemies` / `teammates` split follows `SetPlayerTeam`; with no scoreboard
teams in play everyone is an enemy, as before.

Four fields the 1.21.11 protocol doesn't hand a client as one ready-made
value `azalea_bot` reconstructs itself (`src/tracker.rs`):

| field | how it's derived |
|---|---|
| `self_hurt` | 20-tick timer restarted by the `HurtAnimation` / `DamageEvent` packet for the bot's own entity |
| `self_shield_disabled` | timer from the `Cooldown` packet that names the shield item (an axe hit) |
| `self_bow_draw` | ticks the bot has held right-click with a bow in hand, over `bow_max_draw_seconds` |
| `self_swap_lockout` | ticks since the bot's last hotbar swap (non-zero only when `spec.json`'s `input_order` is `modern`) |

`export_model.py` writes the constants these normalizations need into
`spec.json`'s `combat_constants`, so a non-default `--sim-config` still
normalizes correctly.

### Deterministic vs. sampled actions

`inference_server.py` takes the policy's **deterministic mode** by default
(`tanh(mean)` continuous, each binary head thresholded at 0.5, `argmax`
held-slot). That is stable but predictable - a frozen policy is easy for a
human opponent to read. Pass `--sample` to instead draw every action from
the trained distribution (`tanh(mean + std·noise)`, a Bernoulli per binary
head, a categorical held-slot), which "uses the training more"; `--temperature`
(default `1.0`) widens (`>1`) or sharpens (`<1`) that sampling. The
exported `policy.pt` now returns `(cont_mean, cont_std, binary_probs,
slot_probs)` so both paths work from one artifact - re-export any older
model.
