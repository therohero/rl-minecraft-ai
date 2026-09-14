# rl-minecraft-ai

A reinforcement-learning Minecraft PvP bot: train a policy by self-play in a
fast headless Rust sim, then run it against a real Minecraft server.

## Layout

The repo is split into three top-level folders:

| folder | what it is |
|---|---|
| [`training/`](training/) | the RL training pipeline. [`training/sim/`](training/sim/README.md) is the headless Rust PvP simulation backend; `training/python/` is the PPO self-play trainer, feature engineering, opponent league, and model export. Checkpoints land in `training/checkpoints/`. |
| [`azalea-bot/`](azalea-bot/README.md) | the seam between a trained checkpoint and a live game: `inference_server.py` serves the exported policy over localhost HTTP, and `azalea_bot/` is a ready-made Rust Minecraft client (via the `azalea` crate) that drives the bot on a real server, with off-the-tick inference and a client-side legality guard. |
| [`mod/`](mod/README.md) | a client-side **Fabric mod** (MC 1.21.11). `/fight` takes over with the trained policy (via `azalea-bot/inference_server.py`) - passive by default, `/fight target` to engage; `/fight train` also records the fight as a dataset (one file per fight), tagged with the kit auto-detected from your inventory (uhc / sword / axe, default sword). |

## Quickstart

**1. Train** (builds the Rust sim, sets up `.venv/`, runs self-play; `Ctrl+C`
stops it after saving `training/checkpoints/latest.pt`):

```bash
./run.sh                 # bash (Linux/macOS/WSL/Git Bash)
.\run.ps1                # native Windows PowerShell
./run_docker.sh          # in a container (Docker or Podman, GPU; see below)
```

**2. Serve the trained model**, then play the bot one of two ways:

- **as a Fabric mod in your own client** - start the model server, then
  launch Minecraft with [`mod/`](mod/README.md) and type `/fight`
  (or `/fight train`):

  ```bash
  ./run_bot_mod.sh         # export latest checkpoint + serve on 127.0.0.1:8800
  .\run_bot_mod.ps1
  ```

- **as a headless bot** on a server:

  ```bash
  ./run_bot.sh  [ip] [port] [username] [inference_url] [mc_version]
  .\run_bot.ps1 [ip] [port] [username] [inference_url] [mc_version] [auth]
  ```

  The `run_bot` scripts export + start the inference server themselves (or
  reuse one already up), then build and run `azalea_bot`. See
  [`azalea-bot/README.md`](azalea-bot/README.md) for auth and other-version
  (ViaProxy) details.

**3. (optional) Fine-tune on live fights.** `/fight train` in the mod records
each fight to `rl-datasets/`. Fold that back into the policy:

```bash
./run_train_mod.sh [dataset_dir]    # advantage-weighted offline update -> latest.pt
./run_bot_mod.sh                    # re-export + serve the updated model
```

## Requirements

Rust (via [rustup](https://rustup.rs)) and Python 3.10+. `run.sh` / `run.ps1`
build the sim, create the repo-local `.venv/`, and install the Python deps
into it - nothing touches your global environment.

**GPU is optional and auto-detected.** The dependency step
(`training/python/ensure_deps.py`, called by every `run*` script) picks the
`torch` wheel that matches the machine: the CUDA build if an NVIDIA GPU is
visible (`nvidia-smi` or `/proc/driver/nvidia`), otherwise the slim
CPU-only build (~200 MB vs ~3.5 GB). It also *re-syncs* on later runs - move
the repo to a GPU box and the next `./run.sh` swaps in the CUDA wheel;
move it back and it swaps in the CPU one and frees the space. Force a
choice with `RL_TORCH_BACKEND=cpu` or `=cuda`. The trainer / inference
server still pick the actual compute device at runtime (`--device auto` by
default - CUDA/ROCm, Apple MPS, Intel XPU, DirectML, else CPU; see
`training/python/device.py`); for a non-NVIDIA accelerator install the
matching wheel yourself per `training/python/requirements.txt`.

## Docker / Podman

Training also runs in a container, with no local Rust, Python or `.venv/`
needed. `run_docker.sh` works with either engine and picks the right flags
for the one it finds. It builds one of two Dockerfile targets - pick
explicitly, it's not a GPU-with-CPU-fallback:

```bash
./run_docker.sh                        # == ./run.sh, in a container - GPU target
./run_docker.sh --num-arenas 512       # args are forwarded to train.py
RL_DOCKER_TARGET=cpu ./run_docker.sh   # the CPU target instead
```

The **CPU target isn't just "no GPU available"** - the PPO update (one big
batched matmul per step) wants a GPU, but rollout collection is many small
forward passes, one per sim step, so it's latency- rather than throughput-
bound (same reasoning as `--torch-threads`/`--update-threads` in
`train.py --help`); a few CPU threads can beat a GPU's per-call launch
overhead there. It's also a much smaller image (~1 GB vs ~7.5 GB, no CUDA
userspace, plain `python:slim` base instead of an NVIDIA one) if you don't
have a GPU at all. `docker compose`'s equivalent is the `train-cpu` service
(`docker compose run --rm --build train-cpu`) alongside `train`.

`training/checkpoints/` is bind-mounted, so `latest.pt`, the numbered
snapshots, `league/` and `metrics.csv` land on the host exactly as they do
with `./run.sh` - a later `./run_bot_mod.sh` exports and serves them without
caring that the training ran in a container. Stopping is the same too:
Ctrl+C (or `docker stop`, via `STOPSIGNAL SIGINT`) lets train.py save
`latest.pt` before it exits.

Only the **training** half of the repo is containerised. `azalea-bot/` and
`mod/` talk to a real Minecraft server and a real game client, so they stay
on the host.

### Podman notes

Rootless Podman differs from Docker in two ways that `run_docker.sh` handles
for you, worth knowing if you run the container by hand:

- **The GPU goes through CDI, not `--gpus`.** Podman has no `--gpus` flag;
  it wants a spec file from the NVIDIA Container Toolkit. One-time setup:

  ```bash
  sudo nvidia-ctk cdi generate --mode=wsl --output=/etc/cdi/nvidia.yaml   # drop --mode=wsl off WSL2
  podman run --rm --device nvidia.com/gpu=all \
      docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi            # verify
  ```

  Without a spec under `/etc/cdi` or `/var/run/cdi`, `run_docker.sh` says so
  and stops rather than silently training on the CPU.
- **`--userns=keep-id` on the bind mount.** The image runs as uid 1000;
  rootless Podman would otherwise map that to a *subuid* (100999-ish) and
  leave you with checkpoints your own account can't write or delete.

`compose.yaml` is the Docker path specifically - its GPU reservation needs
**Compose v2**. The `docker-compose` 1.29.2 that Debian/Ubuntu package
silently ignores `deploy.resources`, so use `run_docker.sh` there.

### How the image differs from `run.sh`

- **`ensure_deps.py` is bypassed.** It picks a `torch` wheel from the
  hardware it sees at *runtime*, which is the wrong question when building
  an image. The [`Dockerfile`](Dockerfile)'s two targets pin a backend each
  instead: the GPU `runtime` target installs `cu128` (torch >= 2.7, the
  first build with Blackwell/sm_120 kernels) - override for an older card or
  a different CUDA with `--build-arg TORCH_INDEX_URL=...`:

  ```bash
  podman build --target runtime --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 -t rl-minecraft-ai-train .
  ```

  The CPU `runtime-cpu` target installs the plain `.../whl/cpu` wheel the
  same way `ensure_deps.py` would on a GPU-less machine.
- **The CUDA base image is the `-base` tag, not `-runtime`,** and its
  bundled CUDA apt source is deleted. The torch wheel carries its own
  CUDA/cuDNN userspace, so the container only needs the driver, which the
  toolkit injects - and nothing is installed from NVIDIA's apt repo, which
  otherwise breaks every build whenever that mirror is mid-sync.
- **Base images are fully qualified** (`docker.io/library/rust:...`). Docker
  infers the registry; Podman refuses to guess unless the host's
  `registries.conf` says so.

`--num-arenas` defaults are container-aware already: the auto-detection
reads `sched_getaffinity`, so a `--cpus`-limited container sizes itself to
what it was actually given rather than to the host's core count.

## Architecture

Three processes, two very different transports, one shared feature
definition.

```
                    TRAIN                                     PLAY LIVE
  ┌───────────────────────────────────┐         ┌──────────────────────────────────┐
  │  train.py (PPO, self-play)        │         │  inference_server.py             │
  │    │  actions  [N·10] f32          │         │    loads model/policy.pt         │
  │    ▼          UDP :9999 (binary)   │         │    POST /act  (HTTP + JSON)      │
  │  mc_pvp_sim (Rust sim, N arenas)  │         │      ▲                 │ action   │
  │    │  observations [N·W] f32       │         │      │ observation    ▼          │
  │    ▼                               │         │  azalea_bot  /  mod's /fight     │
  │  RolloutBuffer ─► GAE ─► update    │         │    (real Minecraft client)      │
  │    │                               │         │      │ guard ─► client API      │
  │    ▼  every --checkpoint-every     │         │      ▼                           │
  │  training/checkpoints/latest.pt ───┼────────►│  export_model.py ─► model/*     │
  └───────────────────────────────────┘         └──────────────────────────────────┘
```

- **`training/sim/`** (`mc_pvp_sim`) is a headless, wall-clock-uncoupled
  reimplementation of just enough *vanilla* Minecraft PvP to train against:
  vanilla per-tick movement/collision on a voxel world, raycast melee with
  the vanilla two-step (client picks off its latency-stale view, server
  re-validates reach), the 1.0 s i-frame window, shields, sweep, bows,
  three kits, placeable blocks + flowing fluids, hunger, opt-in splash
  potions (9-effect status table) and enchants (Fire Aspect / Flame /
  Punch / Knockback / Knockback Resistance), and simulated network latency
  as a domain-randomization knob. It runs `N` arenas in
  parallel across CPU cores (rayon) and steps as fast as the trainer feeds
  it actions. Full detail in [`training/sim/README.md`](training/sim/README.md).

- **`training/python/`** launches the sim as a subprocess, speaks the
  binary-over-UDP protocol (`env.py`), and trains **one shared policy**
  (`ppo_agent.py`) that controls *every* player slot on *both* teams in
  *every* arena. Because every observation is built relative to "self" and
  the nearest teammates/enemies (`features.py`, mirroring
  `sim/src/observation.rs`), shared-policy control *is* self-play - no
  separate opponent process. An opponent league (`opponents.py`) mixes in
  frozen past snapshots and a fixed scripted bot for a fraction of arenas
  so training doesn't converge to something only a mirror of itself can't
  punish.

- **`azalea-bot/`** is the live seam. `export_model.py` turns a checkpoint
  into a portable `model/policy.pt` (TorchScript) + `model/spec.json` (the
  exact obs/action field order and the sim constants that run trained
  against). `inference_server.py` loads those and serves `POST /act` +
  `GET /spec` over HTTP+JSON so a bot in any language can drive the policy.
  `azalea_bot/` is a reference client: it builds the trained observation
  from live game state every tick, runs inference **off the tick loop**,
  and passes every action through a **client-side legality guard** before
  it reaches the wire.

- **`mod/`** is a second consumer of that same HTTP seam - a Fabric client
  mod whose `/fight` command does what `azalea_bot` does but from inside a
  real vanilla client, and whose `/fight train` records `(observation,
  action)` JSONL per tick - one file per fight, each with a trailing
  win/loss outcome record - for the offline fine-tune loop.

### The observation, in four places

The feature vector the policy sees must be byte-identical wherever it's
built. It is defined in four files that have to stay in lockstep:

| file | role |
|---|---|
| `training/sim/src/observation.rs` | the source of truth - builds the wire `Observation` from arena state |
| `training/python/features.py` | decodes that wire row into the policy input (and `observation_to_row` does the same from a dict, for the mod dataset) |
| `azalea-bot/azalea_bot/src/main.rs` (`build_observation`) | reconstructs the same fields from live `azalea` client state |
| `mod/src/client/java/rl/minecraft/ai/client/obs/ObservationBuilder.java` | the Java port of `build_observation` for the Fabric mod |

`sim/src/protocol.rs::WIRE_VERSION` is bumped on any incompatible layout
change; `env.py` carries the matching constant and the sim drops datagrams
whose version doesn't match. The `Hello` handshake sends every
normalization constant and the full resolved `SimConfig`, so the Python
side never hand-copies a value, and `spec.json` carries the same set
downstream to the live bots.

### One training update, end to end

1. **Rollout** (`--rollout-len` steps, default 128). Each step: the CPU
   `collect_model` samples an action for all `2·num_arenas` slots
   (`ActorCritic.act` - plain tensor ops, not `torch.distributions`, on the
   tiny per-step batch), `build_actions` packs them into the `[N, 10]` f32
   wire array, opponent-controlled slots are overwritten with the league
   opponent's actions, `env.step` sends the batch over UDP and reads the
   state batch back, and the transition is appended to the on-CPU
   `RolloutBuffer`.
2. **GAE** over the filled buffer (`compute_gae`, γ/λ), then a bulk transfer
   of the whole buffer to the training device (a no-op on CPU; on GPU this
   is the *only* host↔device transfer per update - collection stays on CPU
   because a per-step `.cpu()` sync would dwarf a network this small).
3. **PPO update** (`--ppo-epochs`, default 2): shuffle, split into
   `--minibatch-size` chunks, clipped surrogate + value loss + annealed
   entropy bonus, global grad-norm clip. Opponent slots are masked out
   (`sample_mask`) so the policy never trains toward imitating a frozen
   snapshot or the scripted bot. LR and entropy coef are linearly annealed
   over `--total-updates`.
4. Sync `collect_model` from the updated weights; every
   `--opponent-snapshot-every` updates, freeze a snapshot into the league
   pool. The pool is mirrored to `training/checkpoints/league/` and reloaded
   on resume, so a restarted run doesn't start with an empty league
   (`--fresh` wipes it with the checkpoints).

The rollout is latency-bound (one UDP round-trip per step), so it runs on
few torch threads (`--torch-threads`); the update is a big batched matmul
the sim sits idle through, so it grabs most cores (`--update-threads`), and
the loop flips between the two counts. `--num-arenas` auto-scales from the
detected CPU count (~12/thread, clamped) since the rollout buffer's RAM
grows linearly with it.

`--pipeline-rollout` overlaps steps 1 and 3 instead of running them back to
back: a background thread keeps collecting the *next* rollout into a second
`RolloutBuffer` while the main thread runs GAE + the PPO update on the one
just filled, handing buffers back and forth through a pair of queues (depth
1, so the collector is never more than one rollout ahead). Both halves
already release the GIL for most of their work (`env.step` blocks on a
socket recv; the PPO update is torch tensor ops), so this is close to free
throughput once the update is a meaningful fraction of wall-time. It always
keeps `collect_model` as a separate CPU copy of the policy (even on CPU,
where it's normally the same object as `model`), synced under a lock right
after every update - so rollout collection lags the training weights by up
to one PPO update, the standard async-collection trade-off. Off by default.

### Checkpoint lifecycle

`_save_checkpoint` writes both `latest.pt` (the auto-resume point, via a
temp-file + atomic rename) and a numbered `policy_update_<n>.pt` history
file, pruning to `--keep-checkpoints`. A checkpoint also gets written once
on **any** exit - normal finish, `Ctrl+C`, or crash (with `SIGINT` ignored
for that critical section so an impatient second `Ctrl+C` can't corrupt
it). Re-running `./run.sh` with the same command resumes from `latest.pt`
(model + optimizer state + update count); `--fresh` wipes everything. A
checkpoint stores the trunk shape (including the LSTM width, if any),
`obs_dim`, `frame_stack`, the sim constants it trained against, and the
full `SimConfig` - resuming refuses to load if the architecture,
observation space or frame-stack depth no longer matches.

### Frame stacking (`--frame-stack N`)

The trunk is a memoryless MLP, but the sim is a POMDP - the view of the
other players is latency-delayed and occlusion-limited. `--frame-stack N`
(default 1) feeds the policy the last `N` observations concatenated, so
the same MLP gets a short history at `N`x the input width, with no
recurrent state and no change to the PPO update. A slot's history is
cleared the tick its match ends. `evaluate.py` picks the depth up from
the candidate checkpoint. The live inference bridges (`azalea-bot`, `mod`)
build a single frame and don't stack yet, so `export_model.py` refuses an
`N>1` checkpoint - train the deployable policy with `--frame-stack 1`.

### LSTM policy head (`--lstm`)

Where `--frame-stack` gives the MLP a *fixed* window of history, `--lstm`
replaces the memoryless trunk head with a single-layer LSTM
(`--lstm-hidden`, default 256) that carries state across ticks -
unbounded memory for the POMDP, truncated at the rollout boundary for
BPTT. The rollout loop carries the hidden state per slot and zeroes it the
tick a match ends (like the frame stacker's history clear); the PPO update
(`_ppo_update_recurrent`) replays each slot's rollout as a sequence, one
minibatch = a batch of whole slot-sequences. `evaluate.py` carries hidden
state per player. The MLP stays the default. The live inference bridges
keep no recurrent state between ticks yet, so `export_model.py` refuses an
`--lstm` checkpoint - train the deployable policy without it.

### Terrain curriculum

`--terrain-curriculum-updates N` ramps the sim's `terrain_max_amplitude`
from `--terrain-curriculum-start` (default 0 = flat) up to the target over
the first `N` PPO updates, in `--terrain-curriculum-stages` discrete steps.
Each step is a fast sim relaunch — the model, optimizer, rollout buffer and
opponent pool stay in memory — so the policy learns flat movement first and
only meets rough terrain once it can walk. Off by default (constant
amplitude); resuming a checkpoint past update `N` just trains at the full
amplitude.

### Metrics

The progress line (every `--log-every` updates, default 10) carries the
rolling return, win rate, `win_vs_scripted`, the PPO losses, entropy,
approximate KL and clip fraction, the terrain amplitude, plus steps/sec.
`--metrics-csv [path]`
also writes those as a CSV (one row per line, appended under the existing
header on resume; no extra dependency); `--tensorboard [dir]` mirrors them
to a TensorBoard event dir when the `tensorboard` package is installed (a
missing package is a one-line warning, not an error). Both default to
`<checkpoint-dir>/metrics.csv` / `<checkpoint-dir>/tb` when given without
an argument.

### Rating a checkpoint (`evaluate.py`)

`win_vs_scripted` in the training log is the only strength signal that
isn't circular, and it's coarse. `python training/python/evaluate.py`
runs a round-robin: a *candidate* checkpoint (default `latest.pt`) against
a ladder of past `policy_update_*.pt` snapshots plus the `ScriptedOpponent`,
in the same sim / kit / reward config the candidate trained under (read
from its checkpoint). Every pair plays `--matches-per-pair` matches split
evenly across the two starting sides, run continuously across
`--num-arenas` arenas, and a Bradley-Terry fit turns the pairwise results
into one Elo-scaled number per player. It only reads checkpoints - no
training state is touched.

### Two transports, on purpose

| link | transport | why |
|---|---|---|
| `train.py` ↔ `mc_pvp_sim` | flat little-endian `f32` batches over **UDP** on loopback, 10-byte framed, strict request/response with a seq number and cached-reply retransmit | millions of steps against one fixed peer - JSON-encoding thousands of small floats per step was a measurable slice of the loop, and TCP's ack/Nagle/head-of-line buys nothing on loopback |
| live bot ↔ `inference_server.py` | **HTTP + JSON**, `POST /act` / `GET /spec` | ~20 Hz against an arbitrary-language client - reach and correctness matter far more than microseconds; measured ~1 ms median round-trip, well inside a 50 ms tick, and `azalea_bot` runs inference off the tick loop anyway |

### The live legality guard

The policy trains in a sim that is vanilla-*shaped*, not vanilla-*exact*,
so its raw output can be physically impossible for a real client (a
170°/tick aim snap, an attack six blocks away or through a wall, a
machine-gun click rate, a mid-air jump). A modern anticheat flags exactly
those. Both live bridges rewrite every action to stay inside what a legit
vanilla client can do - rotation rate-limited, low-pass smoothed,
sub-degree tremor, snapped to the vanilla 0.15° mouse grid; attacks only
when a hitbox is genuinely under the crosshair (reach + line of sight +
aim settled) at a randomised human click cadence; illegal sprint/jump
dropped; crouch debounced. It's geometry, rate-limiting and humanisation
only - it never invents inputs. `azalea_bot`'s is `src/guard.rs` (every
knob has an `AZALEA_GUARD_*` override); the mod's is `ClientGuard.java`
(lighter, since a real client already enforces jump-on-ground, real hunger
cost and the real attack cooldown for free). Details in each sub-README.

## Tests

```bash
pytest                          # Python trainer / feature tests
cd training/sim && cargo test   # Rust sim tests
cd azalea-bot/azalea_bot && cargo test   # live bridge: guard geometry, config
cd mod && ./gradlew build       # Fabric client mod

python training/python/smoke_train.py   # end-to-end: real sim + train.py, fresh + resume, <1 min
```

**CI** (`.github/workflows/ci.yml`) runs all of the above on every push and
pull request, in four parallel jobs: the two `cargo test` suites, `pytest`,
`smoke_train.py` (real sim + `train.py`), and the mod's `./gradlew build`.

`smoke_train.py` is the fast end-to-end check for a training-side change:
it runs the actual Rust sim and `train.py` twice on a tiny config (few
arenas, short rollouts, a handful of updates) against a throwaway
checkpoint dir, then asserts on the run's logs and the checkpoint/league
files it leaves behind — a fresh run trains and checkpoints, a second run
resumes from `latest.pt` at the right update (also checking the
`--metrics-csv` output), then `evaluate.py` rates the resulting
checkpoints and prints an Elo table. Needs the sim built once
(`cd training/sim && cargo build --release`).

The `pytest` suite cross-checks the inference bridge's per-observation
decode (`features.observation_to_row`) against training's vectorized one
(`wire_batch_to_obs`) so the two never drift, and checks that
`ActorCritic.evaluate` reproduces the log-probs `ActorCritic.act` sampled.
