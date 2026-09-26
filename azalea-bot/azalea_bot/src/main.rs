//! Live bridge between a trained policy and a real Minecraft server.
//!
//! This connects an actual Minecraft client (via the `azalea` crate) to a
//! server. Every game tick the handler:
//!   1. builds the same observation dict the Python trainer used
//!      (`build_observation`; see python/features.py and
//!      sim/src/protocol.rs::Observation),
//!   2. hands it to an **async inference worker** (`mod inference`) that
//!      POSTs it to the local model server (azalea-bot/inference_server.py)
//!      off the tick loop, so a slow model never freezes the client,
//!   3. takes the worker's most recent action, runs it through the
//!      **client-side legality guard** (`mod guard`) - rotation rate limit,
//!      reach / line-of-sight / click-rate checks on attacks, hunger-gated
//!      sprint - and applies the result through Azalea's client API
//!      (`apply_action`).
//!
//! It never touches PyTorch directly - the seam is plain HTTP+JSON, so the
//! trained model can be swapped or retrained independently of this bot.
//!
//! Usage:
//!   cargo run --release -- [server] [username] [inference_url] [--auth offline|microsoft]
//!
//! Example:
//!   cargo run --release -- localhost:25565 TrainedBot http://127.0.0.1:8800/act
//!
//! `--auth offline` (the default) connects unauthenticated - offline/cracked
//! servers, and any server reached through a local ViaProxy (which does the
//! real upstream auth itself). `--auth microsoft` runs azalea's Microsoft
//! device-code flow (a code + URL is printed on first run) and caches the
//! token under `~/.minecraft/azalea-auth.json`, keyed by the `username`
//! argument; the in-game name then comes from the Microsoft profile.
//!
//! ## What the bot actually perceives
//!
//! Everything the client receives is read live: own HP / velocity / look /
//! absorption / on-fire / using-item / hunger / crouch state, the real
//! 9-slot hotbar and full inventory item counts, nearby players (best-effort
//! health, the shift-key-down bit, and each one's main-hand item from
//! `SetEquipment`), in-flight arrows, the block-top height under and around
//! the bot's feet (`self_ground_height` / `self_slope_*`), and a yaw-rotated
//! block-grid view scanned out of the loaded world, matching
//! `sim/src/blocks.rs::column_view` column-for-column.
//!
//! Four fields the 1.21.11 protocol doesn't hand a client as one ready-made
//! value are reconstructed in `mod tracker` from packets and from the bot's
//! own inputs: `self_hurt` (a 20-tick timer started by `HurtAnimation` /
//! `DamageEvent` for our entity), `self_shield_disabled` (from the
//! `Cooldown` packet naming the shield item), `self_bow_draw` (ticks holding
//! right-click with a bow) and `self_swap_lockout` (the modern input-order
//! post-swap lock-out the bot itself incurs). Other players' `eating` /
//! `held_ranged` come from their tracked main-hand item; friend/foe for the
//! `enemies` / `teammates` split is read from `SetPlayerTeam` (with no
//! scoreboard teams in play, everyone is a foe, as before).
//!
//! ## Legality guard (`mod guard`)
//!
//! The policy trained in a sim that is vanilla-*shaped*, not vanilla-*exact*,
//! so its raw output can be physically impossible for a real client: a
//! 170 deg/tick aim snap, an attack six blocks away or through a wall, a
//! machine-gun click rate, sprinting on an empty hunger bar, a mid-air jump.
//! A modern anticheat (GrimAC, Vulcan, Themis, ...) flags precisely those.
//! The guard rewrites every action to stay inside what a legit vanilla
//! client can do - rotation is rate-limited and low-pass smoothed, given a
//! sub-degree per-tick tremor and snapped to the vanilla mouse-sensitivity
//! grid (the rotation "GCD"); an attack only fires when a player hitbox is
//! genuinely under the crosshair (within reach, line of sight clear, aim
//! settled) at a randomised human click cadence; a mid-air jump is dropped;
//! the crouch toggle is debounced; sprint is dropped when it would be
//! illegal. Every limit has an `AZALEA_GUARD_*` env override;
//! `AZALEA_GUARD_DISABLE=1` passes the raw action through. It is geometry,
//! rate limiting and humanisation only - it never invents inputs, and it is
//! meant to keep an honest RL policy from *looking* like a cheat on servers
//! you are authorized to run, not to hide one.
//!
//! ## Async inference (`mod inference`)
//!
//! The HTTP round-trip to the model server runs on its own Tokio task, not
//! in the tick handler. Each tick publishes the freshest observation and
//! applies the latest action already available, so the client acts every
//! 50 ms tick no matter how slow inference is - the action is at most one
//! round-trip stale, which the handler tracks and warns about.
//!
//! IMPORTANT: this bot speaks exactly one Minecraft wire protocol - the one
//! `azalea` is pinned to in `Cargo.toml` (the `+mcX.Y.Z` build tag). It has
//! no built-in protocol translation. To reach a server on any other version,
//! run **ViaProxy** locally and point `--server` at it (`run_bot.sh` can
//! download and launch it for you - see `azalea-bot/README.md`); ViaProxy
//! translates between the two versions and handles the upstream
//! authentication, so the bot connects to it with `--auth offline`.
//! Connecting natively to a mismatched server fails like this: login
//! succeeds, the bot looks connected for a few seconds, then the server
//! drops it. The `Event::Disconnect` handler below logs the reason (and
//! azalea's bundled `AutoReconnectPlugin` retries every ~5 s, so you see the
//! disconnect on a loop rather than a silently frozen bot).
//!
//! `Cargo.toml` also builds azalea with `default-features = false` to drop
//! its `log` feature - otherwise azalea adds bevy's `LogPlugin`, which
//! fights this bot's own `env_logger` for the global logger and prints
//! "Could not set global logger as it is already set" on every startup.

mod guard;
mod inference;
mod tracker;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::watch;

use azalea::Account;
use azalea::BlockPos;
use azalea::block::BlockState;
use azalea::block::fluid_state::FluidKind;
use azalea::entity::metadata::{
    AbstractArrow, AbstractEntityShiftKeyDown, AbstractLivingUsingItem, Health, OnFire,
    Player as PlayerMarker, PlayerAbsorption,
};
use azalea::entity::{LocalEntity, Position};
use azalea::container::ContainerClientExt;
use azalea::inventory::operations::SwapClick;
use azalea::inventory::ItemStack;
use azalea::local_player::TabList;
use azalea::physics::collision::BlockWithShape;
use azalea::player::GameProfileComponent;
use azalea::prelude::*;
use azalea::registry::builtin::ItemKind;
use azalea::world::MinecraftEntityId;
use azalea::{ClientBuilder, SprintDirection, WalkDirection};
use bevy_ecs::prelude::{Entity, With, Without};
use log::{info, warn};
use serde::{Deserialize, Serialize};

use guard::{Guard, GuardConfig, SafeAction};
use inference::ActionCell;
use tracker::{Relation, Tracker};

/// Number of physical hotbar slots - mirrors `sim/src/kit.rs::HOTBAR_SLOTS`
/// and the policy's held-slot action head.
const HOTBAR_SLOTS: usize = 9;
/// Width of the `inventory` observation block - all `kit::Item`s except
/// `Empty` (`sim/src/kit.rs::ITEM_COUNT - 1`).
const INVENTORY_ITEMS: usize = 16;

/// Sim/normalisation constants the observation build needs. Defaults mirror
/// `SimConfig::default()`; the real values for the loaded policy are fetched
/// once from the inference server's `GET /spec` at startup (see `fetch_spec`).
#[derive(Clone, Copy, Debug)]
struct Consts {
    max_hp: f64,
    arena_radius: f64,
    match_time_seconds: f64,
    terrain_max_amplitude: f64,
    /// A diamond sword's vanilla attack-cooldown recharge (20 / 1.6 ticks).
    attack_recharge_ticks: f64,
    block_view_size: usize,
    max_observed_enemies: usize,
    max_observed_teammates: usize,
    max_observed_projectiles: usize,
    /// Combat-timing constants the bot needs to normalise the observation
    /// fields it now derives itself (see `mod tracker`), mirroring
    /// `sim/src/config.rs::CombatConfig`. Fetched from `spec.json`'s
    /// `combat_constants`.
    bow_max_draw_seconds: f64,
    axe_shield_disable_seconds: f64,
    swap_lockout_seconds: f64,
    hurt_invulnerability_seconds: f64,
    /// `Legacy` (free hotbar swaps) or `Modern` (each swap costs a one-tick
    /// attack/use lock-out). Only `Modern` makes `self_swap_lockout` ever
    /// non-zero.
    input_order: InputOrder,
    /// `0` for the default memoryless MLP policy. `> 0` for an `--lstm`
    /// checkpoint: the width of the `(h, c)` state `mod inference` then
    /// carries between ticks and threads through every `/act` call (see
    /// `LstmState` and `inference::run_worker`). From `spec.json`'s
    /// `arch.lstm_hidden`.
    lstm_hidden: usize,
}

/// Mirrors `sim/src/config.rs::InputOrder` - which of the vanilla
/// hotbar/attack/use orderings the policy trained against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputOrder {
    Legacy,
    Modern,
}

impl Consts {
    /// Ticks of attack/use lock-out one hotbar swap costs: the modern
    /// input order rounds `swap_lockout_seconds` up to whole ticks, the
    /// legacy order pays nothing.
    fn swap_lockout_ticks(&self) -> u32 {
        match self.input_order {
            InputOrder::Modern => (self.swap_lockout_seconds / 0.05).ceil().max(1.0) as u32,
            InputOrder::Legacy => 0,
        }
    }
}

impl Default for Consts {
    fn default() -> Self {
        Consts {
            max_hp: 20.0,
            arena_radius: 12.0,
            match_time_seconds: 90.0,
            terrain_max_amplitude: 3.0,
            attack_recharge_ticks: 12.5,
            block_view_size: 5,
            max_observed_enemies: 3,
            max_observed_teammates: 2,
            max_observed_projectiles: 2,
            bow_max_draw_seconds: 1.0,
            axe_shield_disable_seconds: 5.0,
            swap_lockout_seconds: 0.05,
            hurt_invulnerability_seconds: 1.0,
            input_order: InputOrder::Legacy,
            lstm_hidden: 0,
        }
    }
}

/// Partial view of `model/spec.json` (`azalea-bot/inference_server.py`
/// serves the whole file at `GET /spec`).
#[derive(Debug, Deserialize)]
struct Spec {
    #[serde(default)]
    sim_constants: SpecSimConstants,
    #[serde(default)]
    combat_constants: SpecCombatConstants,
    #[serde(default)]
    max_observed_enemies: Option<usize>,
    #[serde(default)]
    max_observed_teammates: Option<usize>,
    #[serde(default)]
    max_observed_projectiles: Option<usize>,
    #[serde(default)]
    block_view_size: Option<usize>,
    #[serde(default)]
    input_order: Option<String>,
    #[serde(default)]
    arch: SpecArch,
}

#[derive(Debug, Default, Deserialize)]
struct SpecArch {
    #[serde(default)]
    lstm_hidden: usize,
}

#[derive(Debug, Default, Deserialize)]
struct SpecSimConstants {
    max_hp: Option<f64>,
    arena_radius: Option<f64>,
    match_time_seconds: Option<f64>,
    terrain_max_amplitude: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct SpecCombatConstants {
    bow_max_draw_seconds: Option<f64>,
    axe_shield_disable_seconds: Option<f64>,
    swap_lockout_seconds: Option<f64>,
    hurt_invulnerability_seconds: Option<f64>,
}

impl Spec {
    fn into_consts(self) -> Consts {
        let d = Consts::default();
        let s = self.sim_constants;
        let cc = self.combat_constants;
        Consts {
            max_hp: s.max_hp.unwrap_or(d.max_hp),
            arena_radius: s.arena_radius.unwrap_or(d.arena_radius),
            match_time_seconds: s.match_time_seconds.unwrap_or(d.match_time_seconds),
            terrain_max_amplitude: s.terrain_max_amplitude.unwrap_or(d.terrain_max_amplitude),
            attack_recharge_ticks: d.attack_recharge_ticks,
            block_view_size: self.block_view_size.unwrap_or(d.block_view_size),
            max_observed_enemies: self.max_observed_enemies.unwrap_or(d.max_observed_enemies),
            max_observed_teammates: self
                .max_observed_teammates
                .unwrap_or(d.max_observed_teammates),
            max_observed_projectiles: self
                .max_observed_projectiles
                .unwrap_or(d.max_observed_projectiles),
            bow_max_draw_seconds: cc.bow_max_draw_seconds.unwrap_or(d.bow_max_draw_seconds),
            axe_shield_disable_seconds: cc
                .axe_shield_disable_seconds
                .unwrap_or(d.axe_shield_disable_seconds),
            swap_lockout_seconds: cc.swap_lockout_seconds.unwrap_or(d.swap_lockout_seconds),
            hurt_invulnerability_seconds: cc
                .hurt_invulnerability_seconds
                .unwrap_or(d.hurt_invulnerability_seconds),
            input_order: match self.input_order.as_deref() {
                Some("modern") => InputOrder::Modern,
                _ => InputOrder::Legacy,
            },
            lstm_hidden: self.arch.lstm_hidden,
        }
    }
}

/// How many ticks an applied action can lag its source observation before
/// the tick loop starts warning about inference falling behind.
const STALE_TICK_WARN: u64 = 8;

#[derive(Clone, Component)]
struct State {
    consts: Arc<Consts>,
    /// Azalea's `Client` doesn't expose a "current look direction" getter
    /// separate from what we last told it, so we track our own commanded
    /// yaw/pitch here (in radians, matching the training sim's convention)
    /// and update it every tick from the inference server's deltas.
    look: Arc<std::sync::Mutex<(f64, f64)>>,
    ticks: Arc<AtomicU64>,
    /// This tick's freshest observation, handed to the async inference
    /// worker (see `mod inference`).
    obs_tx: Arc<inference::ObsSender>,
    /// Latest action the worker has produced.
    action: ActionCell,
    /// Client-side legality guard (see `mod guard`).
    guard: Arc<Mutex<Guard>>,
    /// Game state rebuilt from packets + the bot's own inputs (see
    /// `mod tracker`) - team membership, other players' held item, and the
    /// self-timers behind `self_hurt` / `self_shield_disabled` /
    /// `self_bow_draw` / `self_swap_lockout`.
    trk: Arc<Mutex<Tracker>>,
    /// This bot's own network entity id, learned on spawn - needed to tell
    /// its own hurt animation apart from everyone else's.
    my_id: Arc<Mutex<Option<MinecraftEntityId>>>,
    /// Tick of the last inventory-screen hotbar hotkey, so the policy can't
    /// strobe the inventory open and closed (see `apply_held_slot`).
    last_hotkey_tick: Arc<AtomicU64>,
    /// Bumped on `Event::Death` - the closest thing this bot has to a
    /// training-side episode boundary. The inference worker (`mod
    /// inference`) watches this to zero its carried LSTM `(h, c)` state for
    /// a recurrent (`--lstm`) policy, the same point the sim zeros it
    /// during training. A no-op for the default memoryless MLP policy.
    episode_gen: Arc<AtomicU64>,
}

impl Default for State {
    /// Azalea requires client state to implement `Default` (it's used
    /// internally when spawning entities before `set_state`'s value is
    /// applied); the real values always come from `set_state` in `main`.
    fn default() -> Self {
        State {
            consts: Arc::new(Consts::default()),
            look: Arc::new(std::sync::Mutex::new((0.0, 0.0))),
            ticks: Arc::new(AtomicU64::new(0)),
            obs_tx: Arc::new(watch::channel(None).0),
            action: ActionCell::default(),
            guard: Arc::new(Mutex::new(Guard::new(GuardConfig::default()))),
            trk: Arc::new(Mutex::new(Tracker::new())),
            my_id: Arc::new(Mutex::new(None)),
            last_hotkey_tick: Arc::new(AtomicU64::new(0)),
            episode_gen: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Mirrors sim/src/protocol.rs::Action field-for-field.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Action {
    pub move_x: f64,
    pub move_z: f64,
    pub yaw_delta: f64,
    pub pitch_delta: f64,
    pub jump: bool,
    pub attack: bool,
    pub sprint: bool,
    #[serde(default)]
    pub use_item: bool,
    #[serde(default)]
    pub sneak: bool,
    /// Held-slot action: `0..HOTBAR_SLOTS` selects that physical slot;
    /// `HOTBAR_SLOTS..HOTBAR_ACTION_DIM` hotkeys `kit::Item` id
    /// `(held_slot - HOTBAR_SLOTS)` into the selected slot (mirrors
    /// `sim/src/kit.rs::HOTBAR_ACTION_DIM`).
    #[serde(default)]
    pub held_slot: i64,
    /// Only present for a recurrent (`--lstm`) policy (`consts.lstm_hidden >
    /// 0`) - see `inference_server.py`'s module docstring. `mod inference`
    /// carries this straight into the next `/act` request's `lstm_state`
    /// and never reads its contents itself.
    #[serde(default)]
    pub lstm_state: Option<LstmState>,
}

/// An LSTM `(h, c)` state, opaque to this bot - it only ever round-trips it
/// between `/act` responses and the next `/act` request, zeroed (by
/// omitting it) at episode start. See `inference_server.py`'s `lstm_state`
/// wire contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct LstmState {
    h: Vec<f32>,
    c: Vec<f32>,
}

/// One nearby player block, mirroring `OtherPlayer` in sim/src/protocol.rs.
#[derive(Debug, Default, Clone, Serialize)]
struct OtherObs {
    present: bool,
    hp: f64,
    rel_x: f64,
    rel_y: f64,
    rel_z: f64,
    vel_x: f64,
    vel_y: f64,
    vel_z: f64,
    ground_height: f64,
    /// 1.0 if actively blocking (a raised shield within its frontal arc).
    blocking: f64,
    eating: f64,
    held_ranged: f64,
    sneaking: f64,
}

/// One observed in-flight arrow / bolt (see `ProjectileObs`).
#[derive(Debug, Default, Clone, Serialize)]
struct ProjectileObs {
    present: bool,
    rel_x: f64,
    rel_y: f64,
    rel_z: f64,
    vel_x: f64,
    vel_y: f64,
    vel_z: f64,
}

/// One yaw-rotated block-grid column: `[top_rel, water, lava, cobweb]`,
/// matching `sim/src/protocol.rs::BlockColumnObs`. `top_rel` is already
/// normalised by `terrain_max_amplitude` (features.py passes it through).
#[derive(Debug, Default, Clone, Serialize)]
struct BlockColumnObs {
    top_rel: f64,
    water: f64,
    lava: f64,
    cobweb: f64,
}

/// Mirrors the fields python/features.py::observation_to_row expects (the
/// team-fight observation from sim/src/protocol.rs::Observation).
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Observation {
    self_hp: f64,
    self_vel_x: f64,
    self_vel_y: f64,
    self_vel_z: f64,
    self_yaw: f64,
    self_pitch: f64,
    self_on_ground: bool,
    self_attack_cooldown: f64,
    /// The server's own view of this bot's connection latency (ms), the
    /// same value shown next to its name in the vanilla tab list.
    self_ping_ms: f64,
    self_shield: f64,
    /// 20-tick hurt-invulnerability timer, restarted by the `HurtAnimation`
    /// / `DamageEvent` packet for this bot's entity (see `mod tracker`).
    self_hurt: f64,
    self_dist_from_center: f64,
    self_held: f64,
    /// Food level 0..20 from the client's own hunger bar.
    self_food: f64,
    self_sneaking: f64,
    self_absorption: f64,
    self_eating: f64,
    /// Bow draw fraction, from the tick count the bot has held right-click
    /// with a bow in hand (`mod tracker`).
    self_bow_draw: f64,
    self_burning: f64,
    /// Shield lock-out fraction, from the `Cooldown` packet naming the
    /// shield item (`mod tracker`).
    self_shield_disabled: f64,
    /// Absolute block-top height under the bot's feet, and the block-top
    /// delta one block forward / to the bot's right - scanned out of the
    /// loaded world, matching `sim/src/observation.rs`'s `self_ground_height`
    /// / `self_slope_forward` / `self_slope_right`.
    self_ground_height: f64,
    self_slope_forward: f64,
    self_slope_right: f64,
    self_arrows: f64,
    /// Currently selected physical hotbar slot.
    self_slot: f64,
    /// Post-swap attack/use lock-out fraction (modern input order only) -
    /// the bot counts down the ticks after each hotbar swap it makes.
    self_swap_lockout: f64,
    /// Block-break progress 0..1 (the sim's `uhc` pickaxe mining). This bot
    /// doesn't mine, so it always reports 0; the field exists to keep the
    /// wire row the same width as `sim/src/protocol.rs`.
    self_mining: f64,
    /// One float per `effects::Effect` (Speed, Slowness, Strength, Weakness,
    /// Regeneration, Poison, InstantHealth, InstantDamage, FireResistance):
    /// `amplifier + 1` while active, else 0. The two instant effects never
    /// persist so they're always 0.
    self_effects: Vec<f64>,
    /// Per-item counts (order = spec.json `inventory_items`).
    inventory: Vec<f64>,
    /// The `kit::Item` id in each physical hotbar slot (this server's kit).
    hotbar: Vec<f64>,
    /// Nearest living players split friend/foe by scoreboard team (see
    /// `mod tracker`); with no teams in play every other player is a foe.
    enemies: Vec<OtherObs>,
    teammates: Vec<OtherObs>,
    projectiles: Vec<ProjectileObs>,
    block_view: Vec<BlockColumnObs>,
    time_left: f64,
    enemies_alive: f64,
    teammates_alive: f64,
}

/// How to authenticate with the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthMode {
    /// Unauthenticated - works on offline/cracked servers and against a
    /// local ViaProxy (which does the real auth upstream itself).
    Offline,
    /// Microsoft device-code login; the token is cached under
    /// `~/.minecraft/azalea-auth.json` keyed by the username argument.
    Microsoft,
}

impl std::fmt::Display for AuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AuthMode::Offline => "offline",
            AuthMode::Microsoft => "microsoft",
        })
    }
}

/// Parsed command line. Positional args are still accepted in the original
/// order (`<server> [username] [inference_url]`) so `run_bot.sh` keeps
/// working; `--server` / `--username` / `--inference-url` / `--auth`
/// override them by name.
struct Cli {
    server: String,
    username: String,
    inference_url: String,
    auth: AuthMode,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> eyre::Result<Cli> {
        let mut server = None;
        let mut username = None;
        let mut inference_url = None;
        let mut auth = AuthMode::Offline;
        let mut positionals = Vec::new();

        let argv: Vec<String> = args.collect();
        let mut i = 0;
        while i < argv.len() {
            let arg = argv[i].clone();
            let mut value = || -> eyre::Result<String> {
                i += 1;
                argv.get(i)
                    .cloned()
                    .ok_or_else(|| eyre::eyre!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--server" => server = Some(value()?),
                "--username" | "--user" => username = Some(value()?),
                "--inference-url" => inference_url = Some(value()?),
                "--auth" => {
                    auth = match value()?.as_str() {
                        "offline" => AuthMode::Offline,
                        "microsoft" | "msa" => AuthMode::Microsoft,
                        other => eyre::bail!("--auth must be 'offline' or 'microsoft', got '{other}'"),
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "usage: azalea_bot [server] [username] [inference_url] \
                         [--auth offline|microsoft] [--server A] [--username N] [--inference-url U]\n\
                         \n\
                         For a server on a Minecraft version other than the one `azalea` is \
                         pinned to, run ViaProxy (see azalea-bot/README.md or run_bot.sh) and \
                         point --server at it."
                    );
                    std::process::exit(0);
                }
                s if s.starts_with("--") => eyre::bail!("unknown flag '{s}'"),
                _ => positionals.push(arg),
            }
            i += 1;
        }

        let mut pos = positionals.into_iter();
        Ok(Cli {
            server: server
                .or_else(|| pos.next())
                .unwrap_or_else(|| "localhost:25565".to_string()),
            username: username
                .or_else(|| pos.next())
                .unwrap_or_else(|| "TrainedBot".to_string()),
            inference_url: inference_url
                .or_else(|| pos.next())
                .unwrap_or_else(|| "http://127.0.0.1:8800/act".to_string()),
            auth,
        })
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse(std::env::args().skip(1))?;
    let server_address = cli.server;
    let bot_username = cli.username;
    let inference_url = cli.inference_url;

    info!(
        "connecting to {server_address} as '{bot_username}' ({} auth), using inference server at {inference_url}",
        cli.auth
    );

    // Inference runs off the tick loop now (see `mod inference`), so a slow
    // request no longer stalls the client - the timeout just caps how long
    // the worker waits before giving up and retrying next tick. Keep-alive
    // + TCP_NODELAY shave a few ms off every localhost round-trip.
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(250))
        .tcp_nodelay(true)
        .pool_idle_timeout(None)
        .build()?;

    let consts = fetch_spec(&http, &inference_url).await.unwrap_or_else(|e| {
        warn!("could not fetch /spec ({e}); using default sim constants - if you trained with a non-default config the live bot will normalise observations wrong");
        Consts::default()
    });
    info!("observation constants: {consts:?}");
    if consts.lstm_hidden > 0 {
        info!(
            "recurrent policy (lstm_hidden={}) - carrying LSTM state between ticks, reset on death",
            consts.lstm_hidden
        );
    }

    let guard_cfg = GuardConfig::from_env();
    info!("legality guard: {guard_cfg:?}");

    let (obs_tx, obs_rx) = watch::channel::<Option<(u64, Observation)>>(None);
    let action = ActionCell::default();
    let act_url = Arc::new(inference_url);
    let episode_gen = Arc::new(AtomicU64::new(0));
    tokio::spawn(inference::run_worker(
        http.clone(),
        act_url.clone(),
        obs_rx,
        action.clone(),
        episode_gen.clone(),
    ));

    let state = State {
        consts: Arc::new(consts),
        look: Arc::new(std::sync::Mutex::new((0.0, 0.0))),
        ticks: Arc::new(AtomicU64::new(0)),
        obs_tx: Arc::new(obs_tx),
        action,
        guard: Arc::new(std::sync::Mutex::new(Guard::new(guard_cfg))),
        trk: Arc::new(Mutex::new(Tracker::new())),
        my_id: Arc::new(Mutex::new(None)),
        last_hotkey_tick: Arc::new(AtomicU64::new(0)),
        episode_gen,
    };

    let account = match cli.auth {
        AuthMode::Offline => Account::offline(&bot_username),
        AuthMode::Microsoft => {
            info!(
                "authenticating with Microsoft (cache key '{bot_username}'); if this is the first \
                 time, a code + URL to open in a browser will be printed below"
            );
            Account::microsoft(&bot_username).await.map_err(|e| {
                eyre::eyre!(
                    "Microsoft auth failed: {e}. The cache key is the value passed as the username \
                     ('{bot_username}'); the account's in-game name is taken from the Microsoft \
                     profile, not that string."
                )
            })?
        }
    };

    let exit = ClientBuilder::new()
        .set_handler(handle)
        .set_state(state)
        .start(account, server_address.as_str())
        .await;
    info!("client exited: {exit:?}");

    Ok(())
}

/// `GET {base}/spec` - `inference_url` is the `/act` endpoint, so swap the
/// last path segment.
async fn fetch_spec(http: &reqwest::Client, inference_url: &str) -> eyre::Result<Consts> {
    let spec_url = match inference_url.rsplit_once('/') {
        Some((base, _)) => format!("{base}/spec"),
        None => format!("{inference_url}/spec"),
    };
    let spec: Spec = http.get(&spec_url).send().await?.error_for_status()?.json().await?;
    Ok(spec.into_consts())
}

async fn handle(bot: Client, event: Event, state: State) -> eyre::Result<()> {
    match event {
        Event::Login => {
            info!("logged in");
        }
        Event::Spawn => {
            // The bot's own network entity id is only knowable once it's in
            // the world; the tracker needs it to tell its own hurt animation
            // from everyone else's.
            let id = bot.minecraft_entity_by_ecs_entity(bot.entity);
            *state.my_id.lock().unwrap() = id;
            info!("spawned (entity id {id:?})");
        }
        Event::Packet(packet) => {
            let my_name = bot.username();
            let my_id = *state.my_id.lock().unwrap();
            state.trk.lock().unwrap().on_packet(&packet, &my_name, my_id);
        }
        Event::Tick => {
            let tick = state.ticks.fetch_add(1, Ordering::Relaxed);

            // 1. publish this tick's observation for the async worker.
            if let Some(observation) = build_observation(&bot, &state) {
                state.obs_tx.send_replace(Some((tick, observation)));
            }

            // 2. apply the freshest action the worker has produced. Missing
            //    only on the first few ticks before the first response.
            if let Some(decision) = state.action.latest() {
                let staleness = tick.saturating_sub(decision.obs_tick);
                if staleness > STALE_TICK_WARN && tick.is_multiple_of(40) {
                    warn!(
                        "inference is {staleness} ticks behind - is the model server keeping up?"
                    );
                }
                let safe = state
                    .guard
                    .lock()
                    .unwrap()
                    .sanitize(&bot, &state, &decision.action);
                apply_action(&bot, &state, &safe, tick);
            }
        }
        Event::Death(_) => {
            // `AutoRespawnPlugin` (bundled by `ClientBuilder::new`) already
            // sends the respawn packet. What needs resetting is our own
            // tracked look state: after a respawn the server resets the
            // player's orientation, so the stale tracked yaw/pitch would
            // feed the policy a wrong `self_yaw`/`self_pitch` until the next
            // `set_direction` corrected it.
            *state.look.lock().unwrap() = (0.0, 0.0);
            state.guard.lock().unwrap().on_respawn();
            // Same episode boundary the sim zeros a recurrent policy's LSTM
            // state at (see `State::episode_gen`'s doc comment) - a no-op
            // for the default non-recurrent policy.
            state.episode_gen.fetch_add(1, Ordering::Relaxed);
            info!("died - respawning automatically, reset tracked look direction + LSTM state");
        }
        Event::Disconnect(reason) => {
            // The server dropped us - a kick (anticheat, whitelist, a
            // protocol-version mismatch), a restart, or the socket dying.
            // azalea's bundled `AutoReconnectPlugin` will retry in ~5 s;
            // logging the reason here just stops the bot from going silently
            // frozen with no explanation (the failure mode the module doc
            // at the top of this file warns about).
            match reason {
                Some(text) => warn!("disconnected by the server: {text} - auto-reconnecting in ~5s"),
                None => warn!(
                    "disconnected from the server (no reason given - often a protocol/version \
                     mismatch or the socket dropping; see the note at the top of main.rs) - \
                     auto-reconnecting in ~5s"
                ),
            }
        }
        _ => {}
    }
    Ok(())
}

fn build_observation(bot: &Client, state: &State) -> Option<Observation> {
    let c = *state.consts;
    let self_pos = azalea::Vec3::from(&bot.get_component::<Position>()?);
    let self_physics = bot.get_component::<azalea::entity::Physics>()?;
    let self_vel = self_physics.velocity;
    let self_on_ground = self_physics.on_ground();
    let (self_yaw, self_pitch) = *state.look.lock().unwrap();

    let using_item = bot.get_component::<AbstractLivingUsingItem>().map(|u| u.0).unwrap_or(false);
    let (inventory, hotbar, arrows, selected_slot, held_id) = read_inventory(bot);
    let held = Item::from_id(held_id);

    // Advance the derived-state timers once per tick, now that we know
    // whether a bow is being drawn (see `mod tracker`).
    let drawing_bow = using_item && held == Item::Bow;
    {
        let mut trk = state.trk.lock().unwrap();
        trk.advance(drawing_bow);
    }

    let dist_from_center = (self_pos.x * self_pos.x + self_pos.z * self_pos.z).sqrt();

    let (sin_y, cos_y) = self_yaw.sin_cos();
    let rot = |dx: f64, dz: f64| (dx * cos_y - dz * sin_y, dx * sin_y + dz * cos_y);

    // Terrain under and around the bot (sim convention: forward = (-sin, cos),
    // right = (cos, sin); `slope_sample_distance` defaults to 1.0 block).
    let self_ground = surface_y_at(bot, self_pos.x, self_pos.z, self_pos.y);
    let ground_forward = surface_y_at(bot, self_pos.x - sin_y, self_pos.z + cos_y, self_pos.y);
    let ground_right = surface_y_at(bot, self_pos.x + cos_y, self_pos.z + sin_y, self_pos.y);

    // Nearest players, split into foes and allies by scoreboard team (see
    // `mod tracker`; with no teams in play everyone is a foe). Each is
    // yaw-rotated into the bot's frame and its held item resolved from the
    // last `SetEquipment` we saw for it.
    let make_other = |p: &PlayerSnapshot| -> OtherObs {
        let (rx, rz) = rot(p.pos.x - self_pos.x, p.pos.z - self_pos.z);
        let mainhand = p
            .mc_id
            .and_then(|id| state.trk.lock().unwrap().mainhand_of(id))
            .map(Item::from_kind);
        let held_ranged = matches!(mainhand, Some(Item::Bow) | Some(Item::Crossbow));
        let eating = p.using_item && matches!(mainhand, Some(m) if m.is_food());
        // Sim `blocking` is a raised shield; the live signal is "hand active"
        // with a shield-ish item (or, for the shield-less sword kit, just
        // hand active - the mainhand is always a sword there).
        let blocking = p.using_item
            && mainhand.map(|m| m.is_shieldish()).unwrap_or(true);
        OtherObs {
            present: true,
            hp: p.hp,
            rel_x: rx,
            rel_y: p.pos.y - self_pos.y,
            rel_z: rz,
            vel_x: p.vel.x,
            vel_y: p.vel.y,
            vel_z: p.vel.z,
            ground_height: surface_y_at(bot, p.pos.x, p.pos.z, p.pos.y),
            blocking: if blocking { 1.0 } else { 0.0 },
            eating: if eating { 1.0 } else { 0.0 },
            held_ranged: if held_ranged { 1.0 } else { 0.0 },
            sneaking: if p.sneaking { 1.0 } else { 0.0 },
        }
    };

    let all_players = nearest_players(bot, state);
    let mut enemies: Vec<OtherObs> = all_players
        .iter()
        .filter(|p| p.relation == Relation::Enemy)
        .take(c.max_observed_enemies)
        .map(&make_other)
        .collect();
    let mut teammates: Vec<OtherObs> = all_players
        .iter()
        .filter(|p| p.relation == Relation::Teammate)
        .take(c.max_observed_teammates)
        .map(&make_other)
        .collect();
    let enemies_alive = enemies.len() as f64;
    let teammates_alive = teammates.len() as f64;
    enemies.resize_with(c.max_observed_enemies, OtherObs::default);
    teammates.resize_with(c.max_observed_teammates, OtherObs::default);

    let projectiles = nearest_arrows(bot, self_pos, &rot, c.max_observed_projectiles);
    let block_view = block_column_view(bot, self_pos, self_yaw, c.block_view_size, c.terrain_max_amplitude);

    // The four fields the 1.21.11 protocol doesn't hand a client directly,
    // reconstructed by `mod tracker` (packets + the bot's own inputs).
    let (t_hurt, t_shield_disabled, t_bow_draw, t_swap_lockout) = {
        let trk = state.trk.lock().unwrap();
        (
            trk.self_hurt(c.hurt_invulnerability_seconds),
            trk.self_shield_disabled(c.axe_shield_disable_seconds),
            trk.self_bow_draw(c.bow_max_draw_seconds),
            trk.self_swap_lockout(c.swap_lockout_seconds),
        )
    };

    Some(Observation {
        self_hp: bot.get_component::<Health>()?.0 as f64,
        self_vel_x: self_vel.x,
        self_vel_y: self_vel.y,
        self_vel_z: self_vel.z,
        self_yaw,
        self_pitch,
        self_on_ground,
        self_attack_cooldown: (1.0
            - bot.attack_cooldown_remaining_ticks() as f64 / c.attack_recharge_ticks)
            .clamp(0.0, 1.0),
        self_ping_ms: bot
            .component::<TabList>()
            .get(&bot.uuid())
            .map(|info| info.latency as f64)
            .unwrap_or(0.0),
        self_shield: if using_item && held.is_shieldish() { 1.0 } else { 0.0 },
        self_hurt: t_hurt,
        self_food: bot.get_component::<azalea::local_player::Hunger>().map(|h| h.food as f64).unwrap_or(20.0),
        self_sneaking: if bot.crouching() { 1.0 } else { 0.0 },
        self_dist_from_center: dist_from_center / c.arena_radius,
        self_held: held_id as f64,
        self_absorption: bot
            .get_component::<PlayerAbsorption>()
            .map(|a| a.0 as f64)
            .unwrap_or(0.0),
        self_eating: if using_item && held.is_food() { 1.0 } else { 0.0 },
        self_bow_draw: t_bow_draw,
        self_burning: if bot.get_component::<OnFire>().map(|f| f.0).unwrap_or(false) {
            1.0
        } else {
            0.0
        },
        self_shield_disabled: t_shield_disabled,
        self_ground_height: self_ground,
        self_slope_forward: ground_forward - self_ground,
        self_slope_right: ground_right - self_ground,
        self_arrows: arrows as f64,
        self_slot: selected_slot as f64,
        self_swap_lockout: t_swap_lockout,
        self_mining: 0.0,
        self_effects: read_effects(bot),
        inventory,
        hotbar,
        enemies,
        teammates,
        projectiles,
        block_view,
        time_left: c.match_time_seconds,
        enemies_alive,
        teammates_alive,
    })
}

/// The `kit::Item` ids from sim/src/kit.rs - used for `self_held`, the
/// `hotbar` layout block and the per-item `inventory` counts. Order is
/// load-bearing (it's the on-wire id).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Item {
    Empty = 0,
    Sword = 1,
    Axe = 2,
    Pickaxe = 3,
    Bow = 4,
    Crossbow = 5,
    Planks = 6,
    Cobweb = 7,
    WaterBucket = 8,
    LavaBucket = 9,
    GoldenApple = 10,
    GoldenHead = 11,
    SplashHealing = 12,
    SplashHarming = 13,
    SplashPoison = 14,
    SplashSpeed = 15,
    SplashStrength = 16,
}

impl Item {
    fn from_id(id: usize) -> Item {
        match id {
            1 => Item::Sword,
            2 => Item::Axe,
            3 => Item::Pickaxe,
            4 => Item::Bow,
            5 => Item::Crossbow,
            6 => Item::Planks,
            7 => Item::Cobweb,
            8 => Item::WaterBucket,
            9 => Item::LavaBucket,
            10 => Item::GoldenApple,
            11 => Item::GoldenHead,
            12 => Item::SplashHealing,
            13 => Item::SplashHarming,
            14 => Item::SplashPoison,
            15 => Item::SplashSpeed,
            16 => Item::SplashStrength,
            _ => Item::Empty,
        }
    }

    /// Maps a live Minecraft item to the training kit's item id, or `Empty`
    /// for anything the kit doesn't model.
    fn from_kind(kind: ItemKind) -> Item {
        match kind {
            ItemKind::DiamondSword | ItemKind::NetheriteSword => Item::Sword,
            ItemKind::DiamondAxe | ItemKind::NetheriteAxe => Item::Axe,
            ItemKind::DiamondPickaxe | ItemKind::NetheritePickaxe => Item::Pickaxe,
            ItemKind::Bow => Item::Bow,
            ItemKind::Crossbow => Item::Crossbow,
            ItemKind::OakPlanks
            | ItemKind::SprucePlanks
            | ItemKind::BirchPlanks
            | ItemKind::JunglePlanks
            | ItemKind::AcaciaPlanks
            | ItemKind::DarkOakPlanks => Item::Planks,
            ItemKind::Cobweb => Item::Cobweb,
            ItemKind::WaterBucket => Item::WaterBucket,
            ItemKind::LavaBucket => Item::LavaBucket,
            ItemKind::GoldenApple => Item::GoldenApple,
            ItemKind::EnchantedGoldenApple => Item::GoldenHead,
            _ => Item::Empty,
        }
    }

    fn id(self) -> usize {
        self as usize
    }

    fn is_food(self) -> bool {
        matches!(self, Item::GoldenApple | Item::GoldenHead)
    }

    /// A held item whose right-click raises a guard. The bot can't see the
    /// off-hand shield separately, so "holding a sword and right-clicking"
    /// is the closest live signal for a raised shield.
    fn is_shieldish(self) -> bool {
        matches!(self, Item::Sword | Item::Axe | Item::Empty)
    }
}

fn is_arrow_item(kind: ItemKind) -> bool {
    matches!(kind, ItemKind::Arrow | ItemKind::SpectralArrow | ItemKind::TippedArrow)
}

/// The `self_effects` block: `amplifier + 1` per `effects::Effect` in order
/// (Speed, Slowness, Strength, Weakness, Regeneration, Poison, InstantHealth,
/// InstantDamage, FireResistance), else 0. Instant effects never persist.
fn read_effects(bot: &Client) -> Vec<f64> {
    use azalea::registry::builtin::MobEffect;
    let active = bot.get_component::<azalea::entity::ActiveEffects>();
    let lvl = |e: MobEffect| -> f64 {
        active
            .as_ref()
            .and_then(|a| a.get_level(e))
            .map_or(0.0, |amp| amp as f64 + 1.0)
    };
    vec![
        lvl(MobEffect::Speed),
        lvl(MobEffect::Slowness),
        lvl(MobEffect::Strength),
        lvl(MobEffect::Weakness),
        lvl(MobEffect::Regeneration),
        lvl(MobEffect::Poison),
        0.0, // instant_health
        0.0, // instant_damage
        lvl(MobEffect::FireResistance),
    ]
}

/// Reads the live inventory: `(inventory counts, hotbar layout[9],
/// total arrows, selected slot, held item id)`.
fn read_inventory(bot: &Client) -> (Vec<f64>, Vec<f64>, u32, usize, usize) {
    let mut counts = vec![0.0f64; INVENTORY_ITEMS];
    let mut hotbar = vec![0.0f64; HOTBAR_SLOTS];
    let mut arrows = 0u32;

    let inv = match bot.get_component::<azalea::entity::inventory::Inventory>() {
        Some(inv) => inv,
        None => return (counts, hotbar, 0, 0, 0),
    };
    let menu = inv.inventory_menu;
    let selected = inv.selected_hotbar_slot as usize;

    let stack_kind = |s: &ItemStack| -> Option<(ItemKind, i32)> {
        s.as_present().map(|d| (d.kind, d.count))
    };

    // Whole-inventory pass: per-kit item counts + arrows.
    for slot in menu.slots() {
        if let Some((kind, n)) = stack_kind(&slot) {
            if is_arrow_item(kind) {
                arrows += n.max(0) as u32;
            }
            let item = Item::from_kind(kind);
            if item != Item::Empty {
                let idx = item.id() - 1; // inventory block omits `Empty`
                if idx < counts.len() {
                    counts[idx] += n.max(0) as f64;
                }
            }
        }
    }

    // Hotbar layout + which item is in the selected slot.
    let hotbar_range = menu.hotbar_slots_range();
    let mut held_id = 0usize;
    for (i, menu_slot) in hotbar_range.clone().enumerate() {
        if i >= HOTBAR_SLOTS {
            break;
        }
        let item = menu
            .slot(menu_slot)
            .and_then(&stack_kind)
            .map(|(kind, _)| Item::from_kind(kind))
            .unwrap_or(Item::Empty);
        hotbar[i] = item.id() as f64;
        if i == selected {
            held_id = item.id();
        }
    }

    (counts, hotbar, arrows, selected.min(HOTBAR_SLOTS - 1), held_id)
}

/// The y of the topmost full-collision block at or below `(x, z)` near
/// `feet_y` - the sim's `support_y` (its terrain observation fields are
/// integer block tops now, so match that here rather than reporting 0).
fn surface_y_at(bot: &Client, x: f64, z: f64, feet_y: f64) -> f64 {
    let world = bot.world();
    let world = world.read();
    let (bx, bz) = (x.floor() as i32, z.floor() as i32);
    let fy = feet_y.floor() as i32;
    for y in (fy - 8..=fy + 4).rev() {
        let bs = world
            .get_block_state(BlockPos::new(bx, y, bz))
            .unwrap_or(BlockState::AIR);
        if bs.is_collision_shape_full() {
            return (y + 1) as f64;
        }
    }
    (fy - 8) as f64
}

/// Nearest other players, nearest first, each tagged friend/foe by
/// scoreboard team (see `mod tracker`). azalea does receive `Health`
/// metadata for tracked entities on many servers, but it defaults to full
/// until the first update - treat it as best-effort. `sneaking` is the
/// entity's shift-key-down metadata bit; `using_item` is the broadcast
/// "hand active" bit.
fn nearest_players(bot: &Client, state: &State) -> Vec<PlayerSnapshot> {
    bot.nearest_entities_by::<&Position, (With<PlayerMarker>, Without<LocalEntity>)>(|_: &Position| {
        true
    })
    .into_iter()
    .filter_map(|e| {
        let pos = bot.get_entity_component::<Position>(e)?;
        let physics = bot.get_entity_component::<azalea::entity::Physics>(e)?;
        let hp = bot
            .get_entity_component::<Health>(e)
            .map(|h| h.0 as f64)
            .unwrap_or(20.0);
        let sneaking = bot
            .get_entity_component::<AbstractEntityShiftKeyDown>(e)
            .map(|s| s.0)
            .unwrap_or(false);
        let using_item = bot
            .get_entity_component::<AbstractLivingUsingItem>(e)
            .map(|u| u.0)
            .unwrap_or(false);
        let name = bot
            .get_entity_component::<GameProfileComponent>(e)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let relation = state.trk.lock().unwrap().relation(&name);
        Some(PlayerSnapshot {
            pos: azalea::Vec3::from(&pos),
            vel: physics.velocity,
            hp,
            sneaking,
            using_item,
            mc_id: bot.minecraft_entity_by_ecs_entity(e),
            relation,
        })
    })
    .collect()
}

/// A nearby player as read from live entity state (see [`nearest_players`]).
struct PlayerSnapshot {
    pos: azalea::Vec3,
    vel: azalea::Vec3,
    hp: f64,
    sneaking: bool,
    using_item: bool,
    /// Network entity id, for looking this player's held item up in the
    /// tracker's equipment map.
    mc_id: Option<MinecraftEntityId>,
    relation: Relation,
}

/// Nearest in-flight arrows, nearest first, position + velocity rotated
/// into the bot's yaw frame - the same ordering `arena.rs::build_observation`
/// produces.
fn nearest_arrows(
    bot: &Client,
    self_pos: azalea::Vec3,
    rot: &impl Fn(f64, f64) -> (f64, f64),
    max: usize,
) -> Vec<ProjectileObs> {
    let mut out: Vec<ProjectileObs> = bot
        .nearest_entities_by::<&Position, With<AbstractArrow>>(|_: &Position| true)
        .into_iter()
        .filter_map(|e| {
            let pos = azalea::Vec3::from(&bot.get_entity_component::<Position>(e)?);
            let vel = bot.get_entity_component::<azalea::entity::Physics>(e)?.velocity;
            let (rx, rz) = rot(pos.x - self_pos.x, pos.z - self_pos.z);
            let (vrx, vrz) = rot(vel.x, vel.z);
            Some(ProjectileObs {
                present: true,
                rel_x: rx,
                rel_y: pos.y - self_pos.y,
                rel_z: rz,
                vel_x: vrx,
                vel_y: vel.y,
                vel_z: vrz,
            })
        })
        .take(max)
        .collect();
    out.resize_with(max, ProjectileObs::default);
    out
}

/// Port of `sim/src/blocks.rs::column_view` against the client's loaded
/// world: a `size` x `size` yaw-rotated grid (`col` = the player's right,
/// `row` = forward), one `[top_rel, water, lava, cobweb]` per column, in the
/// same row-major order the sim emits. `top_rel` is `(surface_y - pos.y)`
/// normalised by `terrain_max_amplitude`.
fn block_column_view(
    bot: &Client,
    pos: azalea::Vec3,
    yaw: f64,
    size: usize,
    max_amp: f64,
) -> Vec<BlockColumnObs> {
    if size == 0 {
        return Vec::new();
    }
    let world = bot.world();
    let world = world.read();
    let (sin_y, cos_y) = yaw.sin_cos();
    let half = (size / 2) as i32;
    let inv_amp = 1.0 / max_amp.max(1e-3);
    let feet_y = pos.y.floor() as i32;

    let mut out = Vec::with_capacity(size * size);
    for row in 0..size as i32 {
        for col in 0..size as i32 {
            let ox = (col - half) as f64;
            let oz = (row - half) as f64;
            // world offset = ox * right(yaw) + oz * forward(yaw),
            // right = (cos, sin), forward = (-sin, cos)  (sim convention).
            let wx = pos.x + ox * cos_y - oz * sin_y;
            let wz = pos.z + ox * sin_y + oz * cos_y;
            let (bx, bz) = (wx.floor() as i32, wz.floor() as i32);

            // Topmost solid block at/below head height, scanning a small
            // window (the sim's arenas are near-flat; a live server isn't,
            // so clamp the search rather than the whole column).
            let mut surface_y = feet_y - 8;
            for y in (feet_y - 8..=feet_y + 4).rev() {
                let bs = world
                    .get_block_state(BlockPos::new(bx, y, bz))
                    .unwrap_or(BlockState::AIR);
                if bs.is_collision_shape_full() {
                    surface_y = y + 1;
                    break;
                }
            }

            let mut water = 0.0;
            let mut lava = 0.0;
            let mut cobweb = 0.0;
            for y in feet_y - 1..=feet_y + 3 {
                let bp = BlockPos::new(bx, y, bz);
                match world.get_fluid_state(bp).map(|f| f.kind) {
                    Some(FluidKind::Water) => water = 1.0,
                    Some(FluidKind::Lava) => lava = 1.0,
                    _ => {}
                }
                if let Some(bs) = world.get_block_state(bp) {
                    if is_cobweb(bs) {
                        cobweb = 1.0;
                    }
                }
            }

            let top_rel = ((surface_y as f64 - pos.y) * inv_amp).clamp(-3.0, 3.0);
            out.push(BlockColumnObs { top_rel, water, lava, cobweb });
        }
    }
    out
}

fn is_cobweb(bs: BlockState) -> bool {
    use azalea::block::BlockTrait;
    Box::<dyn BlockTrait>::from(bs).id() == "cobweb"
}

/// Execute a fully-sanitized [`SafeAction`] through Azalea's client API.
/// Every decision (look integration, sprint legality, which entity to hit,
/// attack/use mutual exclusion) was already made in `guard::Guard::sanitize`
/// - this function only performs, it decides nothing.
fn apply_action(bot: &Client, state: &State, safe: &SafeAction, tick: u64) {
    // Sprinting is a distinct client state from just moving forward: the
    // server only grants the sprint-knockback bonus / cancels crits when
    // the client is actually flagged sprinting, and `bot.walk` clears that
    // flag - so call exactly one of the two.
    match safe.sprint {
        Some(sprint_dir) => bot.sprint(sprint_dir),
        None => bot.walk(safe.walk),
    }

    bot.set_direction(safe.yaw_deg, safe.pitch_deg);
    apply_held_slot(bot, state, safe.held_slot, tick);
    bot.set_crouching(safe.sneak);

    if safe.jump {
        bot.jump();
    }

    if let Some(entity) = safe.attack {
        bot.attack(entity);
    } else if safe.use_item {
        // Raise the shield / draw the bow / eat / place.
        bot.start_use_item();
    }
}

/// Nearest **enemy** players (scoreboard-team aware - see `mod tracker`),
/// nearest first, as raw entity ids for `bot.attack`. The legality guard
/// only ever swings at these, so the bot can't friendly-fire.
pub(crate) fn nearest_player_entities(bot: &Client, state: &State) -> Vec<Entity> {
    bot.nearest_entities_by::<&Position, (With<PlayerMarker>, Without<LocalEntity>)>(|_: &Position| {
        true
    })
    .into_iter()
    .filter(|&e| {
        let name = bot
            .get_entity_component::<GameProfileComponent>(e)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        state.trk.lock().unwrap().relation(&name) == Relation::Enemy
    })
    .collect()
}

/// Minimum ticks between two inventory-screen hotkeys, so the policy can't
/// strobe the screen open and closed (an obvious anticheat flag). Selecting
/// a slot the item is *already* on is unlimited - it's just a number key.
const HOTKEY_MIN_GAP_TICKS: u64 = 10;

/// Apply the policy's held-slot action, mirroring
/// `sim/src/kit.rs::HOTBAR_ACTION_DIM`: `0..HOTBAR_SLOTS` selects (presses)
/// that number key; `HOTBAR_SLOTS..` is a vanilla number-key hotkey for
/// `kit::Item` id `(a - HOTBAR_SLOTS)`.
///
/// The hotkey is resolved the least-invasive way possible: if the wanted
/// item is already sitting in some hotbar slot, just select that slot (no
/// container traffic at all). Only when it's buried in the main inventory
/// do we open the inventory screen, send the swap, and let the handle close
/// it - rate-limited by `HOTKEY_MIN_GAP_TICKS`. Any actual selected-slot
/// change starts the modern input-order swap lock-out via the tracker.
fn apply_held_slot(bot: &Client, state: &State, held_slot: i64, tick: u64) {
    let selected_now = bot
        .get_component::<azalea::entity::inventory::Inventory>()
        .map(|inv| inv.selected_hotbar_slot)
        .unwrap_or(0);

    let note_swap = |to: u8| {
        if to != selected_now {
            let ticks = state.consts.swap_lockout_ticks();
            state.trk.lock().unwrap().note_hotbar_swap(ticks);
        }
    };

    if (0..HOTBAR_SLOTS as i64).contains(&held_slot) {
        let want_slot = held_slot as u8;
        if want_slot != selected_now {
            bot.set_selected_hotbar_slot(want_slot);
            note_swap(want_slot);
        }
        return;
    }

    let item_id = (held_slot - HOTBAR_SLOTS as i64) as usize;
    let want = Item::from_id(item_id);
    if want == Item::Empty {
        return; // "empty hand" has no clean live equivalent - skip
    }

    // Already holdable with a plain number key?
    if let Some(slot) = bot.query_self::<&azalea::entity::inventory::Inventory, _>(|inv| {
        let menu = &inv.inventory_menu;
        let hotbar = menu.hotbar_slots_range();
        (0..HOTBAR_SLOTS).find(|&i| {
            menu.slot(*hotbar.start() + i)
                .and_then(|s| s.as_present())
                .map(|d| Item::from_kind(d.kind))
                == Some(want)
        })
    }) {
        let slot = slot as u8;
        if slot != selected_now {
            bot.set_selected_hotbar_slot(slot);
            note_swap(slot);
        }
        return;
    }

    // Buried in the main inventory: open the screen, swap, close.
    let last = state.last_hotkey_tick.load(Ordering::Relaxed);
    if tick.saturating_sub(last) < HOTKEY_MIN_GAP_TICKS {
        return;
    }
    let source_slot = bot.query_self::<&azalea::entity::inventory::Inventory, _>(|inv| {
        let menu = &inv.inventory_menu;
        let selected_menu_slot = *menu.hotbar_slots_range().start() + selected_now as usize;
        menu.slots()
            .iter()
            .position(|s| s.as_present().map(|d| Item::from_kind(d.kind)) == Some(want))
            .filter(|&s| s != selected_menu_slot)
    });
    let Some(source_slot) = source_slot else { return };

    if let Some(handle) = bot.open_inventory() {
        handle.click(SwapClick {
            source_slot: source_slot as u16,
            target_slot: selected_now,
        });
        // `handle` drops here, sending the close-container packet.
    }
    state.last_hotkey_tick.store(tick, Ordering::Relaxed);
    // The selected slot *index* is unchanged, but its contents just did -
    // that still costs the modern input-order lock-out.
    let ticks = state.consts.swap_lockout_ticks();
    state.trk.lock().unwrap().note_hotbar_swap(ticks);
}

/// Azalea's movement API is 8-directional (+ none), not a continuous
/// vector, so we discretize the policy's continuous move_x/move_z output
/// into the nearest of those 9 states.
pub(crate) fn discretize_walk_direction(move_x: f64, move_z: f64) -> WalkDirection {
    const DEAD_ZONE: f64 = 0.25;
    let forward = move_z > DEAD_ZONE;
    let backward = move_z < -DEAD_ZONE;
    let right = move_x > DEAD_ZONE;
    let left = move_x < -DEAD_ZONE;

    match (forward, backward, left, right) {
        (true, _, true, _) => WalkDirection::ForwardLeft,
        (true, _, _, true) => WalkDirection::ForwardRight,
        (true, _, _, _) => WalkDirection::Forward,
        (_, true, true, _) => WalkDirection::BackwardLeft,
        (_, true, _, true) => WalkDirection::BackwardRight,
        (_, true, _, _) => WalkDirection::Backward,
        (_, _, true, _) => WalkDirection::Left,
        (_, _, _, true) => WalkDirection::Right,
        _ => WalkDirection::None,
    }
}

/// Sprinting (like vanilla) only works while moving forward - there's no
/// such thing as sprinting backward or purely sideways.
pub(crate) fn sprint_direction_for(walk_dir: WalkDirection) -> Option<SprintDirection> {
    match walk_dir {
        WalkDirection::Forward => Some(SprintDirection::Forward),
        WalkDirection::ForwardLeft => Some(SprintDirection::ForwardLeft),
        WalkDirection::ForwardRight => Some(SprintDirection::ForwardRight),
        _ => None,
    }
}
