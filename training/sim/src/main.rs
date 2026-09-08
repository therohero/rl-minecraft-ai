//! Entry point for the high-speed Minecraft-style PvP simulation backend.
//!
//! Usage:
//!   mc_pvp_sim [OPTIONS] [port] [num_arenas] [seed]
//!
//! Options:
//!   -p, --port <PORT>        UDP port to listen on            (default 9999)
//!   -n, --arenas <N>         parallel arenas                  (default 64)
//!   -s, --seed <SEED>        base RNG seed                     (default: random)
//!   -c, --config <FILE>      JSON file of SimConfig overrides  (default: built-in)
//!       --dump-config        print the resolved config as JSON and exit
//!   -h, --help               show this message
//!
//! The three bare positional args are still accepted for backwards
//! compatibility (`mc_pvp_sim 9999 64 123`), but the flags take precedence.
//!
//! Defaults to port 9999, 64 parallel arenas, and a random seed (logged on
//! startup so the run can be reproduced later by passing it explicitly).
//! The simulation is fully uncoupled from wall-clock time: it steps as fast
//! as it can read actions and write states over the socket, with no
//! internal tick-rate throttle, and arenas are stepped in parallel across
//! CPU cores via rayon.

mod arena;
mod blocks;
mod collision;
mod combat;
mod config;
mod effects;
mod kit;
mod observation;
mod physics;
mod player;
mod projectile;
mod protocol;
mod server;
mod terrain;

use config::{Kit, SimConfig};

const HELP: &str = "mc_pvp_sim - high-speed PvP simulation backend

Usage: mc_pvp_sim [OPTIONS] [port] [num_arenas] [seed]

Options:
  -p, --port <PORT>      UDP port to listen on            (default 9999)
  -n, --arenas <N>       parallel arenas                  (default 64)
  -s, --seed <SEED>      base RNG seed                    (default: random)
  -t, --team-size <N>    players per team (1 = 1v1 duel)  (default 1)
  -k, --kit <KIT>        combat kit: sword | axe | uhc     (default sword)
  -c, --config <FILE>    JSON file of SimConfig overrides (default: built-in)
      --dump-config      print the resolved config as JSON and exit
  -h, --help             show this message

A partial config file only needs the keys it overrides, e.g.
  { \"arena_radius\": 20, \"terrain_flat_only\": true, \"reward\": { \"win\": 50 } }
Run with --dump-config to see every available key and its default.";

struct Args {
    port: u16,
    num_arenas: usize,
    seed: Option<u64>,
    team_size: Option<usize>,
    kit: Option<Kit>,
    config_path: Option<String>,
    dump_config: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut port: u16 = 9999;
    let mut num_arenas: usize = 64;
    let mut seed: Option<u64> = None;
    let mut team_size: Option<usize> = None;
    let mut kit: Option<Kit> = None;
    let mut config_path: Option<String> = None;
    let mut dump_config = false;
    let mut positionals: Vec<String> = Vec::new();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut take = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-p" | "--port" => port = take("--port")?.parse().map_err(|e| format!("bad --port: {e}"))?,
            "-n" | "--arenas" | "--num-arenas" => {
                num_arenas = take("--arenas")?.parse().map_err(|e| format!("bad --arenas: {e}"))?
            }
            "-s" | "--seed" => {
                seed = Some(take("--seed")?.parse().map_err(|e| format!("bad --seed: {e}"))?)
            }
            "-t" | "--team-size" => {
                team_size =
                    Some(take("--team-size")?.parse().map_err(|e| format!("bad --team-size: {e}"))?)
            }
            "-k" | "--kit" => {
                kit = Some(match take("--kit")?.to_lowercase().as_str() {
                    "sword" => Kit::Sword,
                    "axe" => Kit::Axe,
                    "uhc" => Kit::Uhc,
                    other => return Err(format!("bad --kit '{other}' (want sword|axe|uhc)")),
                })
            }
            "-c" | "--config" => config_path = Some(take("--config")?),
            "--dump-config" => dump_config = true,
            s if s.starts_with('-') => return Err(format!("unknown option: {s}")),
            s => positionals.push(s.to_string()),
        }
    }

    // Legacy positional form: port, num_arenas, seed - only used to fill a
    // value a flag didn't already set.
    if let Some(v) = positionals.first() {
        port = v.parse().map_err(|e| format!("bad positional port: {e}"))?;
    }
    if let Some(v) = positionals.get(1) {
        num_arenas = v.parse().map_err(|e| format!("bad positional num_arenas: {e}"))?;
    }
    if let Some(v) = positionals.get(2) {
        seed = Some(v.parse().map_err(|e| format!("bad positional seed: {e}"))?);
    }

    Ok(Args { port, num_arenas, seed, team_size, kit, config_path, dump_config })
}

fn main() -> std::io::Result<()> {
    // Timestamps + level on every line, defaulting to `info` so useful
    // startup/throughput/error logs show up without extra setup - set
    // RUST_LOG=debug (or =error, etc.) to change verbosity.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("argument error: {e}\nrun with --help for usage");
        std::process::exit(2);
    });

    let mut sim_config = match &args.config_path {
        Some(path) => SimConfig::load(path).unwrap_or_else(|e| {
            eprintln!("failed to load config '{path}': {e}");
            std::process::exit(2);
        }),
        None => SimConfig::default(),
    };
    if let Some(ts) = args.team_size {
        sim_config.team_size = ts;
    }
    if let Some(k) = args.kit {
        sim_config.kit = k;
    }
    sim_config.normalize();
    if let Err(e) = sim_config.validate() {
        eprintln!("invalid config: {e}");
        std::process::exit(2);
    }

    if args.dump_config {
        println!("{}", serde_json::to_string_pretty(&sim_config).unwrap());
        return Ok(());
    }

    log::info!(
        "sim config: {}",
        serde_json::to_string(&sim_config).unwrap_or_default()
    );
    let _ = config::install(sim_config);

    server::run(args.port, args.num_arenas, args.seed)
}
