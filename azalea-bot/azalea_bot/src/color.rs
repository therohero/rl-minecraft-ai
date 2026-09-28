//! Terminal coloring for this bot's console output: a compact colored log
//! format (dim timestamp, colored level tag, dim target) plus a `paint`
//! helper for highlighting the handful of connection-lifecycle lines
//! (connect/login/spawn/death/exit) that matter most when watching a live
//! session. Auto-disables under `NO_COLOR` or when stdout isn't a terminal
//! (e.g. piped to a log file), so redirected output stays plain ASCII.

use std::io::{IsTerminal, Write};
use std::sync::OnceLock;

use env_logger::fmt::style::{AnsiColor, Style};

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal())
}

/// Wraps `s` in `color` when the terminal supports it, otherwise returns it
/// unchanged. Used for the handful of connection-lifecycle log lines below
/// `init_logger`'s per-level coloring, to make state changes (connecting,
/// spawned, died, disconnected) easy to spot while scrolling past routine
/// per-tick logs.
pub fn paint(color: AnsiColor, s: &str) -> String {
    if enabled() {
        let style = color.on_default();
        format!("{style}{s}{style:#}")
    } else {
        s.to_string()
    }
}

/// Installs an `env_logger` with a compact, colored format: a dim
/// timestamp, a colored level tag (green info / yellow warn / bold red
/// error, via `env_logger`'s own auto-detection), a dim module target, then
/// the message.
pub fn init_logger() {
    let color = enabled();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(move |buf, record| {
            let ts = buf.timestamp_seconds();
            if color {
                let level_style = buf.default_level_style(record.level());
                let dim = Style::new().dimmed();
                writeln!(
                    buf,
                    "{dim}{ts}{dim:#} {level_style}{:>5}{level_style:#} {dim}{}:{dim:#} {}",
                    record.level(),
                    record.target(),
                    record.args(),
                )
            } else {
                writeln!(buf, "{ts} {:>5} {}: {}", record.level(), record.target(), record.args())
            }
        })
        .init();
}
