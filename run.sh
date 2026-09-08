#!/usr/bin/env bash
# One-shot entry point: builds the Rust simulation backend (release mode)
# if needed, ensures Python dependencies are installed, then starts
# self-play training. Ctrl+C stops it (train.py saves `latest.pt` first).
# Any arguments you pass are forwarded straight to train.py, e.g.:
#
#   ./run.sh                          # train with defaults (arena count auto-detected from CPU count, CPU/GPU auto)
#   ./run.sh --num-arenas 128         # override the arena count
#   ./run.sh --device cuda            # force GPU
#
# To actually use the trained bot, start the model server separately with
# `./run_bot_mod.sh` (for the mod/ Fabric client) or `./run_bot.sh` (for the
# headless azalea bot).
#
# Re-running this script is safe and fast: cargo/pip only do real work
# when something actually changed.
#
# Needs a bash - this runs as-is on Linux and macOS, and on Windows under
# WSL or Git Bash/MSYS2. Native Windows (PowerShell, no bash) users: run
# `run.ps1` instead, or follow the manual steps in README.md's Quickstart.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$SCRIPT_DIR/training/sim"
PYTHON_DIR="$SCRIPT_DIR/training/python"
# `cargo build` names the binary `mc_pvp_sim.exe` on Windows (including
# under Git Bash/MSYS2, which still uses the native Windows Rust/cargo
# toolchain) and plain `mc_pvp_sim` on Linux/macOS/WSL.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) SIM_BINARY="$SIM_DIR/target/release/mc_pvp_sim.exe" ;;
    *) SIM_BINARY="$SIM_DIR/target/release/mc_pvp_sim" ;;
esac

log() { printf '[run.sh] %s\n' "$*"; }
die() { printf '[run.sh] ERROR: %s\n' "$*" >&2; exit 1; }

command -v cargo  >/dev/null 2>&1 || die "cargo not found - install Rust: https://rustup.rs"
command -v python3 >/dev/null 2>&1 || die "python3 not found - install Python 3.10+"

log "building simulation backend (release mode)..."
( cd "$SIM_DIR" && cargo build --release ) || die "Rust build failed - see errors above"
[ -x "$SIM_BINARY" ] || die "build succeeded but binary not found at $SIM_BINARY"

# All Python work happens in a repo-local virtualenv (.venv/) so this never
# touches the system/global site-packages. Created once, reused after that.
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    log "creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || die "failed to create virtualenv - is the python3-venv package installed?"
fi
if [ -x "$VENV_DIR/bin/python" ]; then
    VENV_PY="$VENV_DIR/bin/python"           # Linux/macOS/WSL
else
    VENV_PY="$VENV_DIR/Scripts/python.exe"   # Git Bash / MSYS2 on Windows
fi
[ -x "$VENV_PY" ] || die "virtualenv python not found at $VENV_PY"

# Installs torch + numpy + pytest, picking the CPU-only or the CUDA torch
# wheel to match whatever machine this is (see training/python/ensure_deps.py).
# Fast no-op once the environment already matches. Override the choice with
# RL_TORCH_BACKEND=cpu|cuda if the autodetect ever guesses wrong.
log "checking Python dependencies (in .venv)..."
"$VENV_PY" "$PYTHON_DIR/ensure_deps.py" \
    || die "failed to install Python dependencies - try: $VENV_PY -m pip install -r training/python/requirements.txt"

log "starting training (Ctrl+C to stop; checkpoints are saved periodically to training/checkpoints/)..."
cd "$PYTHON_DIR"
exec "$VENV_PY" train.py --sim-binary "$SIM_BINARY" "$@"
