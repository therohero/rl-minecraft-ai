#!/usr/bin/env bash
# Starts the inference server that the Fabric client mod (mod/) connects to.
#
# It exports the latest training checkpoint to azalea-bot/model/ and then runs
# azalea-bot/inference_server.py in the foreground. Leave this running, launch
# Minecraft with the mod installed, and use /fight (or /fight train).
#
# Usage:
#   ./run_bot_mod.sh [port] [-- extra args passed to inference_server.py]
# Examples:
#   ./run_bot_mod.sh                 # serve on 127.0.0.1:8800 (the mod's default)
#   ./run_bot_mod.sh 8801
#   ./run_bot_mod.sh 8800 -- --sample --device cuda
#
# This does NOT train and does NOT run the headless azalea bot - it is just
# the model server. Train first with ./run.sh; run the headless bot with
# ./run_bot.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/training/python"
CHECKPOINT_DIR="$SCRIPT_DIR/training/checkpoints"
BRIDGE_DIR="$SCRIPT_DIR/azalea-bot"
MODEL_DIR="$BRIDGE_DIR/model"

# First arg is the port only if it's not an option; everything after (an
# optional "--" separator included) is forwarded to inference_server.py.
PORT="${INFERENCE_PORT:-8800}"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then PORT="$1"; shift; fi
if [ "${1:-}" = "--" ]; then shift; fi

log() { printf '[run_bot_mod.sh] %s\n' "$*"; }
die() { printf '[run_bot_mod.sh] ERROR: %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found - install Python 3.10+"

# Prefer the repo-local virtualenv that run.sh builds (its torch is the one
# the model was exported with); fall back to a plain python3.
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    PY="python3"
    log "no .venv found - using system 'python3' (run ./run.sh once to create it)"
fi

# Keep the .venv's torch matched to this machine (CPU vs CUDA wheel) - a fast
# no-op when it already is. Skipped for the system-python fallback.
if [ "$PY" != "python3" ]; then
    "$PY" "$PYTHON_DIR/ensure_deps.py" || die "Python dependency check failed"
fi

CHECKPOINT="$CHECKPOINT_DIR/latest.pt"
if [ -f "$CHECKPOINT" ]; then
    # Always re-export so the served model matches the current feature code.
    log "exporting $CHECKPOINT -> $MODEL_DIR ..."
    ( cd "$PYTHON_DIR" && "$PY" export_model.py --checkpoint "$CHECKPOINT" --out-dir "$MODEL_DIR" ) \
        || die "model export failed"
elif [ -f "$MODEL_DIR/policy.pt" ]; then
    log "no checkpoint at $CHECKPOINT - serving the model already in $MODEL_DIR"
else
    die "no checkpoint at $CHECKPOINT and no model in $MODEL_DIR - train first with ./run.sh"
fi

log "serving on http://127.0.0.1:$PORT/act - leave this running, then start Minecraft + the mod and use /fight"
exec "$PY" "$BRIDGE_DIR/inference_server.py" --model-dir "$MODEL_DIR" --port "$PORT" "$@"
