#!/usr/bin/env bash
# Script to run the Minecraft bot on a real server.
# Usage:
#   ./run_bot.sh [ip] [port] [username] [inference_url] [mc_version]
# Example:
#   ./run_bot.sh play.example.com 25565 TrainedBot
#   ./run_bot.sh play.example.com 25565 TrainedBot "" 1.21.4   # via ViaProxy
#
# Environment:
#   MC_VERSION   server's Minecraft version - when set (or given as the 5th
#                arg) the bot connects through a local ViaProxy that
#                translates between it and the version `azalea` speaks. Needs
#                a JRE on PATH. ViaProxy is downloaded once into
#                azalea-bot/viaproxy/.
#   AUTH         'offline' (default) or 'microsoft' - passed to the bot as
#                --auth. With ViaProxy, leave this 'offline' (ViaProxy does
#                the upstream auth; see azalea-bot/viaproxy/viaproxy.yml).

set -euo pipefail

# Default arguments
IP="${1:-localhost}"
PORT="${2:-25565}"
USERNAME="${3:-TrainedBot}"
INFERENCE_URL="${4:-http://127.0.0.1:8800/act}"
MC_VERSION="${5:-${MC_VERSION:-}}"
AUTH="${AUTH:-offline}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$SCRIPT_DIR/azalea-bot/azalea_bot"
PYTHON_DIR="$SCRIPT_DIR/training/python"
BRIDGE_DIR="$SCRIPT_DIR/azalea-bot"
VIAPROXY_DIR="$BRIDGE_DIR/viaproxy"
VIAPROXY_PORT=25568

log() { printf '[run_bot.sh] %s\n' "$*"; }
die() { printf '[run_bot.sh] ERROR: %s\n' "$*" >&2; exit 1; }

# Check dependencies
command -v cargo  >/dev/null 2>&1 || die "cargo not found - install Rust: https://rustup.rs"
command -v python3 >/dev/null 2>&1 || die "python3 not found - install Python 3.10+"

# Prefer the repo-local virtualenv `run.sh` builds (its torch is the one the
# model was exported with); fall back to a plain `python3` if there's no venv.
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/.venv/bin/python"                # Linux/macOS/WSL
    log "using the repo virtualenv: $PY"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PY="$SCRIPT_DIR/.venv/Scripts/python.exe"        # Git Bash / MSYS2 on Windows
    log "using the repo virtualenv: $PY"
else
    PY="python3"
    log "no .venv found - using system 'python3' (run ./run.sh once to create the virtualenv)"
fi

# Keep the .venv's torch matched to this machine (CPU vs CUDA wheel) - a fast
# no-op when it already is. Skipped for the system-python fallback.
if [ "$PY" != "python3" ]; then
    "$PY" "$PYTHON_DIR/ensure_deps.py" || die "Python dependency check failed"
fi

# 1. Ensure a model is exported
if [ ! -f "$BRIDGE_DIR/model/policy.pt" ]; then
    log "Model policy.pt not found. Attempting to export from latest checkpoint..."
    CHECKPOINT="$SCRIPT_DIR/training/checkpoints/latest.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        die "No training checkpoint found at $CHECKPOINT. Please run training first to generate a model."
    fi
    log "Exporting checkpoint $CHECKPOINT..."
    ( cd "$PYTHON_DIR" && "$PY" export_model.py --checkpoint "$CHECKPOINT" --out-dir "$BRIDGE_DIR/model" ) \
        || die "Failed to export model checkpoint."
fi

# 2. Check if inference server is already running on port 8800
INFERENCE_PORT=8800
port_open() { local p="${1:-$INFERENCE_PORT}"; "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', $p))==0 else 1)"; }
log "Checking if inference server is already running on port $INFERENCE_PORT..."
if port_open "$INFERENCE_PORT"; then
    log "Inference server is already running on port $INFERENCE_PORT. Reusing it."
    SERVER_PID=""
else
    log "Starting inference server in the background..."
    "$PY" "$BRIDGE_DIR/inference_server.py" --model-dir "$BRIDGE_DIR/model" --port $INFERENCE_PORT > "$BRIDGE_DIR/inference_server.log" 2>&1 &
    SERVER_PID=$!

    # Clean up the background process if this script is terminated or exits
    trap 'if [ -n "${SERVER_PID:-}" ]; then log "Stopping inference server (PID $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT

    # Wait for inference server to start up
    log "Waiting for inference server to start..."
    for _ in $(seq 1 20); do
        if port_open "$INFERENCE_PORT"; then
            log "Inference server successfully started."
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            cat "$BRIDGE_DIR/inference_server.log" >&2
            die "Inference server failed to start. See log at $BRIDGE_DIR/inference_server.log"
        fi
        sleep 0.5
    done
    port_open "$INFERENCE_PORT" || { cat "$BRIDGE_DIR/inference_server.log" >&2; die "Inference server did not come up within 10s. See $BRIDGE_DIR/inference_server.log"; }
fi

# 3. Optionally start ViaProxy to translate to the server's Minecraft version.
SERVER_ADDR="$IP:$PORT"
if [ -n "$MC_VERSION" ]; then
    command -v java >/dev/null 2>&1 || die "MC_VERSION is set but 'java' is not on PATH - ViaProxy needs a JRE (17+)."
    mkdir -p "$VIAPROXY_DIR"
    JAR="$VIAPROXY_DIR/ViaProxy.jar"
    if [ ! -f "$JAR" ]; then
        log "Downloading ViaProxy (one-time) into $VIAPROXY_DIR ..."
        URL="$(curl -fsSL https://api.github.com/repos/ViaVersion/ViaProxy/releases/latest \
               | "$PY" -c "import sys,json; a=json.load(sys.stdin)['assets']; print(next(x['browser_download_url'] for x in a if x['name'].endswith('.jar') and 'java8' not in x['name']))")" \
            || die "could not resolve the ViaProxy download URL"
        curl -fsSL -o "$JAR" "$URL" || die "ViaProxy download failed ($URL)"
    fi
    # ViaProxy reads viaproxy.yml from its working dir when run headless.
    cat > "$VIAPROXY_DIR/viaproxy.yml" <<EOF
bind-address: 127.0.0.1:$VIAPROXY_PORT
target-address: $SERVER_ADDR
target-version: $MC_VERSION
auth-method: NONE
proxy-online-mode: false
EOF
    log "Starting ViaProxy: $SERVER_ADDR ($MC_VERSION) -> 127.0.0.1:$VIAPROXY_PORT"
    ( cd "$VIAPROXY_DIR" && java -jar "$JAR" ) > "$VIAPROXY_DIR/viaproxy.log" 2>&1 &
    VIAPROXY_PID=$!
    trap 'kill "$VIAPROXY_PID" 2>/dev/null || true; if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi' EXIT
    for _ in $(seq 1 60); do
        port_open "$VIAPROXY_PORT" && break
        kill -0 "$VIAPROXY_PID" 2>/dev/null || { cat "$VIAPROXY_DIR/viaproxy.log" >&2; die "ViaProxy exited during startup"; }
        sleep 0.5
    done
    port_open "$VIAPROXY_PORT" || { cat "$VIAPROXY_DIR/viaproxy.log" >&2; die "ViaProxy did not open $VIAPROXY_PORT within 30s"; }
    log "ViaProxy is up."
    SERVER_ADDR="127.0.0.1:$VIAPROXY_PORT"
fi

# 4. Connect the bot
log "Connecting the bot to $SERVER_ADDR as '$USERNAME' (auth: $AUTH)..."
cd "$BOT_DIR"
cargo run --release -- "$SERVER_ADDR" "$USERNAME" "$INFERENCE_URL" --auth "$AUTH"
