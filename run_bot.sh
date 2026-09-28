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
#                --auth. Without MC_VERSION (connecting natively) this is the
#                bot's own auth. With MC_VERSION (through ViaProxy) the bot
#                always connects to the *local* ViaProxy as 'offline' -
#                ViaProxy is what needs to hold a real account for the
#                *upstream* connection to an online-mode server, so AUTH here
#                instead picks ViaProxy's auth-method: 'offline' (default) ->
#                auth-method NONE (an offline/cracked, or already-unauthenticated,
#                target server); 'microsoft' -> auth-method ACCOUNT, and this
#                script drives ViaProxy's one-time interactive account setup
#                (a Microsoft device-code login: it prints a URL + code to
#                open in a browser) the first time, then reuses the saved
#                account (azalea-bot/viaproxy/saves.json) on every run after.

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

# Colored status output; auto-disabled when stdout isn't a terminal (e.g.
# redirected to a file) or NO_COLOR is set, so logs stay plain ASCII.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_RED=$'\033[1;31m'; C_RESET=$'\033[0m'
else
    C_CYAN=''; C_GREEN=''; C_RED=''; C_RESET=''
fi

log() { printf '%s[run_bot.sh]%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
ok()  { printf '%s[run_bot.sh]%s %s%s%s\n' "$C_CYAN" "$C_RESET" "$C_GREEN" "$*" "$C_RESET"; }
die() { printf '%s[run_bot.sh] ERROR:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

# Whether azalea-bot/viaproxy/saves.json already has at least one saved
# ViaProxy account (any type - offline/microsoft/bedrock).
viaproxy_has_saved_account() {
    "$PY" -c "
import json, sys
try:
    with open('$VIAPROXY_DIR/saves.json') as f:
        accounts = json.load(f).get('accountsV4', [])
except (OSError, ValueError):
    accounts = []
sys.exit(0 if accounts else 1)
" 2>/dev/null
}

# Drives ViaProxy's own interactive CLI ('account add microsoft') so the
# upstream (real-server) connection can authenticate as a real Microsoft
# account - the piece that used to require hand-editing viaproxy.yml plus
# knowing ViaProxy's own account UI/CLI. Idempotent: a no-op once an account
# is already saved. The device-code login itself can't be scripted away (it's
# Microsoft's own OAuth flow - a human has to open the URL and sign in), so
# this only removes everything *around* that one unavoidable step.
ensure_viaproxy_microsoft_account() {
    viaproxy_has_saved_account && return 0
    log "No saved ViaProxy account yet - starting its one-time Microsoft login."
    log "A device code + URL will be printed below; open the URL in a browser and sign in with"
    log "the Microsoft account you want the bot to connect as. This only has to be done once -"
    log "the saved login (azalea-bot/viaproxy/saves.json) is reused on every run after."
    # ViaProxy's CLI console reads one command per line from stdin; the
    # 'account add microsoft' handler blocks until the login finishes (or
    # times out) before the console reads the next line, so 'stop' - already
    # sitting in the pipe - only runs after that. Foreground (inherits this
    # script's stdout) so the code/URL and any errors are visible live.
    ( cd "$VIAPROXY_DIR" && printf 'account add microsoft\nstop\n' | java -jar "$JAR" cli )
    viaproxy_has_saved_account || die "ViaProxy still has no saved account after the login attempt - see the output above and try again"
    ok "ViaProxy account saved."
}

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
            ok "Inference server successfully started."
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
    # AUTH=microsoft -> ViaProxy itself authenticates upstream as a real
    # account (auth-method ACCOUNT, index 0 - the only account this script
    # ever adds); AUTH=offline (default) -> unauthenticated upstream, as
    # before (auth-method NONE).
    if [ "$AUTH" = "microsoft" ]; then
        ensure_viaproxy_microsoft_account
        VIAPROXY_AUTH_LINES="auth-method: ACCOUNT
minecraft-account-index: 0"
    else
        VIAPROXY_AUTH_LINES="auth-method: NONE"
    fi
    cat > "$VIAPROXY_DIR/viaproxy.yml" <<EOF
bind-address: 127.0.0.1:$VIAPROXY_PORT
target-address: $SERVER_ADDR
target-version: $MC_VERSION
$VIAPROXY_AUTH_LINES
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
    ok "ViaProxy is up."
    SERVER_ADDR="127.0.0.1:$VIAPROXY_PORT"
    # The bot itself always talks to the *local*, unauthenticated ViaProxy -
    # AUTH picked ViaProxy's own upstream auth-method above, not the bot's.
    BOT_AUTH="offline"
else
    BOT_AUTH="$AUTH"
fi

# 4. Connect the bot
ok "Connecting the bot to $SERVER_ADDR as '$USERNAME' (auth: $BOT_AUTH)..."
cd "$BOT_DIR"
cargo run --release -- "$SERVER_ADDR" "$USERNAME" "$INFERENCE_URL" --auth "$BOT_AUTH"
