#!/usr/bin/env bash
# Offline-trains the policy on the episodes the Fabric mod recorded with
# `/fight train`, then leaves training/checkpoints/latest.pt updated (the old
# one is kept as latest.pt.pre_finetune). Re-run ./run_bot_mod.sh afterwards
# to serve the updated model.
#
# Usage:
#   ./run_train_mod.sh [dataset_dir] [-- extra args for train_from_episodes.py]
# Examples:
#   ./run_train_mod.sh                         # dataset_dir defaults to training/datasets/
#   ./run_train_mod.sh ~/.minecraft/rl-datasets
#   ./run_train_mod.sh training/datasets -- --dry-run
#   ./run_train_mod.sh training/datasets -- --epochs 8 --win 20 --loss 20
#
# The mod writes to <game dir>/rl-datasets by default - point dataset_dir at
# that, or set `dataset_dir` in the mod config to <repo>/training/datasets so
# this works with no argument.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/training/python"

DATASET_DIR="$SCRIPT_DIR/training/datasets"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then DATASET_DIR="$1"; shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
# make relative paths resolve against the repo root, not training/python
case "$DATASET_DIR" in /*) ;; *) DATASET_DIR="$SCRIPT_DIR/$DATASET_DIR" ;; esac

log() { printf '[run_train_mod.sh] %s\n' "$*"; }
die() { printf '[run_train_mod.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[ -d "$DATASET_DIR" ] || die "no dataset dir at $DATASET_DIR - record fights with /fight train first (or pass the path)"

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

log "offline-training on episodes in $DATASET_DIR ..."
cd "$PYTHON_DIR"
exec "$PY" train_from_episodes.py --episodes "$DATASET_DIR" "$@"
