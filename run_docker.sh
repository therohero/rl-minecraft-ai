#!/usr/bin/env bash
# Same job as ./run.sh, but inside a container: builds the image (see
# Dockerfile) if needed, then runs self-play training in it. Ctrl+C stops it
# after train.py saves `latest.pt`. Any arguments are forwarded straight to
# train.py, e.g.:
#
#   ./run_docker.sh                        # train with defaults (GPU target)
#   ./run_docker.sh --num-arenas 512       # override the arena count
#   RL_DOCKER_TARGET=cpu ./run_docker.sh   # the runtime-cpu target instead
#
# Checkpoints are bind-mounted, so they show up in training/checkpoints/ on
# the host exactly as with ./run.sh - ./run_bot_mod.sh then exports and
# serves them without knowing a container was involved.
#
# Works with either Docker or rootless Podman; the differences between them
# (GPU flag, uid mapping) are handled below. Nothing else is needed on the
# host - no local Rust, Python or venv.
#
# Env knobs:
#   RL_DOCKER_ENGINE=docker|podman   skip engine autodetection
#   RL_DOCKER_TARGET=gpu|cpu         which Dockerfile stage to build/run
#                                    (default gpu) - see the Dockerfile
#                                    header for when CPU actually wins;
#                                    it's a distinct target, not a
#                                    fallback picked for you
#   RL_DOCKER_GPU=0                  within the gpu target, run without
#                                    --gpus/CDI (the image still carries
#                                    CUDA torch) - ignored for RL_DOCKER_TARGET=cpu
#   RL_DOCKER_IMAGE=<name>           image tag to build/run (default
#                                    rl-minecraft-ai-train[-cpu])

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '[run_docker.sh] %s\n' "$*"; }
die() { printf '[run_docker.sh] ERROR: %s\n' "$*" >&2; exit 1; }

TARGET="${RL_DOCKER_TARGET:-gpu}"
case "$TARGET" in
    gpu) DOCKERFILE_TARGET=runtime ;;
    cpu) DOCKERFILE_TARGET=runtime-cpu ;;
    *) die "RL_DOCKER_TARGET must be 'gpu' or 'cpu' (got '$TARGET')" ;;
esac
if [ -n "${RL_DOCKER_IMAGE:-}" ]; then
    IMAGE="$RL_DOCKER_IMAGE"
elif [ "$TARGET" = "cpu" ]; then
    IMAGE="rl-minecraft-ai-train-cpu"
else
    IMAGE="rl-minecraft-ai-train"
fi
log "target: $TARGET (Dockerfile stage '$DOCKERFILE_TARGET'), image: $IMAGE"

# --- which engine -----------------------------------------------------------
# `docker` may be the podman-docker shim rather than Docker proper, and the
# two need genuinely different flags, so resolve to the real thing.
ENGINE="${RL_DOCKER_ENGINE:-}"
if [ -z "$ENGINE" ]; then
    # Note the command substitution rather than a `... | grep -qi podman`
    # pipeline: grep -q exits at the first match, and under `pipefail` the
    # resulting SIGPIPE on the left-hand side would make a *successful*
    # match look like a failed test.
    docker_version="$(docker --version 2>&1 || true)"
    if command -v podman >/dev/null 2>&1 && { [ -z "$docker_version" ] || [[ "$docker_version" == *[Pp]odman* ]]; }; then
        # Either no Docker at all, or `docker` is the podman-docker shim -
        # go straight to podman and skip the shim's wrapper noise.
        ENGINE=podman
    elif [ -n "$docker_version" ]; then
        ENGINE=docker
    else
        die "no container engine found - install Docker or Podman"
    fi
fi
command -v "$ENGINE" >/dev/null 2>&1 || die "engine '$ENGINE' not found on PATH"
log "using $ENGINE ($("$ENGINE" --version 2>/dev/null | tail -1))"

# --- GPU --------------------------------------------------------------------
# Docker takes --gpus; rootless Podman goes through CDI instead, which needs
# nvidia-container-toolkit to have written a spec (see README). None of this
# applies to the cpu target - it's a plain CPU-only image, nothing to pass
# through.
GPU_ARGS=()
if [ "$TARGET" = "cpu" ]; then
    :
elif [ "${RL_DOCKER_GPU:-1}" = "0" ]; then
    log "RL_DOCKER_GPU=0 - running the gpu-target image without GPU access"
elif [ "$ENGINE" = "docker" ]; then
    GPU_ARGS=(--gpus all)
elif compgen -G "/etc/cdi/*.yaml" >/dev/null 2>&1 || compgen -G "/etc/cdi/*.json" >/dev/null 2>&1 \
  || compgen -G "/var/run/cdi/*.yaml" >/dev/null 2>&1 || compgen -G "/var/run/cdi/*.json" >/dev/null 2>&1; then
    GPU_ARGS=(--device nvidia.com/gpu=all)
else
    cat >&2 <<'MSG'
[run_docker.sh] ERROR: no CDI spec found, so Podman cannot see the GPU.
  Podman does not support Docker's --gpus; it needs a CDI spec written by
  the NVIDIA Container Toolkit. Install it, then (on WSL2):

    sudo nvidia-ctk cdi generate --mode=wsl --output=/etc/cdi/nvidia.yaml
    podman run --rm --device nvidia.com/gpu=all \
        docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi   # verify

  On a native Linux host drop `--mode=wsl`. To train on the CPU meanwhile:

    RL_DOCKER_TARGET=cpu ./run_docker.sh
MSG
    exit 1
fi

# --- uid mapping ------------------------------------------------------------
# The image runs as uid 1000. Under rootless Podman that uid maps to a
# *subuid* on the host by default (100999-ish), which would leave the
# bind-mounted checkpoints owned by a user you can't even delete as. keep-id
# pins container uid 1000 to your real one instead. Docker has no such
# remapping, so uid 1000 is already the host's uid 1000.
USERNS_ARGS=()
if [ "$ENGINE" = "podman" ] && [ "$(id -u)" != "0" ]; then
    USERNS_ARGS=(--userns=keep-id)
fi

# Created here rather than left to the engine: a bind-mount source the
# daemon has to invent is created root-owned, and then the container's
# unprivileged user can't write checkpoints into it.
mkdir -p "$SCRIPT_DIR/training/checkpoints"

# A TTY makes Ctrl+C reach train.py as SIGINT (which is what makes it save
# `latest.pt` on the way out), but would break a piped/CI invocation.
TTY_ARGS=()
[ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)

if [ "$TARGET" = "cpu" ]; then
    log "building image (first run pulls CPU torch - ~200 MB - and compiles the sim)..."
else
    log "building image (first run pulls CUDA torch - a few GB - and compiles the sim)..."
fi
"$ENGINE" build --target "$DOCKERFILE_TARGET" -t "$IMAGE" "$SCRIPT_DIR" || die "image build failed - see errors above"

log "starting training in a container (Ctrl+C to stop; checkpoints -> training/checkpoints/)..."
exec "$ENGINE" run --rm --init \
    "${TTY_ARGS[@]}" "${GPU_ARGS[@]}" "${USERNS_ARGS[@]}" \
    -v "$SCRIPT_DIR/training/checkpoints:/app/training/checkpoints" \
    "$IMAGE" "$@"
