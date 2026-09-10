# syntax=docker/dockerfile:1
#
# GPU training image: builds the Rust sim, installs a CUDA torch, and runs
# `train.py` as the entrypoint. Only the training half of the repo is in
# here - `azalea-bot/` and `mod/` need a real Minecraft server / client and
# are not containerised.
#
#   docker compose run --rm --build train              # train with defaults
#   docker compose run --rm train --num-arenas 512     # extra args -> train.py
#
# or without compose:
#
#   docker build -t rl-minecraft-ai-train .
#   docker run --rm -it --gpus all --init \
#       -v "$PWD/training/checkpoints:/app/training/checkpoints" \
#       rl-minecraft-ai-train
#
# See README.md's "Docker (GPU training)" section.

# Runtime base. `-base` is deliberate: the torch wheel bundles its own
# CUDA/cuDNN userspace, so all this image needs from the host is the driver,
# which the NVIDIA Container Toolkit injects - a ~100 MB layer instead of
# ~2 GB. Declared up here because a global ARG has to precede the *first*
# FROM for buildah/podman to accept it.
ARG CUDA_IMAGE=docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04

# ---------------------------------------------------------------------------
# stage 1 - build mc_pvp_sim (release)
# ---------------------------------------------------------------------------
# Base images are fully qualified (`docker.io/...`) throughout: Docker
# infers that registry for a bare name, Podman refuses to guess unless the
# host's registries.conf says so. Spelling it out keeps the build working on
# both without anyone having to edit /etc/containers/registries.conf.
FROM docker.io/library/rust:1-slim-bookworm AS sim-build

WORKDIR /build

# Dependency layer: with only the manifests present, cargo compiles every
# crates.io dependency and stops at a stub binary. Editing src/ afterwards
# reuses this layer instead of rebuilding rayon/serde/etc from scratch.
COPY training/sim/Cargo.toml training/sim/Cargo.lock ./
RUN mkdir src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build --release --locked \
 && rm -rf src target/release/mc_pvp_sim \
           target/release/deps/mc_pvp_sim* \
           target/release/.fingerprint/mc_pvp_sim-*

COPY training/sim/src ./src
RUN cargo build --release --locked

# ---------------------------------------------------------------------------
# stage 2 - runtime
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: torch's CPU kernels link against OpenMP, which the base image
# doesn't carry.
#
# The CUDA apt source that ships in the base image is dropped first: nothing
# here is installed from it (the torch wheel brings its own CUDA userspace),
# and leaving it in means every build depends on that mirror being healthy -
# a sync in progress there fails `apt-get update` outright.
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-venv \
 && rm -rf /var/lib/apt/lists/*

# A venv rather than a system install: Ubuntu 24.04's python is
# externally-managed (PEP 668), and this keeps the layout close to the
# repo-local `.venv/` that run.sh builds.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv "$VIRTUAL_ENV" && pip install --upgrade pip

# ensure_deps.py is skipped on purpose - it picks a wheel from whatever
# hardware it sees at *runtime*, which is the wrong question at image-build
# time. The backend is pinned here instead.
#
# cu128 (torch >= 2.7) is the first build with Blackwell (sm_120) kernels;
# override for an older card or a newer CUDA, e.g.
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC="torch>=2.7"
RUN pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}" \
 && pip install "numpy>=1.24" "pytest>=8.0"

# uid/gid 1000 so bind-mounted checkpoints come back owned by the caller on
# a typical single-user Linux host. Ubuntu 24.04 ships its own uid-1000
# `ubuntu` account, which has to go first.
RUN userdel -r ubuntu 2>/dev/null || true \
 && useradd --create-home --uid 1000 --shell /bin/bash trainer

COPY --from=sim-build /build/target/release/mc_pvp_sim /usr/local/bin/mc_pvp_sim
COPY training/python /app/training/python
RUN mkdir -p /app/training/checkpoints && chown -R trainer:trainer /app

USER trainer
WORKDIR /app/training/python

# train.py saves `latest.pt` from a `finally` that only runs on
# KeyboardInterrupt; the default SIGTERM would kill it mid-run and lose the
# last updates. So `docker stop` / `compose down` sends SIGINT instead.
STOPSIGNAL SIGINT

# --checkpoint-dir defaults to ../checkpoints -> /app/training/checkpoints,
# which is the bind-mount point. Anything passed after the image name is
# forwarded to train.py.
ENTRYPOINT ["python3", "train.py", "--sim-binary", "/usr/local/bin/mc_pvp_sim"]
