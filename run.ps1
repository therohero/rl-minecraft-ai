# One-shot entry point for native Windows (PowerShell) - builds the Rust
# simulation backend (release mode) if needed, ensures Python dependencies
# are installed, then starts self-play training. Ctrl+C stops it (train.py
# saves `latest.pt` first). Any arguments you pass are forwarded straight to
# train.py, e.g.:
#
#   .\run.ps1                          # train with defaults (arena count auto-detected from CPU count, CPU/GPU auto)
#   .\run.ps1 --num-arenas 128         # override the arena count
#   .\run.ps1 --device cuda            # force GPU
#
# To actually use the trained bot, start the model server separately with
# `.\run_bot_mod.ps1` (for the mod\ Fabric client) or `.\run_bot.ps1` (for
# the headless azalea bot).
#
# Re-running this script is safe and fast: cargo/pip only do real work when
# something actually changed.
#
# On Linux, macOS, or Windows under WSL/Git Bash, use `./run.sh` instead -
# this script is the native-Windows (no bash) equivalent of it; keep the
# two in sync if you change one.

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[run.ps1] $msg" }
function Die($msg) { Write-Host "[run.ps1] ERROR: $msg" -ForegroundColor Red; exit 1 }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SimDir = Join-Path $ScriptDir "training\sim"
$PythonDir = Join-Path $ScriptDir "training\python"
$SimBinary = Join-Path $SimDir "target\release\mc_pvp_sim.exe"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die "cargo not found - install Rust: https://rustup.rs"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Die "python not found - install Python 3.10+"
}

Write-Log "building simulation backend (release mode)..."
Push-Location $SimDir
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { Die "Rust build failed - see errors above" }
} finally {
    Pop-Location
}
if (-not (Test-Path $SimBinary -PathType Leaf)) {
    Die "build succeeded but binary not found at $SimBinary"
}

# All Python work happens in a repo-local virtualenv (.venv\) so this never
# touches the global site-packages. Created once, reused after that.
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPy -PathType Leaf)) {
    Write-Log "creating virtualenv at $VenvDir ..."
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Die "failed to create virtualenv" }
}
if (-not (Test-Path $VenvPy -PathType Leaf)) { Die "virtualenv python not found at $VenvPy" }

# Installs torch + numpy + pytest, picking the CPU-only or the CUDA torch
# wheel to match this machine (see training/python/ensure_deps.py). Fast
# no-op once the environment already matches. Override with the env var
# RL_TORCH_BACKEND=cpu|cuda if the autodetect ever guesses wrong.
Write-Log "checking Python dependencies (in .venv)..."
& $VenvPy (Join-Path $PythonDir "ensure_deps.py")
if ($LASTEXITCODE -ne 0) {
    Die "failed to install Python dependencies - try: $VenvPy -m pip install -r training/python/requirements.txt"
}

Write-Log "starting training (Ctrl+C to stop; periodic checkpoints -> training\checkpoints\)..."
Push-Location $PythonDir
try {
    & $VenvPy train.py --sim-binary $SimBinary @args
} finally {
    Pop-Location
}
