# Starts the inference server that the Fabric client mod (mod\) connects to
# (native Windows / PowerShell).
#
# It exports the latest training checkpoint to azalea-bot\model\ and then runs
# azalea-bot\inference_server.py in the foreground. Leave this running, launch
# Minecraft with the mod installed, and use /fight (or /fight train).
#
# Usage:
#   .\run_bot_mod.ps1 [-Port <n>] [extra args passed to inference_server.py]
# Examples:
#   .\run_bot_mod.ps1                       # serve on 127.0.0.1:8800 (the mod's default)
#   .\run_bot_mod.ps1 -Port 8801
#   .\run_bot_mod.ps1 --sample --device cuda
#
# This does NOT train and does NOT run the headless azalea bot - it is just
# the model server. Train first with .\run.ps1; run the headless bot with
# .\run_bot.ps1. This is the native-Windows equivalent of run_bot_mod.sh -
# keep the two in sync.

param(
    [int]$Port = $(if ($env:INFERENCE_PORT) { [int]$env:INFERENCE_PORT } else { 8800 }),
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Extra
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[run_bot_mod.ps1] $msg" }
function Die($msg) { Write-Host "[run_bot_mod.ps1] ERROR: $msg" -ForegroundColor Red; exit 1 }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonDir = Join-Path $ScriptDir "training\python"
$CheckpointDir = Join-Path $ScriptDir "training\checkpoints"
$BridgeDir = Join-Path $ScriptDir "azalea-bot"
$ModelDir = Join-Path $BridgeDir "model"

# Prefer the repo-local virtualenv that run.ps1 builds; fall back to system python.
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPy -PathType Leaf) {
    $Py = $VenvPy
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Py = "python"
    Write-Log "no .venv found - using system 'python' (run .\run.ps1 once to create it)"
} else {
    Die "python not found - install Python 3.10+ (or run .\run.ps1 first)"
}

# Keep the .venv's torch matched to this machine (CPU vs CUDA wheel) - a fast
# no-op when it already is. Skipped for the system-python fallback.
if ($Py -eq $VenvPy) {
    & $Py (Join-Path $PythonDir "ensure_deps.py")
    if ($LASTEXITCODE -ne 0) { Die "Python dependency check failed" }
}

$Checkpoint = Join-Path $CheckpointDir "latest.pt"
if (Test-Path $Checkpoint -PathType Leaf) {
    # Always re-export so the served model matches the current feature code.
    Write-Log "exporting $Checkpoint -> $ModelDir ..."
    Push-Location $PythonDir
    try {
        & $Py export_model.py --checkpoint $Checkpoint --out-dir $ModelDir
        if ($LASTEXITCODE -ne 0) { Die "model export failed" }
    } finally {
        Pop-Location
    }
} elseif (Test-Path (Join-Path $ModelDir "policy.pt") -PathType Leaf) {
    Write-Log "no checkpoint at $Checkpoint - serving the model already in $ModelDir"
} else {
    Die "no checkpoint at $Checkpoint and no model in $ModelDir - train first with .\run.ps1"
}

Write-Log "serving on http://127.0.0.1:$Port/act - leave this running, then start Minecraft + the mod and use /fight"
$serverArgs = @((Join-Path $BridgeDir "inference_server.py"), "--model-dir", $ModelDir, "--port", "$Port")
if ($Extra) { $serverArgs += $Extra }
& $Py @serverArgs
