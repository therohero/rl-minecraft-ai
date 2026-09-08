# Offline-trains the policy on the episodes the Fabric mod recorded with
# `/fight train`, then leaves training\checkpoints\latest.pt updated (the old
# one is kept as latest.pt.pre_finetune). Re-run .\run_bot_mod.ps1 afterwards
# to serve the updated model.
#
# Usage:
#   .\run_train_mod.ps1 [-DatasetDir <path>] [extra args for train_from_episodes.py]
# Examples:
#   .\run_train_mod.ps1                                  # DatasetDir defaults to training\datasets\
#   .\run_train_mod.ps1 -DatasetDir $env:APPDATA\.minecraft\rl-datasets
#   .\run_train_mod.ps1 --dry-run
#   .\run_train_mod.ps1 --epochs 8 --win 20 --loss 20
#
# The mod writes to <game dir>\rl-datasets by default - point -DatasetDir at
# that, or set `dataset_dir` in the mod config to <repo>\training\datasets so
# this works with no argument. Native-Windows equivalent of run_train_mod.sh;
# keep the two in sync.

param(
    [string]$DatasetDir,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Extra
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[run_train_mod.ps1] $msg" }
function Die($msg) { Write-Host "[run_train_mod.ps1] ERROR: $msg" -ForegroundColor Red; exit 1 }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonDir = Join-Path $ScriptDir "training\python"

if (-not $DatasetDir) { $DatasetDir = Join-Path $ScriptDir "training\datasets" }
elseif (-not [System.IO.Path]::IsPathRooted($DatasetDir)) { $DatasetDir = Join-Path $ScriptDir $DatasetDir }

if (-not (Test-Path $DatasetDir -PathType Container)) {
    Die "no dataset dir at $DatasetDir - record fights with /fight train first (or pass -DatasetDir)"
}

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

Write-Log "offline-training on episodes in $DatasetDir ..."
Push-Location $PythonDir
try {
    $pyArgs = @("train_from_episodes.py", "--episodes", $DatasetDir)
    if ($Extra) { $pyArgs += $Extra }
    & $Py @pyArgs
} finally {
    Pop-Location
}
