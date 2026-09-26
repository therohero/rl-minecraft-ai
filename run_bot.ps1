# Run the trained Minecraft bot against a real server (native Windows / PowerShell).
#
# Usage:
#   .\run_bot.ps1 [ip] [port] [username] [inference_url] [mc_version] [auth]
# Example:
#   .\run_bot.ps1 play.example.com 25565 TrainedBot
#   .\run_bot.ps1 play.example.com 25565 TrainedBot "" 1.21.4   # via ViaProxy
#
# `mc_version` (5th arg): the server's Minecraft version. When set, the bot
# connects through a local ViaProxy that translates to the version `azalea`
# speaks; needs a JRE on PATH. `auth` (6th arg): 'offline' (default) or
# 'microsoft', passed to the bot as --auth.
#
# This is the native-Windows equivalent of run_bot.sh - keep the two in
# sync if you change one. It:
#   1. exports training/checkpoints/latest.pt to azalea-bot/model/ if no model is there yet,
#   2. starts azalea-bot/inference_server.py in the background (reusing one
#      that's already listening on port 8800),
#   3. builds and runs azalea-bot/azalea_bot, pointed at the server.

param(
    [string]$Ip = "localhost",
    [string]$Port = "25565",
    [string]$Username = "TrainedBot",
    [string]$InferenceUrl = "http://127.0.0.1:8800/act",
    [string]$McVersion = "",
    [string]$Auth = "offline"
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[run_bot.ps1] $msg" }
function Die($msg) { Write-Host "[run_bot.ps1] ERROR: $msg" -ForegroundColor Red; exit 1 }

# Whether azalea-bot\viaproxy\saves.json already has a saved ViaProxy account.
function Test-ViaProxyAccount($ViaDir) {
    $path = Join-Path $ViaDir "saves.json"
    if (-not (Test-Path $path -PathType Leaf)) { return $false }
    try {
        $accounts = (Get-Content $path -Raw | ConvertFrom-Json).accountsV4
        return ($null -ne $accounts) -and ($accounts.Count -gt 0)
    } catch {
        return $false
    }
}

# Drives ViaProxy's own interactive CLI ('account add microsoft') so the
# upstream (real-server) connection can authenticate as a real Microsoft
# account - see run_bot.sh's ensure_viaproxy_microsoft_account (keep the two
# in sync). The device-code login itself can't be scripted away - a human
# still has to open the printed URL and sign in - this only removes
# everything *around* that one unavoidable step. Idempotent.
function Confirm-ViaProxyMicrosoftAccount($ViaDir, $Jar) {
    if (Test-ViaProxyAccount $ViaDir) { return }
    Write-Log "no saved ViaProxy account yet - starting its one-time Microsoft login."
    Write-Log "a device code + URL will be printed below; open the URL in a browser and sign in with"
    Write-Log "the Microsoft account you want the bot to connect as. This only has to be done once -"
    Write-Log "the saved login (azalea-bot\viaproxy\saves.json) is reused on every run after."
    Push-Location $ViaDir
    try {
        "account add microsoft`nstop`n" | & java -jar $Jar cli
    } finally {
        Pop-Location
    }
    if (-not (Test-ViaProxyAccount $ViaDir)) {
        Die "ViaProxy still has no saved account after the login attempt - see the output above and try again"
    }
    Write-Log "ViaProxy account saved."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotDir = Join-Path $ScriptDir "azalea-bot\azalea_bot"
$PythonDir = Join-Path $ScriptDir "training\python"
$BridgeDir = Join-Path $ScriptDir "azalea-bot"
$InferencePort = 8800

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die "cargo not found - install Rust: https://rustup.rs"
}

# Prefer the repo-local virtualenv created by run.ps1; fall back to a
# system Python if it isn't there yet.
$VenvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPy -PathType Leaf) {
    $Py = $VenvPy
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Py = "python"
    Write-Log "no .venv found - using system 'python' (run .\run.ps1 once to create the virtualenv)"
} else {
    Die "python not found - install Python 3.10+ (or run .\run.ps1 first to set up .venv)"
}

# Keep the .venv's torch matched to this machine (CPU vs CUDA wheel) - a fast
# no-op when it already is. Skipped for the system-python fallback.
if ($Py -eq $VenvPy) {
    & $Py (Join-Path $PythonDir "ensure_deps.py")
    if ($LASTEXITCODE -ne 0) { Die "Python dependency check failed" }
}

# 1. Ensure a model is exported.
$ModelPath = Join-Path $BridgeDir "model\policy.pt"
if (-not (Test-Path $ModelPath -PathType Leaf)) {
    Write-Log "model policy.pt not found - exporting from the latest checkpoint..."
    $Checkpoint = Join-Path $ScriptDir "training\checkpoints\latest.pt"
    if (-not (Test-Path $Checkpoint -PathType Leaf)) {
        Die "no training checkpoint at $Checkpoint - run training first (.\run.ps1)"
    }
    Push-Location $PythonDir
    try {
        & $Py export_model.py --checkpoint $Checkpoint --out-dir (Join-Path $BridgeDir "model")
        if ($LASTEXITCODE -ne 0) { Die "failed to export model checkpoint" }
    } finally {
        Pop-Location
    }
}

# 2. Start the inference server unless one is already listening.
function Test-Port($p) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $p); $c.Close(); return $true
    } catch { return $false }
}

$ServerProc = $null
if (Test-Port $InferencePort) {
    Write-Log "inference server already running on port $InferencePort - reusing it"
} else {
    Write-Log "starting inference server in the background..."
    $LogFile = Join-Path $BridgeDir "inference_server.log"
    $ServerProc = Start-Process -FilePath $Py `
        -ArgumentList @((Join-Path $BridgeDir "inference_server.py"), "--model-dir", (Join-Path $BridgeDir "model"), "--port", "$InferencePort") `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" `
        -NoNewWindow -PassThru

    Write-Log "waiting for inference server to start..."
    $ok = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port $InferencePort) { $ok = $true; break }
        if ($ServerProc.HasExited) {
            if (Test-Path $LogFile) { Get-Content $LogFile | Write-Host }
            Die "inference server failed to start - see $LogFile"
        }
    }
    if (-not $ok) { Die "inference server did not come up on port $InferencePort" }
    Write-Log "inference server is up."
}

# 3. Optionally start ViaProxy to translate to the server's Minecraft version.
$ServerAddr = "${Ip}:${Port}"
$ViaProxyProc = $null
$ViaProxyPort = 25568
if ($McVersion -ne "") {
    if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
        Die "McVersion is set but 'java' is not on PATH - ViaProxy needs a JRE (17+)."
    }
    $ViaDir = Join-Path $BridgeDir "viaproxy"
    New-Item -ItemType Directory -Force -Path $ViaDir | Out-Null
    $Jar = Join-Path $ViaDir "ViaProxy.jar"
    if (-not (Test-Path $Jar -PathType Leaf)) {
        Write-Log "downloading ViaProxy (one-time) into $ViaDir ..."
        $rel = Invoke-RestMethod "https://api.github.com/repos/ViaVersion/ViaProxy/releases/latest"
        $asset = $rel.assets | Where-Object { $_.name -like "*.jar" -and $_.name -notlike "*java8*" } | Select-Object -First 1
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $Jar
    }
    # $Auth picks ViaProxy's own upstream auth-method here (not the bot's -
    # see the param block doc above): 'microsoft' -> a real account
    # (auth-method ACCOUNT, index 0 - the only account this script ever
    # adds); 'offline' (default) -> unauthenticated upstream, as before.
    if ($Auth -eq "microsoft") {
        Confirm-ViaProxyMicrosoftAccount $ViaDir $Jar
        $ViaProxyAuthLines = "auth-method: ACCOUNT`nminecraft-account-index: 0"
    } else {
        $ViaProxyAuthLines = "auth-method: NONE"
    }
    @"
bind-address: 127.0.0.1:$ViaProxyPort
target-address: $ServerAddr
target-version: $McVersion
$ViaProxyAuthLines
proxy-online-mode: false
"@ | Set-Content -Path (Join-Path $ViaDir "viaproxy.yml")
    Write-Log "starting ViaProxy: $ServerAddr ($McVersion) -> 127.0.0.1:$ViaProxyPort"
    $ViaLog = Join-Path $ViaDir "viaproxy.log"
    $ViaProxyProc = Start-Process -FilePath "java" -ArgumentList @("-jar", $Jar) `
        -WorkingDirectory $ViaDir -RedirectStandardOutput $ViaLog -RedirectStandardError "$ViaLog.err" `
        -NoNewWindow -PassThru
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port $ViaProxyPort) { $ok = $true; break }
        if ($ViaProxyProc.HasExited) { Get-Content $ViaLog -ErrorAction SilentlyContinue | Write-Host; Die "ViaProxy exited during startup" }
    }
    if (-not $ok) { Die "ViaProxy did not open port $ViaProxyPort" }
    Write-Log "ViaProxy is up."
    $ServerAddr = "127.0.0.1:$ViaProxyPort"
    # The bot itself always talks to the *local*, unauthenticated ViaProxy -
    # $Auth picked ViaProxy's own upstream auth-method above, not the bot's.
    $BotAuth = "offline"
} else {
    $BotAuth = $Auth
}

# 4. Connect the bot. Stop the background processes afterwards if we started them.
Write-Log "connecting the bot to $ServerAddr as '$Username' (auth: $BotAuth)..."
Push-Location $BotDir
try {
    cargo run --release -- $ServerAddr $Username $InferenceUrl --auth $BotAuth
} finally {
    Pop-Location
    if ($ViaProxyProc -and -not $ViaProxyProc.HasExited) {
        Write-Log "stopping ViaProxy (PID $($ViaProxyProc.Id))..."
        Stop-Process -Id $ViaProxyProc.Id -ErrorAction SilentlyContinue
    }
    if ($ServerProc -and -not $ServerProc.HasExited) {
        Write-Log "stopping inference server (PID $($ServerProc.Id))..."
        Stop-Process -Id $ServerProc.Id -ErrorAction SilentlyContinue
    }
}