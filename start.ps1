# AlphaLineage one-command launcher (single self-contained process).
#
# Builds the UI, then runs ONE uvicorn that serves both the API and the built UI on
# http://localhost:8000, and opens a browser tab. Quit from the in-app gear menu (⚙) shuts
# this process down cleanly - backend and the served UI go down together.
#
# Run:  powershell -ExecutionPolicy Bypass -File start.ps1   (or double-click start.cmd)

param(
    [int]$Port = 8000,
    [switch]$NoBuild,   # reuse an existing frontend/dist instead of rebuilding
    [switch]$NoOpen     # do not open the browser automatically
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$FrontendDir = Join-Path $RepoRoot "frontend"
$DistDir = Join-Path $FrontendDir "dist"

# --- locate python + npm ----------------------------------------------------------
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
$Npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue)
if (-not $Npm) { $Npm = Get-Command npm -ErrorAction Stop }
$Npm = $Npm.Source

# --- build the UI (same-origin docker target) -------------------------------------
if (-not $NoBuild -or -not (Test-Path -LiteralPath $DistDir)) {
    if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir "node_modules"))) {
        Write-Host "Installing frontend dependencies..."
        & $Npm --prefix $FrontendDir install
    }
    Write-Host "Building the UI..."
    & $Npm --prefix $FrontendDir run build:docker
}

# --- environment for the single process -------------------------------------------
$env:ALPHALINEAGE_STATIC_DIR = $DistDir
$env:ALPHALINEAGE_ALLOW_SHUTDOWN = "1"   # let the in-app Quit stop this process
$env:PYTHONPATH = Join-Path $RepoRoot "src"
if (-not $env:ALPHALINEAGE_DATA_DIR) {
    $env:ALPHALINEAGE_DATA_DIR = Join-Path $RepoRoot "data_cache"
}

# --- free the port if a previous AlphaLineage instance was left running ------------
# Closing the window (instead of Ctrl+C) can orphan the foreground uvicorn, which keeps
# holding the port; the next launch would then boot, fail to bind, and exit immediately.
# Detect that here: stop a stale AlphaLineage instance automatically, but never kill
# an unrelated process that happens to own the port.
foreach ($conn in @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
    if ($proc -and $proc.Name -eq "python.exe" -and $proc.CommandLine -match "alphalineage\.api\.app") {
        Write-Host "Stopping a stale AlphaLineage instance on port $Port (PID $($proc.ProcessId))..."
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 300
    }
    else {
        $who = if ($proc) { "$($proc.Name) (PID $($proc.ProcessId))" } else { "PID $($conn.OwningProcess)" }
        Write-Host "Port $Port is already in use by $who."
        Write-Host "Stop it, or relaunch on another port:  start.cmd -Port <other>"
        exit 1
    }
}

# --- open the browser once the server is healthy ----------------------------------
if (-not $NoOpen) {
    Start-Job -ScriptBlock {
        param($url)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                if ((Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
                    Start-Process $url
                    return
                }
            } catch { Start-Sleep -Milliseconds 500 }
        }
    } -ArgumentList "http://localhost:$Port" | Out-Null
}

Write-Host ""
Write-Host "AlphaLineage is starting at http://localhost:$Port"
Write-Host "Quit from the in-app gear menu, or press Ctrl+C here."
Write-Host ""

# --- run the server in the foreground (Quit / Ctrl+C ends it) ---------------------
try {
    & $Python -m uvicorn alphalineage.api.app:app --host 127.0.0.1 --port $Port
}
finally {
    # Best-effort: if this run's server is somehow still bound on exit, reclaim the port
    # so the next launch starts cleanly. (The pre-flight check above is the primary guard.)
    foreach ($conn in @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
        $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
        if ($p -and $p.Name -eq "python.exe" -and $p.CommandLine -match "alphalineage\.api\.app") {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}
