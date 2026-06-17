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
& $Python -m uvicorn alphalineage.api.app:app --host 127.0.0.1 --port $Port
