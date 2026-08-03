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

$RuntimeDir = Join-Path $RepoRoot ".runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$LogPath = Join-Path $RuntimeDir "app.log"

# --- pre-flight: can the app even be imported? -------------------------------------
# uvicorn imports the app before it binds a port, so a missing dependency exits instantly
# with the traceback going to a console window that then closes. That looks like "the app
# ejects itself" and leaves nothing to read. Check it here, where the error can be shown and
# the window kept open. The usual cause is dependency drift: a venv created before a new
# requirement was added.
#
# Both probes capture stderr, and under `$ErrorActionPreference = "Stop"` a native command
# writing to stderr raises a terminating error in Windows PowerShell. A stray DeprecationWarning
# would then kill the launcher during the very check meant to make it robust, so capture runs
# with the preference relaxed.
function Invoke-PythonProbe {
    param([string]$Code)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Python -c $Code 2>&1
        return [pscustomobject]@{ Output = $output; ExitCode = $LASTEXITCODE }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

Write-Host "Checking the environment..."
$Probe = Invoke-PythonProbe -Code "import alphalineage.api.app"
$ImportCheck = $Probe.Output
if ($Probe.ExitCode -ne 0) {
    $ImportCheck | Out-File -FilePath $LogPath -Encoding utf8
    Write-Host ""
    Write-Host "AlphaLineage could not start: the application failed to import." -ForegroundColor Red
    Write-Host ""
    $ImportCheck | Select-Object -Last 20 | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    if ($ImportCheck -match "ModuleNotFoundError|ImportError") {
        Write-Host "This is almost always a dependency missing from the environment." -ForegroundColor Yellow
        Write-Host "Reinstall with the SAME interpreter this launcher uses, then try again:" -ForegroundColor Yellow
        Write-Host "    & '$Python' -m pip install -e `".[dev]`""
        Write-Host ""
        Write-Host "  (If you use uv:  uv pip install -e `".[dev]`" )" -ForegroundColor DarkGray
        Write-Host ""
    }
    Write-Host "The full output was saved to $LogPath"
    if ($Host.Name -eq "ConsoleHost") { Read-Host "Press Enter to close" }
    exit 1
}

# Optional features are warned about, never fatal: a broken agent install must not stop a
# user from training, explaining, or backtesting.
$AgentProbe = Invoke-PythonProbe -Code "from alphalineage.agent import service; print(service.unavailable_reason())"
$AgentCheck = $AgentProbe.Output
if ($AgentProbe.ExitCode -eq 0 -and $AgentCheck -and ($AgentCheck -join "").Trim()) {
    Write-Host ""
    Write-Host "Note: the Explain page's agent features are unavailable." -ForegroundColor Yellow
    Write-Host "  $AgentCheck"
    Write-Host "  To enable them:  & '$Python' -m pip install -e `".[dev]`""
    Write-Host "  Everything else works normally."
    Write-Host ""
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
Write-Host "Quit from the in-app header, or press Ctrl+C here."
Write-Host "Server output is also written to $LogPath"
Write-Host ""

# --- run the server in the foreground (Quit / Ctrl+C ends it) ---------------------
try {
    $ServerArgs = @("-m", "uvicorn", "alphalineage.api.app:app", "--host", "127.0.0.1", "--port", "$Port")
    $EnvFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path -LiteralPath $EnvFile) {
        $ServerArgs += @("--env-file", $EnvFile)
    }
    # Capture the console to a log via a transcript, NOT by redirecting the server's streams.
    #
    # uvicorn logs to stderr. Piping it with `2>&1 |` makes PowerShell wrap every ordinary log
    # line in a NativeCommandError record, which it then prints in red with a CategoryInfo
    # block — so a perfectly healthy startup looks like a crash. A transcript records the same
    # text without touching the child process's streams at all, which keeps the console clean
    # and Ctrl+C behaving normally.
    $Transcribing = $false
    try {
        Start-Transcript -LiteralPath $LogPath -Append -Force | Out-Null
        $Transcribing = $true
    } catch {
        Write-Host "(could not open $LogPath for logging; continuing without it)" -ForegroundColor DarkGray
    }

    # A non-zero exit from a native command throws under PowerShell 7's native error handling,
    # which would skip the diagnostic below. Handle the exit code explicitly instead.
    $PreviousNativeErrorAction = $null
    if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) {
        $PreviousNativeErrorAction = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        & $Python @ServerArgs
        $ServerExit = $LASTEXITCODE
    }
    finally {
        if ($null -ne $PreviousNativeErrorAction) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativeErrorAction
        }
        if ($Transcribing) { try { Stop-Transcript | Out-Null } catch { } }
    }
    # Ctrl+C on Windows terminates python with STATUS_CONTROL_C_EXIT (0xC000013A), which is a
    # normal way to stop the server, not a fault. Reporting it as a crash — and pausing for a
    # keypress — would make every deliberate shutdown look like a failure.
    $CleanExits = @(0, -1073741510, 3221225786)
    if ($CleanExits -notcontains $ServerExit) {
        Write-Host ""
        Write-Host "The server exited with code $ServerExit. See $LogPath" -ForegroundColor Red
        if ($Host.Name -eq "ConsoleHost") { Read-Host "Press Enter to close" }
    }
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
