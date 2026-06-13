param(
    [ValidateSet("app", "demo")]
    [string]$Mode = "app",

    [int]$FrontendPort = 5173,
    [int]$BackendPort = 8000,

    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RuntimeDir = Join-Path $RepoRoot ".runtime"
$TempDir = Join-Path $RepoRoot ".tmp"
$FrontendDir = Join-Path $RepoRoot "frontend"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}

$NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $NpmCommand) {
    $NpmCommand = Get-Command npm -ErrorAction Stop
}
$Npm = $NpmCommand.Source

New-Item -ItemType Directory -Force -Path $RuntimeDir, $TempDir | Out-Null

function Repair-ProcessPathEnvironment {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    if (-not [string]::IsNullOrWhiteSpace($pathValue)) {
        [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
        [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
    }
}

function Test-ProcessAlive {
    param([string]$PidFile)
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $false
    }
    $processIdText = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if (-not $processIdText) {
        return $false
    }
    return [bool](Get-Process -Id ([int]$processIdText) -ErrorAction SilentlyContinue)
}

function Start-LoggedCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory
    )

    $pidFile = Join-Path $RuntimeDir "$Name.pid"
    if (Test-ProcessAlive -PidFile $pidFile) {
        $existingPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
        Write-Host "$Name already running as PID $existingPid"
        return
    }

    $stdout = Join-Path $RuntimeDir "$Name.out.log"
    $stderr = Join-Path $RuntimeDir "$Name.err.log"

    Repair-ProcessPathEnvironment
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -LiteralPath $pidFile -Value $process.Id
    Write-Host "Started $Name as PID $($process.Id)"
    Write-Host "  logs: $stdout"
}

function Set-ProcessEnv {
    param(
        [string]$Name,
        [AllowNull()][string]$Value
    )
    if ($null -eq $Value) {
        Remove-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
        return
    }
    Set-Item -Path "Env:$Name" -Value $Value
}

if ($Restart) {
    & (Join-Path $PSScriptRoot "Stop-AlphaLineage.ps1")
}

if ($Mode -eq "app") {
    $oldTemp = $env:TEMP
    $oldTmp = $env:TMP
    $oldPythonPath = $env:PYTHONPATH
    try {
        Set-ProcessEnv -Name "TEMP" -Value $TempDir
        Set-ProcessEnv -Name "TMP" -Value $TempDir
        Set-ProcessEnv -Name "PYTHONPATH" -Value (Join-Path $RepoRoot "src")
        Start-LoggedCommand `
            -Name "backend" `
            -FilePath $Python `
            -ArgumentList @("-m", "uvicorn", "alphalineage.api.app:app", "--reload", "--port", "$BackendPort") `
            -WorkingDirectory $RepoRoot
    } finally {
        Set-ProcessEnv -Name "TEMP" -Value $oldTemp
        Set-ProcessEnv -Name "TMP" -Value $oldTmp
        Set-ProcessEnv -Name "PYTHONPATH" -Value $oldPythonPath
    }
}

$oldViteApiBase = $env:VITE_API_BASE
try {
    Set-ProcessEnv -Name "VITE_API_BASE" -Value "http://localhost:$BackendPort"
    if ($Mode -eq "app") {
        $frontendArgs = @("run", "dev:app", "--", "--host", "127.0.0.1", "--port", "$FrontendPort", "--strictPort")
    } else {
        $frontendArgs = @("run", "dev", "--", "--mode", "demo", "--host", "127.0.0.1", "--port", "$FrontendPort", "--strictPort")
    }
    Start-LoggedCommand `
        -Name "frontend" `
        -FilePath $Npm `
        -ArgumentList $frontendArgs `
        -WorkingDirectory $FrontendDir
} finally {
    Set-ProcessEnv -Name "VITE_API_BASE" -Value $oldViteApiBase
}

Write-Host ""
Write-Host "AlphaLineage $Mode mode is starting."
Write-Host "Frontend: http://localhost:$FrontendPort"
if ($Mode -eq "app") {
    Write-Host "Backend:  http://localhost:$BackendPort"
}
Write-Host "Stop with: powershell -ExecutionPolicy Bypass -File scripts\Stop-AlphaLineage.ps1"
