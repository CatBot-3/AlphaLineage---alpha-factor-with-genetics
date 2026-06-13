param()

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RuntimeDir = Join-Path $RepoRoot ".runtime"

function Stop-ProcessTree {
    param([int]$TargetProcessId)

    $process = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }

    $children = @()
    try {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $TargetProcessId" -ErrorAction Stop
    } catch {
        Write-Host "Could not inspect child processes for PID $TargetProcessId; stopping the parent only."
    }
    foreach ($child in $children) {
        Stop-ProcessTree -TargetProcessId ([int]$child.ProcessId)
    }

    Stop-Process -Id $TargetProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped PID $TargetProcessId ($($process.ProcessName))"
}

function Stop-FromPidFile {
    param([string]$Name)

    $pidFile = Join-Path $RuntimeDir "$Name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) {
        Write-Host "$Name was not started by the local script."
        return
    }

    $processIdText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($processIdText) {
        Stop-ProcessTree -TargetProcessId ([int]$processIdText)
    }
    Remove-Item -LiteralPath $pidFile -Force
}

Stop-FromPidFile -Name "frontend"
Stop-FromPidFile -Name "backend"

Write-Host "AlphaLineage local servers stopped."
