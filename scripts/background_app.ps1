# On-demand Task Scheduler launch: its process tree belongs to Windows, not the submitting shell.
param(
    [int]$Port = 8000,
    [switch]$NoBuild,
    [switch]$NoOpen,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$RuntimeDir = Join-Path $RepoRoot '.runtime'
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$hash = [System.Security.Cryptography.SHA256]::Create()
try {
    $workspaceKey = ([BitConverter]::ToString($hash.ComputeHash(
        [Text.Encoding]::UTF8.GetBytes($RepoRoot.ToLowerInvariant())))).Replace('-', '').Substring(0, 12)
} finally { $hash.Dispose() }
$TaskName = "AlphaLineage-$workspaceKey-$Port"
$Description = "AlphaLineage background app for $RepoRoot on port $Port"
$ConfigPath = Join-Path $RuntimeDir "background-$Port.json"
$StatePath = Join-Path $RuntimeDir "server-$Port.json"
$PowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$RepoRoot\start.ps1`" -Port $Port -BackgroundWorker -BackgroundConfig `"$ConfigPath`" -NoOpen"
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($CurrentUser -match 'CodexSandbox') {
    throw 'Background registration must run as the signed-in Windows user, outside the Codex sandbox.'
}
$taskService = New-Object -ComObject Schedule.Service
$taskService.Connect()
$taskFolder = $taskService.GetFolder('\')
$Existing = @(Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)[0]
$UserSid = ([System.Security.Principal.NTAccount]$CurrentUser).Translate(
    [System.Security.Principal.SecurityIdentifier]).Value
$ExistingUserSid = $null
if ($Existing) {
    try {
        $ExistingUserSid = if ($Existing.Principal.UserId -match '^S-1-') {
            ([System.Security.Principal.SecurityIdentifier]$Existing.Principal.UserId).Value
        } else {
            ([System.Security.Principal.NTAccount]$Existing.Principal.UserId).Translate(
                [System.Security.Principal.SecurityIdentifier]).Value
        }
    } catch { }
}
if ($Existing -and ($Existing.Description -ne $Description -or
    $Existing.Actions.Execute -ne $PowerShell -or $Existing.Actions.Arguments -ne $Arguments -or
    $ExistingUserSid -ne $UserSid)) {
    throw "Task $TaskName already exists with a different owner/action; it was not changed."
}
function Get-BackgroundState {
    if (Test-Path -LiteralPath $StatePath) {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if (-not $state.mode) {
            # Earlier launches recorded the mode in the event log, rather than server state.
            $events = Join-Path $RuntimeDir "launcher-$Port.events.jsonl"
            $launch = if (Test-Path -LiteralPath $events) {
                Get-Content -LiteralPath $events | ForEach-Object { $_ | ConvertFrom-Json } |
                    Where-Object { $_.event -eq 'launcher_started' -and $_.launch_id -eq $state.launch_id } |
                    Select-Object -Last 1
            }
            if ($launch.details.background) {
                $state | Add-Member -NotePropertyName mode -NotePropertyValue 'background'
            }
        }
        return $state
    }
    return $null
}
function Get-OwnedBackend($State) {
    if (-not $State -or $State.mode -ne 'background' -or -not $State.backend_creation_filetime) {
        return $null
    }
    try {
        $process = Get-Process -Id $State.backend_pid -ErrorAction Stop
        if ($process.StartTime.ToFileTimeUtc().ToString() -eq $State.backend_creation_filetime) {
            return $process
        }
    } catch { }
    return $null
}
function Get-OwnedHealth($State) {
    if (-not $State -or -not $State.launch_id -or -not (Get-OwnedBackend $State)) { return $null }
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        if ($response.Headers['X-AlphaLineage-Instance'] -eq $State.launch_id -and
            $response.Headers['X-AlphaLineage-Pid'] -eq "$($State.backend_pid)") { return $response }
    } catch { }
    return $null
}
if ($Stop) {
    $state = Get-BackgroundState
    if (-not (Get-OwnedBackend $state) -and (-not $Existing -or $Existing.State -ne 'Running')) {
        Write-Host "The background app on port $Port is already stopped."
        exit 0
    }
    [ordered]@{ time = [DateTime]::UtcNow.ToString('o'); event = 'stop_requested';
        task = $TaskName; launch_id = $state.launch_id } | ConvertTo-Json -Compress |
        Add-Content -LiteralPath (Join-Path $RuntimeDir "launcher-$Port.events.jsonl") -Encoding UTF8
    if (Get-OwnedHealth $state) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/shutdown" -Method Post -TimeoutSec 5 | Out-Null
        } catch {
            [ordered]@{ time = [DateTime]::UtcNow.ToString('o'); event = 'shutdown_request_failed';
                task = $TaskName; launch_id = $state.launch_id; error = $_.Exception.Message } |
                ConvertTo-Json -Compress | Add-Content -LiteralPath (
                    Join-Path $RuntimeDir "launcher-$Port.events.jsonl") -Encoding UTF8
        }
        for ($i = 0; $i -lt 40; $i++) {
            $taskRunning = $Existing -and (Get-ScheduledTask -TaskName $TaskName).State -eq 'Running'
            if (-not $taskRunning -and -not (Get-OwnedBackend $state)) { break }
            Start-Sleep -Milliseconds 250
        }
    }
    if ($Existing -and (Get-ScheduledTask -TaskName $TaskName).State -eq 'Running') {
        # This is the verified task/action, not a process selected merely by its port.
        Stop-ScheduledTask -TaskName $TaskName
    }
    $backend = Get-OwnedBackend $state
    if ($backend) {
        # A terminated launcher can leave the exact owned backend alive after its task ends.
        Stop-Process -Id $backend.Id -Force
        [ordered]@{ time = [DateTime]::UtcNow.ToString('o'); event = 'owned_backend_stopped';
            task = $TaskName; launch_id = $state.launch_id; backend_pid = $backend.Id;
            reason = 'explicit background shutdown' } | ConvertTo-Json -Compress |
            Add-Content -LiteralPath (Join-Path $RuntimeDir "launcher-$Port.events.jsonl") -Encoding UTF8
    }
    Write-Host "Stopped $TaskName. Saved checkpoints remain available."
    exit 0
}
if (($Existing -and $Existing.State -eq 'Running') -or (Get-OwnedBackend (Get-BackgroundState))) {
    Write-Host "The background app is already running at http://localhost:$Port ($TaskName)."
    if (-not $NoOpen -and (Get-OwnedHealth (Get-BackgroundState))) {
        Start-Process "http://localhost:$Port"
    }
    exit 0
}
$Config = [ordered]@{
    repo_root = $RepoRoot; port = $Port; no_build = [bool]$NoBuild
    environment = [ordered]@{
        ALPHALINEAGE_DATA_DIR = $env:ALPHALINEAGE_DATA_DIR
        ALPHALINEAGE_TEMP_DIR = $env:ALPHALINEAGE_TEMP_DIR
    }
}
$Config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
if (-not $Existing) {
    $Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
    $Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $RepoRoot
    $Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action -Principal $Principal -Settings $Settings | Out-Null
}
Start-ScheduledTask -TaskName $TaskName
for ($i = 0; $i -lt 120; $i++) {
    $state = Get-BackgroundState
    if (Get-OwnedHealth $state) {
        Write-Host "Background app is ready at http://localhost:$Port ($TaskName; backend PID $($state.backend_pid))."
        if (-not $NoOpen) { Start-Process "http://localhost:$Port" }
        exit 0
    }
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    if ($i -gt 2 -and (Get-ScheduledTask -TaskName $TaskName).State -ne 'Running') {
        throw "The background app exited (task result $($info.LastTaskResult)); see $RuntimeDir\launcher-$Port.events.jsonl and backend-$Port.log."
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "Background task $TaskName is still starting. Check $RuntimeDir\launcher-$Port.events.jsonl."
