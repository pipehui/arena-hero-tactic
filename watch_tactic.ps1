[CmdletBinding()]
param(
    [ValidateRange(2, 60)]
    [int]$PollSeconds = 5,

    [ValidateRange(1, 300)]
    [int]$InitialRestartDelaySeconds = 5,

    [ValidateRange(5, 900)]
    [int]$MaximumRestartDelaySeconds = 60,

    [ValidateRange(30, 3600)]
    [int]$StableRunSeconds = 300,

    [ValidateRange(60, 3600)]
    [int]$HeartbeatTimeoutSeconds = 240
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$pythonPath = 'D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe'
$tacticPath = Join-Path $PSScriptRoot 'balanced_tactic.py'
$runnerPath = Join-Path $PSScriptRoot 'run_tactic.ps1'
$logDirectory = Join-Path $PSScriptRoot 'logs\watchdog'
$watchdogLogPath = Join-Path $logDirectory 'watchdog.log'
$heartbeatPath = Join-Path $logDirectory 'tactic_heartbeat.json'
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$mutexName = 'Local\ArenaHeroTacticWatchdog-DProjectsArena'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Arena Hero Conda Python was not found: $pythonPath"
}
if (-not (Test-Path -LiteralPath $tacticPath -PathType Leaf)) {
    throw "Arena Hero tactic was not found: $tacticPath"
}
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
    throw "Arena Hero runner was not found: $runnerPath"
}
if (-not (Test-Path -LiteralPath $logDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
}

function Write-WatchdogLog {
    param(
        [Parameter(Mandatory)]
        [string]$Event,
        [string]$Message = ''
    )

    $timestamp = [DateTimeOffset]::Now.ToString('o')
    $safeMessage = $Message.Replace("`r", ' ').Replace("`n", ' ')
    $line = "$timestamp event=$Event"
    if (-not [string]::IsNullOrWhiteSpace($safeMessage)) {
        $line += " $safeMessage"
    }
    [IO.File]::AppendAllText(
        $watchdogLogPath,
        $line + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Get-ArenaTacticProcesses {
    $escapedTacticPath = [regex]::Escape($tacticPath)
    @(
        Get-CimInstance Win32_Process -Filter (
            "Name = 'python.exe' OR Name = 'pythonw.exe'"
        ) |
            Where-Object {
                $_.ExecutablePath -ieq $pythonPath -and
                $_.CommandLine -match $escapedTacticPath
            }
    )
}

$createdNew = $false
$mutex = [Threading.Mutex]::new($true, $mutexName, [ref]$createdNew)
if (-not $createdNew) {
    Write-WatchdogLog -Event 'WATCHDOG_ALREADY_RUNNING'
    $mutex.Dispose()
    exit 0
}

$observedProcessId = $null
$missingSince = $null
$restartAt = $null
$restartDelay = $InitialRestartDelaySeconds
$duplicateSignature = $null
$hasCompletedInitialScan = $false

try {
    Write-WatchdogLog -Event 'WATCHDOG_STARTED' -Message (
        "poll_seconds=$PollSeconds initial_delay=$InitialRestartDelaySeconds " +
        "maximum_delay=$MaximumRestartDelaySeconds"
    )

    while ($true) {
        try {
            $processes = @(Get-ArenaTacticProcesses)
        }
        catch {
            Write-WatchdogLog -Event 'PROCESS_QUERY_FAILED' -Message (
                "error_type=$($_.Exception.GetType().Name)"
            )
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $isInitialScan = -not $hasCompletedInitialScan
        $hasCompletedInitialScan = $true

        if ($processes.Count -gt 1) {
            $signature = (
                $processes.ProcessId |
                    Sort-Object |
                    ForEach-Object { [string]$_ }
            ) -join ','
            if ($signature -ne $duplicateSignature) {
                Write-WatchdogLog -Event 'DUPLICATE_TACTICS_DETECTED' -Message (
                    "process_ids=$signature action=monitor_only"
                )
                $duplicateSignature = $signature
            }
            $missingSince = $null
            $restartAt = $null
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $duplicateSignature = $null

        if ($processes.Count -eq 1) {
            $process = $processes[0]
            if ($observedProcessId -ne $process.ProcessId) {
                $observedProcessId = $process.ProcessId
                Write-WatchdogLog -Event 'TACTIC_RUNNING' -Message (
                    "process_id=$observedProcessId"
                )
            }
            $missingSince = $null
            $restartAt = $null

            $startedAt = if ($process.CreationDate -is [DateTime]) {
                $process.CreationDate
            }
            else {
                [Management.ManagementDateTimeConverter]::ToDateTime(
                    [string]$process.CreationDate
                )
            }

            $heartbeatFresh = $false
            if (Test-Path -LiteralPath $heartbeatPath -PathType Leaf) {
                $heartbeatItem = Get-Item -LiteralPath $heartbeatPath
                $heartbeatFresh = (
                    ([DateTime]::UtcNow - $heartbeatItem.LastWriteTimeUtc).TotalSeconds -lt
                    $HeartbeatTimeoutSeconds
                )
            }
            $processAgeSeconds = ([DateTime]::Now - $startedAt).TotalSeconds
            if (
                $processAgeSeconds -ge $HeartbeatTimeoutSeconds -and
                -not $heartbeatFresh
            ) {
                Write-WatchdogLog -Event 'TACTIC_HEARTBEAT_STALE' -Message (
                    "process_id=$($process.ProcessId) " +
                    "timeout_seconds=$HeartbeatTimeoutSeconds action=restart"
                )
                Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
                $observedProcessId = $null
                $missingSince = $null
                $restartAt = $null
                Start-Sleep -Seconds $PollSeconds
                continue
            }
            if (
                ([DateTime]::Now - $startedAt).TotalSeconds -ge $StableRunSeconds
            ) {
                $restartDelay = $InitialRestartDelaySeconds
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if ($null -ne $observedProcessId) {
            Write-WatchdogLog -Event 'TACTIC_STOPPED' -Message (
                "process_id=$observedProcessId"
            )
            $observedProcessId = $null
        }

        if ($null -eq $missingSince) {
            $missingSince = [DateTimeOffset]::Now
            $scheduledDelay = if ($isInitialScan) { 0 } else { $restartDelay }
            $restartAt = $missingSince.AddSeconds($scheduledDelay)
            Write-WatchdogLog -Event 'RESTART_SCHEDULED' -Message (
                "delay_seconds=$scheduledDelay"
            )
        }

        if ([DateTimeOffset]::Now -ge $restartAt) {
            # Recheck immediately before launch so a manually started tactic is
            # adopted instead of racing it with a second process.
            $processes = @(Get-ArenaTacticProcesses)
            if ($processes.Count -eq 0) {
                $runtimeLogPath = Join-Path $logDirectory (
                    'tactic_' + [DateTime]::Now.ToString('yyyyMMdd_HHmmss') +
                    '.log'
                )
                $arguments = @(
                    '-NoProfile',
                    '-NonInteractive',
                    '-ExecutionPolicy',
                    'Bypass',
                    '-File',
                    ('"' + $runnerPath + '"'),
                    '-Unattended',
                    '-OutputLogPath',
                    ('"' + $runtimeLogPath + '"')
                )
                $startParameters = @{
                    FilePath = $powershellPath
                    ArgumentList = $arguments
                    WorkingDirectory = $PSScriptRoot
                    WindowStyle = 'Hidden'
                    PassThru = $true
                }
                $launcher = Start-Process @startParameters
                Write-WatchdogLog -Event 'TACTIC_LAUNCHED' -Message (
                    "launcher_process_id=$($launcher.Id) " +
                    "runtime_log=$runtimeLogPath"
                )
                $restartDelay = [Math]::Min(
                    $MaximumRestartDelaySeconds,
                    [Math]::Max(
                        $InitialRestartDelaySeconds,
                        $restartDelay * 2
                    )
                )
            }
            $missingSince = $null
            $restartAt = $null
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
catch {
    Write-WatchdogLog -Event 'WATCHDOG_FAILED' -Message (
        "error_type=$($_.Exception.GetType().Name)"
    )
    throw
}
finally {
    Write-WatchdogLog -Event 'WATCHDOG_STOPPED'
    try {
        $mutex.ReleaseMutex()
    }
    catch [ApplicationException] {
        # The mutex can already be abandoned when Task Scheduler terminates us.
    }
    $mutex.Dispose()
}
