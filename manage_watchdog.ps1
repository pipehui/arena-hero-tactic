[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Start', 'Stop', 'Status', 'Uninstall')]
    [string]$Action = 'Status'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$taskName = 'Arena Hero Tactic Watchdog'
$watchdogPath = Join-Path $PSScriptRoot 'watch_tactic.ps1'
$runnerPath = Join-Path $PSScriptRoot 'run_tactic.ps1'
$keyPath = Join-Path $PSScriptRoot 'key.txt'
$pythonPath = 'D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe'
$tacticPath = Join-Path $PSScriptRoot 'balanced_tactic.py'
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$watchdogLogPath = Join-Path $PSScriptRoot 'logs\watchdog\watchdog.log'

function Get-WatchdogTask {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

function Get-TaskTriggerCount {
    param($Task)

    if ($null -eq $Task -or $null -eq $Task.Triggers) {
        return 0
    }
    return [int]$Task.Triggers.Count
}

function Get-ExactPowerShellProcesses {
    param(
        [Parameter(Mandatory)]
        [string]$ScriptPath
    )

    $escapedPath = [regex]::Escape($ScriptPath)
    @(
        Get-CimInstance Win32_Process -Filter (
            "Name = 'powershell.exe' OR Name = 'pwsh.exe'"
        ) |
            Where-Object { $_.CommandLine -match $escapedPath }
    )
}

function Get-WatchdogProcesses {
    @(Get-ExactPowerShellProcesses -ScriptPath $watchdogPath)
}

function Get-TacticRunnerProcesses {
    @(Get-ExactPowerShellProcesses -ScriptPath $runnerPath)
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

function Test-UnattendedCredential {
    if (Test-Path -LiteralPath $keyPath -PathType Leaf) {
        $fileKey = [IO.File]::ReadAllText($keyPath).Trim(
            [char[]]" `t`r`n`v`f"
        )
        if (-not [string]::IsNullOrWhiteSpace($fileKey)) {
            return $true
        }
    }
    foreach ($scope in @('User', 'Machine')) {
        $storedKey = [Environment]::GetEnvironmentVariable(
            'ARENA_HERO_API_KEY',
            $scope
        )
        if (-not [string]::IsNullOrWhiteSpace($storedKey)) {
            return $true
        }
    }
    return $false
}

function Register-OnDemandWatchdogTask {
    if (-not (Test-Path -LiteralPath $watchdogPath -PathType Leaf)) {
        throw "Watchdog script was not found: $watchdogPath"
    }
    if (-not (Test-UnattendedCredential)) {
        throw (
            'No unattended Arena Hero credential is available. Put the key in ' +
            "$keyPath or set ARENA_HERO_API_KEY for the user account."
        )
    }

    $existingTask = Get-WatchdogTask
    if ($null -ne $existingTask -and $existingTask.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $taskName
    }

    $taskActionParameters = @{
        Execute = $powershellPath
        Argument = (
            '-NoProfile -NonInteractive -WindowStyle Hidden ' +
            '-ExecutionPolicy Bypass -File "' + $watchdogPath + '"'
        )
        WorkingDirectory = $PSScriptRoot
    }
    $taskAction = New-ScheduledTaskAction @taskActionParameters
    $settingsParameters = @{
        AllowStartIfOnBatteries = $true
        DontStopIfGoingOnBatteries = $true
        MultipleInstances = 'IgnoreNew'
        RestartCount = 999
        RestartInterval = (New-TimeSpan -Minutes 1)
        ExecutionTimeLimit = [TimeSpan]::Zero
    }
    $settings = New-ScheduledTaskSettingsSet @settingsParameters
    $principalParameters = @{
        UserId = $currentUser
        LogonType = 'Interactive'
        RunLevel = 'Limited'
    }
    $principal = New-ScheduledTaskPrincipal @principalParameters
    $definitionParameters = @{
        Action = $taskAction
        Settings = $settings
        Principal = $principal
        Description = (
            'On-demand watchdog for the Arena Hero balanced tactic. No automatic trigger.'
        )
    }
    $definition = New-ScheduledTask @definitionParameters
    $registerParameters = @{
        TaskName = $taskName
        InputObject = $definition
        Force = $true
    }
    Register-ScheduledTask @registerParameters | Out-Null
}

function Write-ServiceLog {
    param(
        [Parameter(Mandatory)]
        [string]$Event,
        [string]$Message = ''
    )

    $directory = Split-Path -Parent $watchdogLogPath
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $line = [DateTimeOffset]::Now.ToString('o') + " event=$Event"
    if (-not [string]::IsNullOrWhiteSpace($Message)) {
        $line += ' ' + $Message.Replace("`r", ' ').Replace("`n", ' ')
    }
    [IO.File]::AppendAllText(
        $watchdogLogPath,
        $line + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

function Stop-OnDemandService {
    $task = Get-WatchdogTask
    if ($null -ne $task -and $task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $taskName
    }

    # Ensure an independently launched watchdog cannot restart the tactic while
    # the exact strategy process is being stopped.
    foreach ($process in @(Get-WatchdogProcesses)) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    $tacticProcesses = @(Get-ArenaTacticProcesses)
    $stoppedIds = @($tacticProcesses.ProcessId)
    foreach ($process in $tacticProcesses) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    foreach ($process in @(Get-TacticRunnerProcesses)) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Write-ServiceLog -Event 'SERVICE_STOPPED' -Message (
        'tactic_process_ids=' + ($stoppedIds -join ',')
    )
}

switch ($Action) {
    'Start' {
        $task = Get-WatchdogTask
        if ($null -eq $task -or (Get-TaskTriggerCount $task) -ne 0) {
            # This also migrates the earlier logon-triggered definition.
            Register-OnDemandWatchdogTask
            $task = Get-WatchdogTask
        }
        if ($task.State -ne 'Running') {
            Start-ScheduledTask -TaskName $taskName
            Write-ServiceLog -Event 'SERVICE_STARTED'
            Write-Output 'Arena Hero background service started.'
        }
        else {
            Write-Output 'Arena Hero background service is already running.'
        }
    }
    'Stop' {
        Stop-OnDemandService
        Write-Output 'Arena Hero background service and tactic were stopped.'
    }
    'Status' {
        $task = Get-WatchdogTask
        $watchdogs = @(Get-WatchdogProcesses)
        $tactics = @(Get-ArenaTacticProcesses)
        [pscustomobject]@{
            ServiceState = if ($watchdogs.Count -gt 0) { 'Running' } else { 'Stopped' }
            AutoStartTriggerCount = if ($null -eq $task) {
                0
            }
            else {
                Get-TaskTriggerCount $task
            }
            WatchdogProcessCount = $watchdogs.Count
            WatchdogProcessIds = ($watchdogs.ProcessId -join ',')
            TacticProcessCount = $tactics.Count
            TacticProcessIds = ($tactics.ProcessId -join ',')
            UnattendedCredentialReady = Test-UnattendedCredential
        }
    }
    'Uninstall' {
        Stop-OnDemandService
        if ($null -ne (Get-WatchdogTask)) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
        Write-Output 'Arena Hero background service was stopped and removed.'
    }
}
