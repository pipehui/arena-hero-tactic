[CmdletBinding()]
param(
    [switch]$Unattended,
    [string]$OutputLogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$environmentName = 'arena-hero-tactic'
$pythonPath = 'D:\Tools\miniconda3\envs\arena-hero-tactic\python.exe'
$tacticPath = Join-Path $PSScriptRoot 'balanced_tactic.py'
$keyPath = Join-Path $PSScriptRoot 'key.txt'
$apiHost = 'api.arenahero.io'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Conda environment '$environmentName' was not found at: $pythonPath"
}

if (-not (Test-Path -LiteralPath $tacticPath -PathType Leaf)) {
    throw "Arena Hero tactic was not found at: $tacticPath"
}

$hadNoProxy = Test-Path Env:NO_PROXY
$originalNoProxy = $env:NO_PROXY
$hadApiKey = Test-Path Env:ARENA_HERO_API_KEY
$originalApiKey = $env:ARENA_HERO_API_KEY
$hadPythonUnbuffered = Test-Path Env:PYTHONUNBUFFERED
$originalPythonUnbuffered = $env:PYTHONUNBUFFERED
$resolvedOutputLogPath = $null

if (-not [string]::IsNullOrWhiteSpace($OutputLogPath)) {
    $resolvedOutputLogPath = if ([IO.Path]::IsPathRooted($OutputLogPath)) {
        [IO.Path]::GetFullPath($OutputLogPath)
    }
    else {
        [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputLogPath))
    }
    $outputDirectory = Split-Path -Parent $resolvedOutputLogPath
    if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    }
}

if ([string]::IsNullOrWhiteSpace($env:ARENA_HERO_API_KEY)) {
    if (Test-Path -LiteralPath $keyPath -PathType Leaf) {
        $apiKey = [IO.File]::ReadAllText($keyPath).Trim(
            [char[]]" `t`r`n`v`f"
        )
        if ([string]::IsNullOrWhiteSpace($apiKey)) {
            throw "Arena Hero API key file is empty: $keyPath"
        }
        $env:ARENA_HERO_API_KEY = $apiKey
    }
    elseif ($Unattended) {
        throw (
            'ARENA_HERO_API_KEY is not set and the unattended key file ' +
            "does not exist: $keyPath"
        )
    }
}

$env:PYTHONUNBUFFERED = '1'
$bypassHosts = @(
    if (-not [string]::IsNullOrWhiteSpace($originalNoProxy)) {
        $originalNoProxy -split ',' |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }
)

if ($bypassHosts -notcontains $apiHost) {
    $bypassHosts += $apiHost
}

$exitCode = 1
$locationPushed = $false

try {
    # Avoid the local HTTP proxy only for Arena Hero while this process is running.
    $env:NO_PROXY = $bypassHosts -join ','

    Push-Location -LiteralPath $PSScriptRoot
    $locationPushed = $true

    Write-Host "Starting Arena Hero tactic..." -ForegroundColor Cyan
    Write-Host "Conda environment: $environmentName"
    Write-Host "Proxy bypass: $apiHost"
    if (-not $Unattended) {
        Write-Host "Press Ctrl+C to stop."
    }

    if ($null -eq $resolvedOutputLogPath) {
        & $pythonPath $tacticPath
    }
    else {
        & $pythonPath $tacticPath *>> $resolvedOutputLogPath
    }
    $exitCode = $LASTEXITCODE
}
finally {
    if ($locationPushed) {
        Pop-Location
    }

    if ($hadNoProxy) {
        $env:NO_PROXY = $originalNoProxy
    }
    else {
        Remove-Item Env:NO_PROXY -ErrorAction SilentlyContinue
    }

    if ($hadApiKey) {
        $env:ARENA_HERO_API_KEY = $originalApiKey
    }
    else {
        Remove-Item Env:ARENA_HERO_API_KEY -ErrorAction SilentlyContinue
    }

    if ($hadPythonUnbuffered) {
        $env:PYTHONUNBUFFERED = $originalPythonUnbuffered
    }
    else {
        Remove-Item Env:PYTHONUNBUFFERED -ErrorAction SilentlyContinue
    }
}

if ($exitCode -ne 0) {
    Write-Host "Tactic stopped with exit code $exitCode." -ForegroundColor Yellow
}

exit $exitCode
