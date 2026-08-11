[CmdletBinding()]
param()

& (Join-Path $PSScriptRoot 'manage_watchdog.ps1') Stop
