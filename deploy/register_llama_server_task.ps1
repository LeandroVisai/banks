# register_llama_server_task.ps1 — registra llama-server como tarea programada
# de Windows que arranca al boot y se reinicia si el proceso muere.
#
# Ejecutar UNA vez como administrador:
#   powershell -ExecutionPolicy Bypass -File deploy\register_llama_server_task.ps1
#
# Gestión:
#   Start-ScheduledTask  -TaskName "banks-llama-server"
#   Stop-ScheduledTask   -TaskName "banks-llama-server"
#   Unregister-ScheduledTask -TaskName "banks-llama-server" -Confirm:$false

$ErrorActionPreference = "Stop"

$taskName  = "banks-llama-server"
$repoRoot  = Split-Path -Parent $PSScriptRoot
$script    = Join-Path $PSScriptRoot "start_llama_server.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $repoRoot

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Force

Write-Host "Tarea '$taskName' registrada. Arrancar ahora: Start-ScheduledTask -TaskName '$taskName'"
