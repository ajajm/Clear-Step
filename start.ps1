# BlindAssit Navigation Assistant PowerShell Launcher
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  BlindAssit Navigation Assistant" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Starting BlindAssit Navigation Server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop.`n"

Set-Location -Path "$PSScriptRoot\server"
python client_camera_server.py $args
