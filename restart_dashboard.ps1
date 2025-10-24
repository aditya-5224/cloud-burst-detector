# Restart Dashboard Script
# Stops any running Streamlit processes and starts the dashboard

Write-Host "🔄 Restarting Cloud Burst Prediction Dashboard..." -ForegroundColor Cyan

# Stop any running Streamlit processes
Write-Host "Stopping existing Streamlit processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Navigate to project directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

# Start Streamlit dashboard
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

streamlit run src/dashboard/app.py
