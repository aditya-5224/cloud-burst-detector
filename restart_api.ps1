# Script to restart the Cloud Burst API properly
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Cloud Burst API Restart Script" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Stop all python processes running uvicorn
Write-Host "`n[1/3] Stopping old API processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -match "uvicorn") {
        Write-Host "  -> Stopping PID $($_.Id)" -ForegroundColor Gray
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

# Wait for ports to clear
Write-Host "`n[2/3] Waiting for port 8000 to clear..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Check if port is free
$portCheck = netstat -ano | findstr ':8000' | findstr 'LISTENING'
if ($portCheck) {
    Write-Host "  ⚠️  Port 8000 still in use, forcing cleanup..." -ForegroundColor Red
    Start-Sleep -Seconds 2
} else {
    Write-Host "  ✅ Port 8000 is free" -ForegroundColor Green
}

# Start the API
Write-Host "`n[3/3] Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "  API will be available at: http://localhost:8000" -ForegroundColor Green
Write-Host "  Docs available at: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "`n  Press Ctrl+C to stop the server`n" -ForegroundColor Gray
Write-Host "===================================================" -ForegroundColor Cyan

python -m uvicorn src.api.main:app --reload --host localhost --port 8000
