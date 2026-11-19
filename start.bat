@echo off
title BlindAssit Navigation Assistant
echo =========================================
echo   BlindAssit Navigation Assistant
echo =========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo Starting BlindAssit Navigation Server...
echo Open in browser: http://localhost:5000
echo.

cd server
python client_camera_server.py %*
pause
