@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Cyberpunk-Capture.ps1" -ViaVortex
if errorlevel 1 (
    echo Capture startup failed. Read the message above.
    pause
    exit /b 1
)
echo Use Play in the Vortex window that opened. Capture is automatic if the research path is reached.
pause
