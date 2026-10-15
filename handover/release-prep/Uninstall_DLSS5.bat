@echo off
REM Uninstall the DLSS5 AMD port (OptiScaler-DLSSNR + nr_runtime) from THIS folder and restore the newest DLSS5_backup.
REM Only touches files in this folder. Never deletes nvngx_dlssnr.dll (the user's own file). Safe to run twice.
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo DLSS5 AMD port - uninstall from:
echo   %CD%
echo.
echo [1] Uninstall and restore backed-up files
echo [2] Cancel
echo.
set /p "UCHOICE=Waiting - "
if not "!UCHOICE!"=="1" ( echo Cancelled. & goto end )

REM 1. OptiScaler's own uninstaller (created by setup_windows.bat), answered with "1"; it deletes itself.
if exist "Remove_OptiScaler.bat" (
    echo.
    echo Running Remove_OptiScaler.bat ...
    > "%TEMP%\dlss5_uninst_answer.txt" echo 1
    cmd /c "%~dp0Remove_OptiScaler.bat" < "%TEMP%\dlss5_uninst_answer.txt" >nul 2>&1
    del /q "%TEMP%\dlss5_uninst_answer.txt" 2>nul
    if exist "Remove_OptiScaler.bat" del /q "Remove_OptiScaler.bat"
)
REM Setup never run: the package's unrenamed OptiScaler files
for %%F in (OptiScaler.dll OptiScaler.ini OptiScaler.log) do if exist "%%F" del /q "%%F"

REM 2. Our files and logs (nvngx_dlssnr.dll is NOT in this list on purpose)
for %%F in (nvngx.dll_dlssnr.dll nr_runtime.dll nr_port.ini dlssnr_nr.log nr_runtime.log) do if exist "%%F" ( del /q "%%F" & echo Removed %%F )
if exist "nr_data\" ( rd /s /q "nr_data" & echo Removed nr_data\ )

REM 3. Restore the newest backup (folder names are yyyyMMdd_HHmmss, so name order = time order)
set "NEWEST="
if exist "DLSS5_backup\" for /f "delims=" %%D in ('dir /b /ad /o-n "DLSS5_backup" 2^>nul') do if not defined NEWEST set "NEWEST=%%D"
if defined NEWEST (
    echo.
    echo Restoring files from DLSS5_backup\!NEWEST! ...
    for %%F in ("DLSS5_backup\!NEWEST!\*") do ( copy /y "%%F" "%%~nxF" >nul & echo Restored %%~nxF )
    echo The backup folder is kept. Delete DLSS5_backup\ yourself once the game works.
) else (
    echo No DLSS5_backup folder - nothing to restore.
)
echo.
echo Done. nvngx_dlssnr.dll ^(your own file^) was left in place.
:end
echo.
pause
