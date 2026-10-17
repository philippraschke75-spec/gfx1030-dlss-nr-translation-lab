@echo off
rem baseline DLL (origin/v0.4.2-hires dlssnr_nr.cpp) for the bit-exact compare; headers from the v0.4.3 worktree (unchanged)
set S=E:\dlss 5\.claude\worktrees\agent-a11d625a7dd598f58\handover\game-ready-sync
set OUT=E:\dlss-runtime\v043\base
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%OUT%"
cl /nologo /LD /O2 /EHsc /std:c++17 /I"%S%\dll" "%OUT%\src\dlssnr_nr.cpp" /Fe"%OUT%\nvngx.dll_dlssnr.dll" /link d3d12.lib || exit /b 1
