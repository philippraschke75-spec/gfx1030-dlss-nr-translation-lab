@echo off
rem ROCm 6.4 explicitly (HIP_PATH 7.2 links amdhip64_7.dll). v0.4.3-rc build
set S=E:\dlss 5\.claude\worktrees\agent-a11d625a7dd598f58\handover\game-ready-sync
set OUT=E:\dlss-runtime\v043\build_tq
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4\
if not exist "%OUT%" mkdir "%OUT%"
"C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe" --offload-host-only -std=c++17 -O2 -shared "%S%\runtime\nr_runtime.cpp" -lcabinet -o "%OUT%\nr_runtime.dll" || exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%OUT%"
cl /nologo /O2 /EHsc /std:c++17 "%S%\runtime\nr_bench.cpp" /Fe"%OUT%\nr_bench.exe" || exit /b 1
cl /nologo /LD /O2 /EHsc /std:c++17 "%S%\dll\dlssnr_nr.cpp" /Fe"%OUT%\nvngx.dll_dlssnr.dll" /link d3d12.lib || exit /b 1
cl /nologo /O2 /EHsc /std:c++17 "%S%\dll\test_sync.cpp" /Fe"%OUT%\test_sync.exe" /link d3d12.lib dxgi.lib d3dcompiler.lib || exit /b 1
