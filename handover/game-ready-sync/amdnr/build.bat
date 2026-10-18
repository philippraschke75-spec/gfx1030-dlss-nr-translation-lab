@echo off
rem Builds Rdna2NrRuntime.dll + nr_runtime.dll (ROCm 6.4 explicitly: HIP_PATH 7.2 links amdhip64_7.dll) + test_amdnr.exe into %OUT%.
setlocal
if "%OUT%"=="" set OUT=E:\dlss-runtime\amdnr\build
set S=%~dp0..
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4\
if not exist "%OUT%" mkdir "%OUT%"
"C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe" --offload-host-only -std=c++17 -O2 -shared "%S%\runtime\nr_runtime.cpp" -lcabinet -o "%OUT%\nr_runtime.dll" || exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%OUT%"
cl /nologo /LD /O2 /EHsc /std:c++17 /W3 "%S%\amdnr\rdna2_nr_runtime.cpp" /Fe"%OUT%\Rdna2NrRuntime.dll" || exit /b 1
cl /nologo /O2 /EHsc /std:c++17 "%S%\amdnr\test_amdnr.cpp" /Fe"%OUT%\test_amdnr.exe" /link d3d12.lib dxgi.lib || exit /b 1
