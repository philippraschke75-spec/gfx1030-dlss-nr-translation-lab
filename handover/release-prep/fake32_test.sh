#!/bin/bash
# Folder with only a 32-bit exe: setup must warn and abort on "2".
D=/e/dlss5-v0.3-prep/fake32; rm -rf $D; mkdir -p $D
cp /c/Windows/SysWOW64/find.exe $D/Launcher.exe
cp /e/dlss5-v0.3-prep/pkg/setup_windows.bat $D/; echo dummy > $D/OptiScaler.dll; echo orig > $D/dxgi.dll
printf '2\r\n\r\n' > $D/ans.txt
printf '@echo off\r\ncd /d E:\\dlss5-v0.3-prep\\fake32\r\ncall E:\\dlss5-v0.3-prep\\fake32\\setup_windows.bat < E:\\dlss5-v0.3-prep\\fake32\\ans.txt > E:\\dlss5-v0.3-prep\\fake32\\out.txt 2>&1\r\n' > $D/run.cmd
timeout 60 cmd //c 'E:\dlss5-v0.3-prep\fake32\run.cmd'
sed -n '12,30p' $D/out.txt; ls $D
