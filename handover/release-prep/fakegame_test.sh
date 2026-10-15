#!/bin/bash
# Install -> uninstall round trip in a fake game folder. No GPU, no real game.
set -u
R=/e/dlss5-v0.3-prep; G=$R/fakegame
snap() { (cd "$G" && find . -type f ! -path './DLSS5_backup/*' -print0 | sort -z | xargs -0 sha256sum); }
rm -rf "$G"; mkdir -p "$G"
cp /e/dlss-runtime/build_v03/test_sync.exe "$G/Game64.exe"
echo "game dxgi proxy (ReShade)" > "$G/dxgi.dll"; echo "old nvngx" > "$G/nvngx.dll"; echo "user weights" > "$G/nvngx_dlssnr.dll"
echo "save data" > "$G/game.ini"
snap > $R/tools/snap_before.txt
# extract the package (OptiScaler\ folder is large; its contents do not matter for the scripts)
cp -r /e/dlss5-v0.2.3/OptiScaler.dll /e/dlss5-v0.2.3/OptiScaler.ini /e/dlss5-v0.2.3/Licenses /e/dlss5-v0.2.3/nr_data \
      /e/dlss5-v0.2.3/nr_port.ini /e/dlss5-v0.2.3/nr_runtime.dll /e/dlss5-v0.2.3/nvngx.dll_dlssnr.dll "$G/"
mkdir -p "$G/OptiScaler"; echo x > "$G/OptiScaler/dummy.txt"
cp $R/pkg/setup_windows.bat $R/pkg/Uninstall_DLSS5.bat "$G/"
# answers: delete old files=1, proxy=Enter(dxgi), overwrite=1, GPU=1(AMD), spoofing=Enter, final pause
timeout 120 cmd //c "E:\dlss5-v0.3-prep\tools\run_bat.cmd setup_windows.bat E:\dlss5-v0.3-prep\tools\ans_setup.txt E:\dlss5-v0.3-prep\tools\setup_out.txt"; echo "setup rc=$?"
echo "--- after setup:"; ls "$G" "$G"/DLSS5_backup/*
timeout 120 cmd //c "E:\dlss5-v0.3-prep\tools\run_bat.cmd Uninstall_DLSS5.bat E:\dlss5-v0.3-prep\tools\ans_uninst.txt E:\dlss5-v0.3-prep\tools\uninst_out.txt"; echo "uninstall rc=$?"
timeout 120 cmd //c "E:\dlss5-v0.3-prep\tools\run_bat.cmd Uninstall_DLSS5.bat E:\dlss5-v0.3-prep\tools\ans_uninst.txt E:\dlss5-v0.3-prep\tools\uninst_out2.txt"; echo "uninstall2 rc=$?"
snap | grep -v -E '\./(setup_windows|Uninstall_DLSS5)\.bat$' > $R/tools/snap_after.txt
echo "--- diff before/after (empty = restored):"; diff $R/tools/snap_before.txt $R/tools/snap_after.txt && echo IDENTICAL
cat $R/tools/snap_after.txt
