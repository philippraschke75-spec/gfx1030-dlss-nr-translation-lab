#!/bin/bash
# run_flicker.sh <model WxH> [frames] [VAR=val ...]: one guarded temporal-flicker run of test_sync (see handover/temporal-2026-10-11/HARNESS.md).
# B = build dir (default E:/dlss-runtime/build_temporal), D = data dir (default E:/dlss-runtime/combo/nr_data), FW/FH = frame (default 2560x1440).
# Source texture: E:/dlss-runtime/color_1920x1080.bin. Extra VAR=val go to test_sync (NR_TEST_VX/VY/JIT/NOISE/SPECK/REF/DUMPDIR).
s=$1; MW=${s%x*}; MH=${s#*x}; B=${B:-E:/dlss-runtime/build_temporal}; D=${D:-E:/dlss-runtime/combo/nr_data}; shift; N=${1:-60}; shift
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\n%s' "$D" "$s" "$INI" > "$B/nr_port.ini"
tasklist | grep -qi fallout4.exe && { echo "FALLOUT RUNNING - abort"; exit 99; }
until mkdir /e/dlss-runtime/gpu.lock 2>/dev/null; do sleep 20; done; trap 'rmdir /e/dlss-runtime/gpu.lock' EXIT
env NR_TEST_W=$MW NR_TEST_H=$MH NR_TEST_FW=${FW:-2560} NR_TEST_FH=${FH:-1440} NR_TEST_SW=1920 NR_TEST_SH=1080 "$@" \
  timeout 150 "$B/test_sync.exe" E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w "$B")" "$N"
