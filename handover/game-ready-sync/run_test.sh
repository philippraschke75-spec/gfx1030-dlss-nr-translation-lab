#!/bin/bash
# GPU-safe test run: global GPU lock, hard 120 s timeout. Usage: run_test.sh [frames] (env NR_TEST_FMT=r11 for R11G11B10)
B=/e/dlss-runtime/build_sync
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done
timeout ${NR_TO:-120} "$B/test_sync.exe" /e/dlss-runtime/color_1280x720.bin 'E:\dlss-runtime\build_sync' "${1:-300}"
rc=$?
rmdir /e/gpu.lock
echo "exit $rc"
