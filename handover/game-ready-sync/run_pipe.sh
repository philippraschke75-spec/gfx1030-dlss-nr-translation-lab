#!/bin/bash
# GPU-safe test_sync run against build_pipe: global GPU lock, hard timeout. Usage: run_pipe.sh <tag> [frames]
# Env passes through (NR_PROF, NR_TEST_W/H/UP/FMT). Output: $B/<tag>.txt (stdout) and $B/<tag>.err (stderr, NR_PROF lines).
B=/e/dlss-runtime/build_pipe
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done
timeout ${NR_TO:-150} "$B/test_sync.exe" /e/dlss-runtime/color_1280x720.bin 'E:\dlss-runtime\build_pipe' "${2:-100}" > "$B/$1.txt" 2> "$B/$1.err"
rc=$?
rmdir /e/gpu.lock
echo "exit $rc"; tail -6 "$B/$1.txt"
