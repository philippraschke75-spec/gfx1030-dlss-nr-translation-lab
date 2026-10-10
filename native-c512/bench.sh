#!/bin/sh
# NOT YET RUN. Bounded GPU timing for perf-c512: 1) byte-compare + per-step time base vs new kernels on g_A,
# 2) full-runtime NR_PROF base vs new (3 runs each), 3) output compare. Each GPU call holds /e/gpu.lock, timeout 150.
D="$(cd "$(dirname "$0")" && pwd)"; B=/e/dlss-runtime/build_c512
lock() { until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done; timeout 150 "$@"; rc=$?; rmdir /e/gpu.lock; return $rc; }
lock python "$D/cmp_co.py" "$B/base.co" "$B/v1.co" 24 54 | tail -4
for r in 1 2 3; do for t in base new; do echo "== $t run $r"; lock python "$D/prof.py" "$B" "$B/$t/nr_data" "$B/out_$t.npy" 8 | tail -2; done; done
python "$D/../handover/game-ready-sync/native/cmp.py" "$B/out_base.npy" "$B/out_new.npy"
# full-frame: OUT dir has test_sync.exe; point nr_port.ini data_dir at $B/new/nr_data, then: lock "$B/test_sync.exe" /e/dlss-runtime/color_1280x720.bin 'E:\dlss-runtime\build_c512' 300
