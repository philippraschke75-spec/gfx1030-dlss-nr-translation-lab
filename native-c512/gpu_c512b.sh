#!/bin/sh
# NOT YET RUN. perf-c512b GPU check: builds base (39a6c99) and new c512.co on the CPU, then under /e/gpu.lock with
# timeout 150 per call: 1) cmp_co byte-compare + per-step times for ffwd2 (and all overridden steps), 2) bench.sh.
# Usage: sh gpu_c512b.sh   (B=/e/dlss-runtime/build_c512; bench.sh expects $B/base.co, $B/v1.co, $B/{base,new}/nr_data)
D="$(cd "$(dirname "$0")" && pwd)"; B=/e/dlss-runtime/build_c512
git -C "$D" show 39a6c99:native-c512/c512.hip > "$B/c512_base.hip" || exit 1
cp "$D/c512.hip" "$D/c512.hip.new" && cp "$B/c512_base.hip" "$D/c512.hip" && sh "$D/build.sh" "$B/base.co" >/dev/null
cp "$D/c512.hip.new" "$D/c512.hip" && rm "$D/c512.hip.new" && sh "$D/build.sh" "$B/v1.co" | grep -A3 k_ffwd2_n || exit 1
lock() { until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done; timeout 150 "$@"; rc=$?; rmdir /e/gpu.lock; return $rc; }
ONLY=ffwd2 lock python "$D/cmp_co.py" "$B/base.co" "$B/v1.co" | tail -6   # must say byte-identical; ffwd2 time drop
lock python "$D/cmp_co.py" "$B/base.co" "$B/v1.co" | tail -4
# copy v1.co into $B/new/nr_data as c512.co, then the full-runtime NR_PROF comparison:
sh "$D/bench.sh"
