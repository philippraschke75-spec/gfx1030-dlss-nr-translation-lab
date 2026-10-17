#!/bin/bash
# bench_small.sh: nr_bench 640x360 / 1280x720 (new model, vit_every=1, history on), build_tq
B=/e/dlss-runtime/v043/build_tq
for s in 640x360 1280x720; do
  NR_BENCH_IN="E:\\dlss-runtime\\color_$s.bin" /e/dlss-runtime/gpu2.sh --exclusive $B/nr_bench.exe "$(cygpath -w $B)" "$(cygpath -w /e/dlss-runtime/v042/nr_data/ca6f64f1dd88b965/$s)" ${s%x*} ${s#*x} 57 1 1 2>&1 | grep -E "BENCH|rror"
done
