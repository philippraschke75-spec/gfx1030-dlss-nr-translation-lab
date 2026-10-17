#!/bin/bash
# bench2.sh: nr_bench 540p/1080p/1440p (new model, vit_every=1), v0.4.3 build and v0.4.2 hires build back to back (same GPU state)
for s in 960x540 1920x1080 2560x1440; do for B in /e/dlss-runtime/v043/build /e/dlss-runtime/v042/hires/build; do
  n=57; [ ${s#*x} -ge 1080 ] && n=25
  r=$(NR_BENCH_IN="E:\\dlss-runtime\\color_$s.bin" /e/dlss-runtime/gpu2.sh --exclusive $B/nr_bench.exe "$(cygpath -w $B)" "$(cygpath -w /e/dlss-runtime/v042/nr_data/ca6f64f1dd88b965/$s)" ${s%x*} ${s#*x} $n 1 1 2>&1 | grep -E "BENCH|rror|nr_create")
  echo "${B#/e/dlss-runtime/} $r"
done; done
