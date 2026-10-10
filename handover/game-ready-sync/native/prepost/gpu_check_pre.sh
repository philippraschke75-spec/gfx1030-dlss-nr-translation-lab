#!/bin/bash
# NOT YET RUN (GPU embargo). Bounded GPU check of the new k_pre_native (q8h + single body + no SGPR spills).
# Each GPU step takes /e/gpu.lock and a 150 s timeout (gpu.sh). Build first: sh build.sh (CPU only).
# 1) old vs new pre on g_A x3: median ms + skip/pool bytes identical  2) full frame NR_PROF old vs new.
set -e
H="$(cd "$(dirname "$0")" && pwd)"; B=/e/dlss-runtime/build_prepost; OLD=/e/dlss-runtime/perf10/nr_data/kernels/native_pre_native.co
mkdir -p "$B/out"
for i in 1 2 3; do
  "$H/gpu.sh" python -I "$H/host.py" pre "$OLD" "$B/out/pre_old.npy"
  "$H/gpu.sh" python -I "$H/host.py" pre "$H/pre_native.co" "$B/out/pre_new.npy"
done
python -I -c "import numpy as np; a=np.load('$B/out/pre_old.npy'); b=np.load('$B/out/pre_new.npy'); print('pre output identical:', np.array_equal(a,b), 'diff bytes:', int((a!=b).sum()))"
D=$B/frame_data_pre; rm -rf "$D"; cp -r /e/dlss-runtime/perf10/nr_data "$D"; cp "$H/pre_native.co" "$D/kernels/native_pre_native.co"
for tag in old new; do dd=/e/dlss-runtime/perf10/nr_data; [ $tag = new ] && dd=$D
  for i in 1 2 3; do NR_PROF=1 "$H/gpu.sh" python -I "$H/../e2e.py" "$B" "$(cygpath -w "$dd")" "$B/out/fp_${tag}_$i.npy" 8 2>"$B/out/fp_${tag}_$i.prof" | tail -1
    grep -E "k_pre_native|k_post_native" "$B/out/fp_${tag}_$i.prof"; done; done
python -I -c "import numpy as np; print('frame identical:', np.array_equal(np.load('$B/out/fp_old_1.npy'), np.load('$B/out/fp_new_1.npy')))"
