#!/bin/bash
# NOT YET RUN (GPU embargo). Bounded GPU check of the packed-q8 post kernel; each step takes the GPU lock + 150 s timeout.
# 1) q8test: exhaustive rounder check on GPU  2) old vs new post kernel on g_A (time + byte-identical output)
# 3) full frame NR_PROF x3 with the new .co swapped into a copy of the data dir.
set -e
B=/e/dlss-runtime/build_prepost; H="$(dirname "$0")"
"$H/gpu.sh" "$B/bin/q8test.exe"
for i in 1 2 3; do
  "$H/gpu.sh" python -I "$H/host.py" post /e/dlss-runtime/perf10/nr_data/kernels/native_post_native.co "$B/out/chk_old.npy"
  "$H/gpu.sh" python -I "$H/host.py" post "$B/release/nr_data/kernels/native_post_native.co" "$B/out/chk_new.npy"
done
python -I -c "import numpy as np; a=np.load('$B/out/chk_old.npy'); b=np.load('$B/out/chk_new.npy'); print('post output identical:', np.array_equal(a,b))"
D=$B/frame_data; rm -rf "$D"; cp -r /e/dlss-runtime/perf10/nr_data "$D"; cp "$B/release/nr_data/kernels/native_post_native.co" "$D/kernels/"; cp /e/dlss-runtime/perf10/nvngx_dlssnr.dll "$B/"   # weights source next to data dir (local only)
for tag in old new; do dd=/e/dlss-runtime/perf10/nr_data; [ $tag = new ] && dd=$D
  for i in 1 2 3; do NR_PROF=1 "$H/gpu.sh" python -I "$H/../e2e.py" "$B" "$(cygpath -w "$dd")" "$B/out/f_${tag}_$i.npy" 8 2>"$B/out/f_${tag}_$i.prof" | tail -1
    grep -E "k_pre_native|k_post_native" "$B/out/f_${tag}_$i.prof"; done; done
python -I -c "import numpy as np; print('frame identical:', np.array_equal(np.load('$B/out/f_old_1.npy'), np.load('$B/out/f_new_1.npy')))"
