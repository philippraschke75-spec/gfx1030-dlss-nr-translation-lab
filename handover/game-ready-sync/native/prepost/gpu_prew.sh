#!/bin/bash
# NOT YET RUN (GPU embargo). Bounded GPU check of PREW (LDS-staged MLP weights): for pre and post, PREW=0 vs PREW=1
# on the g_A frame, 3 alternating runs each (median launch ms), outputs must be byte-identical.
# Every GPU step goes through gpu.sh (machine-wide /e/gpu.lock + timeout 150).
set -e
H="$(cd "$(dirname "$0")" && pwd)"; O=/e/dlss-runtime/build_prew; mkdir -p "$O"
bash "$H/build_prew.sh" "$O" | grep -E "PREW|_count"
for k in pre post; do
  for i in 1 2 3; do for w in 0 1; do
    "$H/gpu.sh" python -I "$H/host.py" $k "$O/${k}_native_w$w.co" "$O/${k}_w$w.npy"
  done; done
  python -I -c "import numpy as np; print('$k PREW=1 output identical:', np.array_equal(np.load('$O/${k}_w0.npy'), np.load('$O/${k}_w1.npy')))"
done
