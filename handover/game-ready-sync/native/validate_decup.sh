#!/bin/bash
# GPU validation + timing of native k_dec_upsample_n (run only when the GPU is free). Holds /e/gpu.lock, every step <=150 s.
# Needs: build_decup.sh output in $K, a release data dir $D (overrides.txt + kernels\), runtime build $B (build.bat, OUT=$B).
H="$(cd "$(dirname "$0")" && pwd)"
K=${K:-$H}; D=${D:-'E:\dlss-runtime\q_1280x720_vit'}; B=${B:-'E:\dlss-runtime\build_vit'}
LINE='_Z14k_dec_upsample11DecUpParams|kernels\native_decup.co|k_dec_upsample_n|=|4|=|=|424|0<-0:424'
DU=$(cygpath -u "$D"); T=$(mktemp -d)
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done; trap 'rmdir /e/gpu.lock' EXIT
cp "$K/native_decup.co" "$DU/kernels/"
grep -v k_dec_upsample "$DU/overrides.txt" > "$T/off.txt"; { cat "$T/off.txt"; echo "$LINE"; } > "$T/on.txt"
# 1) per-step byte compare vs the translated kernel on the recorded operands (absolute .co path for stepcheck)
sed "s#kernels\\\\#$(cygpath -w "$DU")\\\\kernels\\\\#" "$T/on.txt" | grep k_dec_upsample > "$T/step.txt"
timeout 150 python "$H/stepcheck.py" "$D" "$(cygpath -w "$T/step.txt")" 99 99 | tail -3
# 2) end to end, 3 runs each, off vs on; outputs must be identical
for r in 1 2 3; do for v in off on; do
  cp "$T/$v.txt" "$DU/overrides.txt"
  NR_PROF=1 timeout 150 python "$H/e2e.py" "$B" "$D" "$(cygpath -w "$T/$v$r.npy")" 8 2> "$T/prof_$v$r.txt" | sed "s/^/$v run $r: /"
  grep "prof  99" "$T/prof_$v$r.txt" | sed "s/^/$v run $r: /"
done; python "$H/cmp.py" "$(cygpath -w "$T/off$r.npy")" "$(cygpath -w "$T/on$r.npy")" | head -2; done
# 3) full-frame sync test with the override on
cp "$T/on.txt" "$DU/overrides.txt"
timeout 150 "$(cygpath -u "$B")/test_sync.exe" 'E:\dlss-runtime\color_1280x720.bin' "$B" 300 | tail -3
