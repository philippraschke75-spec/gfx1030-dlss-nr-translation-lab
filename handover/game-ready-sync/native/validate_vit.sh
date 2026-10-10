#!/bin/bash
# GPU validation + timing of the swizzled native ViT kernels (vit_swz.hip -> native_vit.co). NOT RUN YET: run only when
# the GPU is free. Holds /e/gpu.lock, every step <=150 s. Pass criterion: "on" output byte-identical to "off" (the
# current native ViT kernels from c512.co), and prof lines for k_vit_* lower.
# Needs: build_vit.sh output in $K, a release data dir $D (overrides.txt + kernels\), runtime build $B (build.bat, OUT=$B).
H="$(cd "$(dirname "$0")" && pwd)"
K=${K:-$H}; D=${D:-'E:\dlss-runtime\q_1280x720_vit'}; B=${B:-'E:\dlss-runtime\build_vit'}
DU=$(cygpath -u "$D"); T=$(mktemp -d)
VIT='k_expand2|k_contract2|k_qkv2'
cat "$DU/overrides.txt" > "$T/off.txt"
grep -qE "$VIT" "$T/off.txt" || { echo "no ViT override lines in $D/overrides.txt"; exit 1; }
# same lines (grid, kernarg recipe), module swapped to the new .co
grep -vE "$VIT" "$T/off.txt" > "$T/on.txt"
grep -E "$VIT" "$T/off.txt" | awk -F'|' 'BEGIN{OFS="|"} {$2="kernels\\native_vit.co"; print}' >> "$T/on.txt"
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done; trap 'rmdir /e/gpu.lock; cp "$T/off.txt" "$DU/overrides.txt"' EXIT
cp "$K/native_vit.co" "$DU/kernels/"
# 1) per-step: new kernels vs translated on recorded operands (written-byte set, exact %, time)
sed "s#kernels\\\\#$(cygpath -w "$DU")\\\\kernels\\\\#" "$T/on.txt" | grep -E "$VIT" > "$T/step.txt"
timeout 150 python "$H/stepcheck.py" "$D" "$(cygpath -w "$T/step.txt")" | tail -12
# 2) end to end, 3 runs each, off (old native) vs on (swizzled); outputs must be identical
for r in 1 2 3; do for v in off on; do
  cp "$T/$v.txt" "$DU/overrides.txt"
  NR_PROF=1 timeout 150 python "$H/e2e.py" "$B" "$D" "$(cygpath -w "$T/$v$r.npy")" 8 2> "$T/prof_$v$r.txt" | sed "s/^/$v run $r: /"
  grep -iE "expand|contract|qkv" "$T/prof_$v$r.txt" | sed "s/^/$v run $r: /"
done; python "$H/cmp.py" "$(cygpath -w "$T/off$r.npy")" "$(cygpath -w "$T/on$r.npy")" | head -2; done
# 3) full-frame sync test with the override on
cp "$T/on.txt" "$DU/overrides.txt"
timeout 150 "$(cygpath -u "$B")/test_sync.exe" 'E:\dlss-runtime\color_1280x720.bin' "$B" 300 | tail -3
