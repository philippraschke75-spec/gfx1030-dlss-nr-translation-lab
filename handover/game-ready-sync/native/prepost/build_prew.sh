#!/bin/bash
# build_prew.sh <outdir>: builds pre/post .co with PREW=0 (scalar weight stream, as before) and PREW=1 (LDS-staged
# MLP weights), plus ISA, and prints resources and load/LDS instruction counts. CPU only.
# Needs post_weights.h and pre_tables.h next to this script (not committed: vendor-derived).
set -e
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-/tmp/prew}"; mkdir -p "$O"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off --cuda-device-only -Wno-ignored-attributes"
for k in post pre; do for w in 0 1; do
  "$R/bin/clang++.exe" $F -DPREW=$w -c -o "$O/${k}_native_w$w.co" ${k}_native.hip
  "$R/bin/clang++.exe" $F -DPREW=$w -S -o "$O/${k}_native_w$w.s" ${k}_native.hip
  echo "== $k PREW=$w"
  grep -E "^\s+\.(vgpr_count|sgpr_count|group_segment_fixed_size|private_segment_fixed_size|vgpr_spill_count|sgpr_spill_count):" "$O/${k}_native_w$w.s" | tr -s ' ' | tr '\n' ' '; echo
  grep -oE "^\s+(s_load\w*|s_buffer_load\w*|global_load\w*|ds_read\w*|scratch_\w*|buffer_load\w*|s_barrier|v_dot2\w*)" "$O/${k}_native_w$w.s" | sort | uniq -c | tr -s ' ' | tr '\n' ';'; echo
done; done
