#!/bin/sh
# build.sh [src.hip]: pre_native.co + .s next to the source, prints register/spill metadata (no GPU)
set -e; cd "$(dirname "$0")"; S=${1:-pre_native.hip}; R="/c/Program Files/AMD/ROCm/6.4"
for x in co s; do f=-c; [ $x = s ] && f=-S
"$R/bin/clang++.exe" -x hip --offload-arch=gfx1030 -O3 -ffp-contract=off --cuda-device-only $f -o "${S%.hip}.$x" "$S" -Wno-ignored-attributes; done
grep -E "^\s+\.(sgpr_count|vgpr_count|sgpr_spill_count|vgpr_spill_count|private_segment_fixed_size):" "${S%.hip}.s"
