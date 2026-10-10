#!/bin/sh
# builds c512.co (or $1) and reports registers / LDS / scratch per kernel
R="/c/Program Files/AMD/ROCm/6.4"
D="$(dirname "$0")"
O="${1:-$D/c512.co}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17 $CFLAGS"
"$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O" "$D/c512.hip" || exit 1
"$R/bin/clang++.exe" $F -I"$R/include" -S -o "$O.s" "$D/c512.hip"
grep -E "^\s+\.name:\s+k_|vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" "$O.s"; rm -f "$O.s"
