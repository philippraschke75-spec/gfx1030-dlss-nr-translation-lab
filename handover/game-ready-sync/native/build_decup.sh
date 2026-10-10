#!/bin/sh
# builds native_decup.co (k_dec_upsample_n) into $1
# (default: this dir) and reports registers / LDS / scratch
R="/c/Program Files/AMD/ROCm/6.4"
cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
for m in decup; do
  "$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_$m.co" $m.hip || exit 1
  "$R/bin/clang++.exe" $F -I"$R/include" -S -o "$O/native_$m.s" $m.hip
  grep -E "^\s+\.name:\s+k_|vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" "$O/native_$m.s"; rm -f "$O/native_$m.s"
done
