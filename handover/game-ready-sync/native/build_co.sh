#!/bin/sh
# b.sh src.hip out.co [flags]
R="/c/Program Files/AMD/ROCm/6.4"; S=$1; O=$2; shift 2
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
"$R/bin/clang++.exe" $F "$@" -I"$R/include" -c -o "$O" "$S" || exit 1
"$R/bin/clang++.exe" $F "$@" -I"$R/include" -S -o "${O%.co}.s" "$S"
grep -E "^\s+\.name:\s+k_|\.vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" "${O%.co}.s"
