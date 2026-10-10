#!/bin/sh
# builds native_vit.co (k_vit_expand_n, k_vit_contract_n, k_vit_qkv_n with the swizzled LDS layout) into $1
# (default: this dir) and reports registers / LDS / scratch
R="/c/Program Files/AMD/ROCm/6.4"
cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
"$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_vit.co" vit_swz.hip || exit 1
"$R/bin/clang++.exe" $F -I"$R/include" -S -o "$O/native_vit.s" vit_swz.hip
grep -E "^\s+\.name:\s+k_|vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" "$O/native_vit.s"; rm -f "$O/native_vit.s"
