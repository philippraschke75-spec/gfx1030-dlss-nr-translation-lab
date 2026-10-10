#!/bin/sh
# builds native_swin32.co (k_swin_c32) into $1 (default: this dir) and reports registers / LDS / scratch
R="/c/Program Files/AMD/ROCm/6.4"
cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
"$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_swin32.co" swin_c32.hip || exit 1
"$R/bin/clang++.exe" $F -I"$R/include" -S -o /tmp/swin32.s swin_c32.hip
grep -E "vgpr_count:|sgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" /tmp/swin32.s
