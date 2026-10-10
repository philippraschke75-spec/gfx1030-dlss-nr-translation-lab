#!/bin/sh
# bsv.sh out.co [extra clang flags]: build swin_var.hip (k_swin_c64/c128/c256) into out.co, print register/LDS/scratch metadata
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="$1"; shift
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
"$R/bin/clang++.exe" $F "$@" -I"$R/include" -c -o "$O" swin_var.hip || exit 1
"$R/bin/clang++.exe" $F "$@" -I"$R/include" -S -o "${O%.co}.s" swin_var.hip
grep -E "^\s+\.name:\s+k_swin|\.vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:|\.wavefront_size" "${O%.co}.s"
