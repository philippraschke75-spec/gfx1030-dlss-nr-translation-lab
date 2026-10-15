#!/bin/sh
# builds native_io.co (io_head.hip: k_import_n, k_export_n, k_repack_n, k_attention_n, k_final_head_n,
# k_conv_res_views_n) into $1 (default: this dir) and reports registers / LDS / scratch
R="/c/Program Files/AMD/ROCm/6.4"
cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17"
"$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_io.co" io_head.hip || exit 1
"$R/bin/clang++.exe" $F -I"$R/include" -S -o "$O/native_io.s" io_head.hip
grep -E "^\s+\.name:\s+k_|vgpr_count:|group_segment_fixed_size:|private_segment_fixed_size:" "$O/native_io.s"; rm -f "$O/native_io.s"
