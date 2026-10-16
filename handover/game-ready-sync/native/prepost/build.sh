#!/bin/sh
# build.sh <outdir> [extra flags for post]: shipped pre/post .co (perf-prepost + post-noweights).
# pre: q8h rounding, forced 8 waves/SIMD (120 VGPR, no spill); post: PREW=0, 7 waves/SIMD (144 VGPR).
# PREW=1 (LDS-staged MLP weights) is bit-exact on the GPU but slower (pre 1.44 -> 1.77 ms), so it is off.
# pre needs pre_tables.h here (vendor-derived, not committed). post ships no weights: k_post_native_prep fills them
# at load from the block70.layer0 record; -DBAKED (needs vendor-derived post_weights.h) only for A/B checks.
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-.}"; [ $# -gt 0 ] && shift
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17 -cuid=dlssnr"
"$R/bin/clang++.exe" $F -I"$R/include" "-DPRE_ATTR=__attribute__((amdgpu_waves_per_eu(8)))" -c -o "$O/native_pre_native.co" pre_native.hip
"$R/bin/clang++.exe" $F -I"$R/include" -DPREW=0 "-DPOST_ATTR=__attribute__((amdgpu_waves_per_eu(7)))" "$@" -c -o "$O/native_post_native.co" post_native.hip
