#!/bin/sh
# build.sh [outdir]: the native kernel modules of the plan (gfx1030; every RDNA2 chip loads them, see nr_runtime).
# Kernarg structs come from ../plan/nr_params.h. pre/post need vendor-derived tables (pre_tables.h / post_weights.h,
# never committed); pre is built below only if pre_tables.h is present, post is not built here.
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17 -I../plan"
for m in "swinvar swin_var" "c512 c512" "vit vit_swz" "decup decup" "swin32 swin_c32" "mirror mirror"; do
  set -- $m
  "$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_$1.co" "$2.hip" || exit 1
done
# pre (perf-prepost q8h + 8 waves/SIMD, perf-rest contiguous contract weights) only if the vendor-derived table is here
[ -f pre_tables.h ] && "$R/bin/clang++.exe" $F -I"$R/include" "-DPRE_ATTR=__attribute__((amdgpu_waves_per_eu(8)))" -c -o "$O/native_pre_native.co" pre_native.hip
