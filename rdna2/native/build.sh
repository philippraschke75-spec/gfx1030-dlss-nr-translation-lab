#!/bin/sh
# build.sh [outdir]: the native kernel modules of the plan (gfx1030; every RDNA2 chip loads them, see nr_runtime).
# Kernarg structs come from ../plan/nr_params.h. Not built here: native_pre_native.co / native_post_native.co
# (they need vendor-derived tables, pre_tables.h / post_weights.h, which are not in the repo; v0.3 ships prebuilt ones).
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-.}"
F="-x hip --offload-arch=gfx1030 -O3 -ffp-contract=off -mcumode --cuda-device-only -Wno-ignored-attributes -std=c++17 -I../plan"
for m in "swinvar swin_var" "c512 c512" "vit vit_swz" "decup decup" "swin32 swin_c32" "mirror mirror"; do
  set -- $m
  "$R/bin/clang++.exe" $F -I"$R/include" -c -o "$O/native_$1.co" "$2.hip" || exit 1
done
