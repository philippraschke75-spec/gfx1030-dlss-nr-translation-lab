#!/bin/sh
# build.sh <outdir>: shipped native_post_native.co (v0.3 flags). No weights inside: k_post_native_prep fills them at
# load from the block70.layer0 record. Add -DBAKED (needs the vendor-derived post_weights.h here) only for A/B checks.
R="/c/Program Files/AMD/ROCm/6.4"; cd "$(dirname "$0")"; O="${1:-.}"; shift
"$R/bin/clang++.exe" -x hip --offload-arch=gfx1030 -O3 -ffp-contract=off --cuda-device-only --no-gpu-bundle-output \
  -Wno-ignored-attributes -I"$R/include" "$@" -c -o "$O/native_post_native.co" post_native.hip
