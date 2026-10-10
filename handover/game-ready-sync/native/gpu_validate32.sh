#!/bin/sh
# NOT RUN YET (GPU was in use). Bounded GPU validation + timing for a c32 native candidate.
# usage: sh gpu_validate32.sh   (builds native_swin32.co into E:/tmp32, per-step exactness, then 3 timing passes)
set -u
N="$(cd "$(dirname "$0")" && pwd)"; O=/e/tmp32; mkdir -p "$O"
sh "$N/build_native.sh" "$O" || exit 1
PY="/e/dlss 5/dlssnr-pytorch/.venv/Scripts/python.exe"
gpu() { until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done; (cd "/e/dlss 5/dlssnr-pytorch" && PYTHONPATH=. timeout 150 "$PY" "$@"); r=$?; rmdir /e/gpu.lock; return $r; }
gpu "$N/check32.py" "$O/native_swin32.co" | tee "$O/check32.log"   # acceptance: every step "exact 100.00%"
for i in 1 2 3; do gpu "$N/time32.py" "$O/native_swin32.co" | tee -a "$O/time32.log"; done
# Full-frame gate afterwards: e2e.py with and without the override line (overrides_c32_optin.txt), then cmp.py must say identical 100.000%.
