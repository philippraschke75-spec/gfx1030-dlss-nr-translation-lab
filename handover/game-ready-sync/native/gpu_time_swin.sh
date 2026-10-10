#!/bin/sh
# gpu_time_swin.sh base.co test.co : bounded GPU run (lock + 150 s timeout) of cmpvar.py, 3 repeats.
# Env SWIN_NT_TEST passes through. Only run when GPU use is allowed again.
D="$(dirname "$0")"; PY="/e/dlss 5/dlssnr-pytorch/.venv/Scripts/python.exe"
for k in 1 2 3; do
  until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done
  timeout 150 "$PY" "$D/cmpvar.py" "$1" "$2" | tail -4
  rmdir /e/gpu.lock
done
