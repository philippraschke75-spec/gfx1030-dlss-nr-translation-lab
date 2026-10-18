#!/bin/bash
# run_ref.sh <out dir> <frames> <modes...>: CPU PyTorch runs of temporal_ref.py, 3 modes at a time (6 threads each). No GPU.
O=$1; N=$2; shift 2; HERE=$(cd "$(dirname "$0")" && pwd -W)
PY="/e/dlss 5/dlssnr-pytorch/.venv/Scripts/python.exe"; export PYTHONPATH="E:/dlss 5/dlssnr-pytorch-flex" DLSSNR_TRITON=0 OMP_NUM_THREADS=6
mkdir -p "$O"; i=0
for m in "$@"; do
  (cd "/e/dlss 5/dlssnr-pytorch-flex" && "$PY" -c "import torch; torch.set_num_threads(6); import runpy, sys; sys.argv=['t', r'$O', '$m', '$N']; runpy.run_path(r'$HERE/temporal_ref.py', run_name='__main__')" 2>&1 | grep -v '^block' > "$O/$m.log") &
  i=$((i + 1)); [ $((i % 3)) = 0 ] && wait
done
wait; echo done
