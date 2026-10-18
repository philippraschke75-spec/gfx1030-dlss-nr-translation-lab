#!/bin/bash
# tr.sh <WxH> [COND style,tone,structure,skin,auto] [INT]: our runtime (GPU, nr_run2, no history) vs dlssnr-pytorch-flex (CPU) on color_WxH.bin
# new model (WEIGHTS_CC_SMALL_FP8, ca6f...); prints PSNR ours vs PyTorch, edit correlation and mean |edit| of both
s=$1; W=${s%x*}; H=${s#*x}; T="E:/dlss 5/.claude/worktrees/agent-a96d4a5622d1c6e46/tools/torch_ref.py"; O=E:/dlss-runtime/v044/tr; mkdir -p $O
n=$O/ours_${s}_${2//,/_}${HIST:+_h$HIST}.npy
[ -f $n ] || COND=$2 /e/dlss-runtime/gpu2.sh py -I "$T" ours E:/dlss-runtime/v044/build E:/dlss-runtime/v042/nr_data/ca6f64f1dd88b965/$s $W $H $n || { echo "ours failed"; exit 1; }
cd "/e/dlss 5/dlssnr-pytorch-flex" && PYTHONPATH="E:/dlss 5/dlssnr-pytorch-flex" COND=$2 INT=${3:-1} WEIGHTS_RES=WEIGHTS_CC_SMALL_FP8 "/e/dlss 5/dlssnr-pytorch/.venv/Scripts/python.exe" "$T" torch E:/dlss-runtime/v042/nvngx_dlssnr.dll $W $H $n
echo "tr $s cond=${2:-default} int=${3:-1} rc $?"
