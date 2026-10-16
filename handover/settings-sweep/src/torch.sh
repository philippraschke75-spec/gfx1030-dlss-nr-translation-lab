#!/bin/bash
# torch.sh ours|torch: torch_ref.py per model res (STR=1,1,1,1, int8_swin on, one frame, no history) -> sweep/ours_<res>.npy / PSNR lines
T="E:/dlss 5/.claude/worktrees/agent-a86b6132d75569813/tools/torch_ref.py"; FPY="E:/dlss 5/dlssnr-pytorch/.venv/Scripts/python.exe"
export STR=1,1,1,1 NR_INT8_SWIN=1
for s in 960x540 832x468 768x432 640x360; do export NR_COLOR=E:/dlss-runtime/sweep/color_$s.bin
  if [ "$1" = ours ]; then bash /e/dlss-runtime/gpu2.sh py -I "$T" ours 'E:\dlss-runtime\sweep\build' "E:/dlss-runtime/sweep/nr_data/$s" ${s%x*} ${s#*x} E:/dlss-runtime/sweep/ours_$s.npy 2>&1 | tail -1
  else (PYTHONPATH="E:/dlss 5/dlssnr-pytorch-flex" "$FPY" "$T" torch E:/dlss-runtime/c32fix/nvngx_dlssnr.dll ${s%x*} ${s#*x} E:/dlss-runtime/sweep/ours_$s.npy 2>&1 | tail -1); fi
done
