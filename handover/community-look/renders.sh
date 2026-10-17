#!/bin/bash
# renders.sh: the still renders behind the PNGs in this folder (static camera, no noise, release nr_port.ini + the
# community preset). Each run is one guarded GPU job (run.sh -> gpu2.sh, 150 s cap). Dumps go to $OUTD (run.sh).
# Then: make_png.py (see README.md for the exact panel commands).
H=$(dirname "$0"); R="max_inflight=3;history=1;nr_split=3;nr_split_k=0;vit_every=0;reuse=bottleneck;int8_swin=1"
C="intensity=2;style=1;local_structure=2;local_tone=2"; Z=(NR_TEST_NOISE=0 NR_TEST_JIT=0 NR_TEST_VX=0 NR_TEST_VY=0)
A=E:/dlss-runtime/color_1920x1080.bin            # capture A: 1920x1080 half4
# capture B: Cyberpunk club (E:/dlss-capture/20260930_191308/frame_000/color.bin, 1280x720 R11G11B10) -> half4, clipped
# to [0,1]: at frame size = model size the DLL hands the colour to the network unclipped, and values > 1 make speckle
K=E:/dlss-runtime/community/club_clip_1280x720.bin
for cfg in "today|$R" "com_add|$R;$C;compose=add" "com_nv|$R;$C;compose=nv" "com_ratio|$R;$C;compose=ratio"; do
  bash "$H/run.sh" "ini:${cfg#*|}" "A_${cfg%%|*}" 960x540 1920x1080 $A 1920x1080 10 "${Z[@]}" || exit 1
done
bash "$H/run.sh" "ini:$R;$C;compose=add" B_addc 960x540 1280x720 $K 1280x720 10 "${Z[@]}" || exit 1
bash "$H/run.sh" "ini:$R;$C;compose=nv" B_nvc 960x540 1280x720 $K 1280x720 10 "${Z[@]}" || exit 1
# B reference: the same network at the frame's own size (1280x720 step) with NVIDIA's intensity rule = what NVIDIA computes
bash "$H/run.sh" "ini:$R;$C;compose=nv" B_ref720c 1280x720 1280x720 $K 1280x720 10 "${Z[@]}"
