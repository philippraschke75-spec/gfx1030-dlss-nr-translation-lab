#!/bin/bash
# batch.sh: all measurement runs behind handover/hdr-transfer/README.md (each one guarded GPU job via run.sh -> gpu2.sh).
# Stills: static camera, no noise, 12 frames, frame 2560x1440; flicker: the harness defaults (pan, jitter, noise), 60 frames, edit_temporal 0.7.
H=$(dirname "$0")
DEF="max_inflight=3;history=1;edit_temporal=0.7;nr_split=3;nr_split_k=0;vit_every=0;reuse=bottleneck;int8_swin=1"   # release Standard preset
SHOW="$DEF;style=2;intensity=2;local_tone=2;local_structure=1.88;skin_structure=2;auto_mask=0"                    # the video's Showcase settings
Z=(NR_TEST_NOISE=0 NR_TEST_JIT=0 NR_TEST_VX=0 NR_TEST_VY=0)
declare -A SRC=([club]=E:/dlss-runtime/hdr_transfer/club1_1920x1080.bin [hall]=E:/dlss-runtime/color_1920x1080.bin)
if [ "${1:-stills}" = stills ]; then
for src in club hall; do for m in 960x540 1920x1080; do for cfg in def show; do for t in 0 0.5 1; do
  [ $cfg = def ] && I=$DEF || I=$SHOW
  bash "$H/run.sh" "${src}_${m}_${cfg}_t$t" $m "${SRC[$src]}" 1920x1080 12 "$I;transfer_strength=$t" "${Z[@]}"
done; done; done; done
fi
if [ "${1:-flicker}" = flicker ]; then
for cfg in def show; do for t in 0 0.5 1; do
  [ $cfg = def ] && I=$DEF || I=$SHOW
  bash "$H/run.sh" "flk_${cfg}_t$t" 960x540 E:/dlss-runtime/color_1920x1080.bin 1920x1080 60 "$I;transfer_strength=$t" NR_TEST_TS=1
done; done
fi
