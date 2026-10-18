#!/bin/bash
# t.sh <build dir> <mode fl|ghost|gobj|pace> <nsplit> [extra ini lines via $INI] : one guarded run, release config (v0.4.x nr_port.ini), v041 data
#   fl   : flicker harness, 2560x1440 frame, 960x540 model, 60 frames (sequential; quality only)
#   ghost: 150 frames, 960x540 frame = model, image moves 13 px/frame, pace fo4 (cpu 8 / gpu 12 ms), out vs NR of the same frame
#   gobj : as ghost, moving object (NR_TEST_OBJ, depth 0.5 vs 0.1)
#   pace : 250 frames 2560x1440, cpu 8 / gpu 12 ms, exclusive (cost)
B=$1; md=$2; N=$3; D=E:/dlss-runtime/v042/nr_data; L=$B/last_$md.txt
printf '[nr]\ndata_dir=%s\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\nmodel=%s\nmax_inflight=3\nhistory=1\nnr_split=%s\nnr_split_k=0\nvit_every=%s\nreuse=bottleneck\nbusy_mode=wait\nbusy_wait_ms=100\nhdr_mode=auto\npass2_gain=0.5\n%b' "$D" "${MODEL:-auto}" $N "${VE:-3}" "$INI" > $B/nr_port.ini
rm -f $B/dlssnr_nr.log; W=$(cygpath -w $B)
case $md in
fl)   env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 $XENV \
        /e/dlss-runtime/gpu2.sh $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$W" ${FR:-60} > $L 2>&1
      grep -E "^EDIT|^HF|^FLICKER|^PULSE|^OUTHASH|FAIL" $L | cut -c1-240 ;;
ghost|gobj) [ $md = gobj ] && O=NR_TEST_OBJ=1 || O=
      env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_GHOST=1 $O $XENV NR_TEST_PACE=150 NR_TEST_CPU_MS=8 NR_TEST_GPU_MS=12 \
        /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$W" 150 > $L 2>&1
      grep -E "ghost|dump|pace \(cpu|FAIL" $L | cut -c1-200 ;;
pace) env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_PACE=250 NR_TEST_CPU_MS=${CPU:-8} NR_TEST_GPU_MS=${GPU:-12} \
        /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$W" 250 > $L 2>&1
      grep -E "pace \(cpu|FAIL" $L | cut -c1-240 ;;
esac
grep -E "ERROR|sync: shader" $B/dlssnr_nr.log 2>/dev/null | cut -c1-240
