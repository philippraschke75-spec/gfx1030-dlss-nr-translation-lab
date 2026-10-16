#!/bin/bash
# run.sh <name> <WxH> "<ini extra lines>" [pace|ghost|flick|all]: one sweep config, release ini + extras, logs in sweep/cfg/<name>/
# Extras go FIRST: GetPrivateProfile* returns the first occurrence of a key.
# pace : Fallout-like GPU-limited pacing (cpu 8 ms, game GPU load 12 ms, 2560x1440 frame), 250 frames, exclusive GPU
# ghost: same pacing with NR_TEST_GHOST=1 at frame = model size (the ghost check needs that), 120 frames
# flick: flicker harness at 2560x1440, 60 frames, last frame dumped (f32 RGB) for the PSNR / PNG
S=/e/dlss-runtime/sweep; n=$1; s=$2; x=$3; what=${4:-all}; MW=${s%x*}; MH=${s#*x}; d=$S/cfg/$n; mkdir -p $d
cp $S/build/nr_runtime.dll $S/build/nvngx.dll_dlssnr.dll $S/build/test_sync.exe $d/
printf '[nr]\n%b\ndata_dir=E:\\dlss-runtime\\sweep\\nr_data\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\nmax_inflight=3\nhistory=1\nnr_split=1\nnr_split_k=0\nvit_every=2\nreuse=bottleneck\nbusy_mode=wait\nbusy_wait_ms=100\nhdr_mode=auto\npass2_gain=0.5\nint8_swin=1\n' "$x" "$s" > $d/nr_port.ini
W=$(cygpath -w $d)
go() { t=$1; N=$2; shift 2; rm -f $d/dlssnr_nr.log; env NR_TEST_W=$MW NR_TEST_H=$MH "$@" bash /e/dlss-runtime/gpu2.sh --exclusive $d/test_sync.exe $C "$W" $N > $d/$t.txt 2>&1; echo "rc=$?" >> $d/$t.txt; cp $d/dlssnr_nr.log $d/$t.log 2>/dev/null; }
L="NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080"
[[ $what == all || $what == pace ]] && { C=E:/dlss-runtime/color_1920x1080.bin; go pace 250 $L NR_TEST_PACE=250 NR_TEST_CPU_MS=8 NR_TEST_GPU_MS=12; }
[[ $what == all || $what == ghost ]] && { C=E:/dlss-runtime/sweep/color_$s.bin; go ghost 120 NR_TEST_PACE=120 NR_TEST_CPU_MS=8 NR_TEST_GPU_MS=12 NR_TEST_GHOST=1 NR_TEST_OBJ=1; }
[[ $what == all || $what == flick ]] && { C=E:/dlss-runtime/color_1920x1080.bin; go flick 60 $L NR_TEST_DUMP="$W\\last"; }
echo "== $n $s: $(grep -hE 'pace \(cpu|pace spikes|^ghost|^FLICKER frame|rc=' $d/pace.txt $d/ghost.txt $d/flick.txt 2>/dev/null | sed 's/  */ /g' | tr '\n' ' ')"
