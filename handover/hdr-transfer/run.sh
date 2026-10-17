#!/bin/bash
# run.sh <tag> <model WxH> <source.bin> <source WxH> <frames> <ini lines k=v;k=v> [VAR=val ...]
# One guarded test_sync flicker-mode run (gpu2.sh, 150 s cap), frame 2560x1440 (FW/FH env to change). Last frame dumped to $OUTD/<tag>_{in,out}.bin
# (f32 RGB, frame size). B = build dir (default E:/dlss-runtime/build_transfer), D = data (v042/nr_data = the v0.4.3 package data).
tag=$1; m=$2; src=$3; s=$4; N=$5; INI=$(printf "%s" "$6" | tr ";" "\n"); shift 6
B=${B:-E:/dlss-runtime/build_transfer}; D=${D:-E:/dlss-runtime/v042/nr_data}; OUTD=${OUTD:-E:/dlss-runtime/hdr_transfer/out}
mkdir -p "$OUTD"
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\n%s\n' "$D" "$m" "$INI" > "$B/nr_port.ini"
env NR_TEST_W=${m%x*} NR_TEST_H=${m#*x} NR_TEST_FW=${FW:-2560} NR_TEST_FH=${FH:-1440} NR_TEST_SW=${s%x*} NR_TEST_SH=${s#*x} \
    NR_TEST_DUMP="$OUTD/$tag" "$@" \
    bash /e/dlss-runtime/gpu2.sh "$B/test_sync.exe" "$src" "$(cygpath -w "$B")" "$N" > "$OUTD/$tag.log" 2>&1
echo "$tag rc $? $(grep -E "FLICKER|OUTHASH|FAIL|evaluate GPU|TS " "$OUTD/$tag.log" | tr '\n' ' ' | cut -c1-400)"
