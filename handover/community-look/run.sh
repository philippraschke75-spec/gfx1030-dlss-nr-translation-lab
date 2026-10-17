#!/bin/bash
# run.sh <tag> <model WxH> <frame WxH> <source.bin> <source WxH> [frames] [VAR=val ...]
# One guarded test_sync flicker-mode run (handover/temporal-2026-10-11/HARNESS.md) through gpu2.sh (150 s cap).
# B = build dir (default E:/dlss-runtime/community/build), D = data dir (default E:/dlss-runtime/v041/nr_data).
# First arg ini:k=v;k=v (or env INI) = extra nr_port.ini lines (compose=..., intensity=..., style=...). Last frame dumped to $OUTD/<tag>_{in,out}.bin.
INI=${INI:-}; case "$1" in ini:*) INI=$(printf "%s" "${1#ini:}" | tr ";" "
"); shift;; esac
tag=$1; m=$2; f=$3; src=$4; s=$5; N=${6:-6}; shift 6
B=${B:-E:/dlss-runtime/community/build}; D=${D:-E:/dlss-runtime/v041/nr_data}; OUTD=${OUTD:-E:/dlss-runtime/community/out}
mkdir -p "$OUTD"
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\n%s\n' "$D" "$m" "$INI" > "$B/nr_port.ini"
env NR_TEST_W=${m%x*} NR_TEST_H=${m#*x} NR_TEST_FW=${f%x*} NR_TEST_FH=${f#*x} NR_TEST_SW=${s%x*} NR_TEST_SH=${s#*x} \
    NR_TEST_DUMP="$OUTD/$tag" "$@" \
    bash /e/dlss-runtime/gpu2.sh "$B/test_sync.exe" "$src" "$(cygpath -w "$B")" "$N" 2>&1 | grep -E "FAIL|HF|FLICKER|PULSE|OUTHASH|state" | tee "$OUTD/$tag.txt"
