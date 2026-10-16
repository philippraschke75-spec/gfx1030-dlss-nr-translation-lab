#!/bin/bash
# hdr_matrix.sh <build dir> <base ini lines> [frames] [dump dir]: synthetic HDR white-out test, 2560x1440 frame, e5/RGBA16F colour, hdr_mode off/auto x
# p1 / p2raw (pass 1's RGBA16F output straight into a second feature) / p2clamp (through janblade's ClampProxy first, NR_TEST_P2CLAMP=1).
B=$1; BASE=$2; N=${3:-8}; O=${4:-E:/dlss-runtime/hdrdump}; D=$(dirname "$0"); mkdir -p "$O"
for fmt in e5 rgba; do for m in off auto; do for v in p1 p2raw p2clamp; do
  p=1; c=0; [ $v != p1 ] && p=2; [ $v = p2clamp ] && c=1
  echo "== $fmt hdr_mode=$m $v"
  SRC=E:/dlss-runtime/color_1920x1080.bin bash "$D/run.sh" "$B" "${BASE}hdr_mode=$m\n" "$N" NR_TEST_HDR=$N NR_TEST_PASSES=$p NR_TEST_P2CLAMP=$c NR_TEST_FW=2560 NR_TEST_FH=1440 \
    NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_FMT=$fmt NR_TEST_HDRDUMP="$O/${fmt}_${m}_$v" 2>&1 | grep "HDR frame $((N-1))\|FAIL\|exit"
done; done; done
