#!/bin/bash
# hdr_matrix.sh <build dir> <base ini lines> [frames]: synthetic HDR white-out test, hdr_mode off/auto x passes 1/2 x e5/RGBA16F colour, 2560x1440 frame.
B=$1; BASE=$2; N=${3:-8}; D=$(dirname "$0")
for fmt in e5 rgba; do for m in off auto; do for p in 1 2; do
  echo "== $fmt hdr_mode=$m passes=$p"
  SRC=E:/dlss-runtime/color_1920x1080.bin bash "$D/run.sh" "$B" "${BASE}hdr_mode=$m\n" "$N" NR_TEST_HDR=$N NR_TEST_PASSES=$p NR_TEST_FW=2560 NR_TEST_FH=1440 \
    NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_FMT=$fmt NR_TEST_HDRDUMP="E:/dlss-runtime/hdrdump/${fmt}_${m}_p$p" 2>&1 | grep "HDR frame $((N-1))\|FAIL\|exit"
done; done; done
