#!/bin/bash
# matrix.sh <build dir> [frames] [dump dir]: synthetic HDR scene at scene white 1/4/15/36 (NR_TEST_WHITE), RGBA16F 2560x1440, hdr_mode=auto,
# p1 / p2raw / p2clamp. Metrics are normalised by the scene scale, so an exposure-invariant DLL prints the same numbers in every row.
B=$1; N=${2:-12}; O=${3:-E:/dlss-runtime/hdrexpdump}; D=$(dirname "$0")/../hdr-fix; mkdir -p "$O"
for w in 1 4 15 36; do for v in p1 p2raw p2clamp; do
  p=1; c=0; [ $v != p1 ] && p=2; [ $v = p2clamp ] && c=1
  printf "white %-2s %-7s " $w $v
  SRC=E:/dlss-runtime/color_1920x1080.bin bash "$D/run.sh" "$B" 'hdr_mode=auto\n' "$N" NR_TEST_HDR=$N NR_TEST_PASSES=$p NR_TEST_P2CLAMP=$c NR_TEST_FW=2560 NR_TEST_FH=1440 \
    NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_FMT=rgba NR_TEST_WHITE=$w NR_TEST_HDRDUMP="$O/w${w}_$v" 2>&1 | grep "HDR frame $((N-1))\|FAIL" | sed "s/^HDR frame [0-9]* (passes [0-9]): //"
done; done
