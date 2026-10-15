#!/bin/bash
# ldr_regress.sh <build dir>: strict PASS (RGBA16F, r11, e5; plain sync, hdr_mode=auto) + flicker harness (history=1, nr_split=1) output hash.
B=$1; D=$(dirname "$0")
for fmt in rgba r11 e5; do echo "== strict $fmt"; bash "$D/run.sh" "$B" 'hdr_mode=auto\n' 30 NR_TEST_FMT=$fmt 2>&1 | grep "frames \|^PASS\|^FAIL\|exit"; done
echo "== flicker history=1 nr_split=1"
SRC=E:/dlss-runtime/color_1920x1080.bin bash "$D/run.sh" "$B" 'history=1\nnr_split=1\nvit_every=2\nreuse=bottleneck\nhdr_mode=auto\n' 40 NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_SPECK=0.9 2>&1 | grep "FHASH\|OUTHASH\|FLICKER\|exit"
