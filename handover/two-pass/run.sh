#!/bin/bash
# run.sh <build> <tag> <passes> [ini lines, \n-separated]: one guarded flicker run (960x540 model, 1920x1080 frame, 40 frames)
# of test_sync with NR_TEST_PASSES=<passes>; log + last-frame dump in E:/dlss-runtime/twopass_runs/<tag>*
B=$1; T=$2; P=$3; X=$4; O=E:/dlss-runtime/twopass_runs; mkdir -p $O
while tasklist | grep -qi fallout4.exe; do sleep 30; done
INI="$(printf 'max_inflight=3\nhistory=1\nvit_every=2\nreuse=bottleneck\n%b\n' "$X")" B=$B D=E:/dlss-runtime/c32fix/nr_data FW=1920 FH=1080 \
  bash "$(dirname "$0")/../game-ready-sync/temporal/run_flicker.sh" 960x540 40 NR_TEST_PASSES=$P NR_TEST_DUMP=$O/$T > $O/$T.txt 2>&1
grep -E "FAIL|HF|FLICKER|PULSE|OUTHASH|native|reused" $O/$T.txt | tail -6
