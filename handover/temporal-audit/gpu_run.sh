#!/bin/bash
# gpu_run.sh <build dir> <tag> [ini lines, \n separated]: test_sync flicker mode, frame = model 960x540, pan 1.3/0.45 px/frame (as temporal_ref.py),
# 32 frames, per-frame edit strength (EDITF). Preset Standard of v0.4.3 + overrides. One gpu2.sh job (<150 s).
B=$1; T=$2; D=E:/dlss-runtime/v042/nr_data; X=$(printf '%b' "$3"); N=${N:-32}
tr -d '\r' < /e/dlss5-v0.4.3/presets/Standard.ini | sed -e "s#^data_dir=.*#data_dir=$D#" -e "s#^model_res=.*#model_res=960x540#" > $B/nr_port.ini
for kv in $X; do sed -i "/^${kv%%=*}=/d" $B/nr_port.ini; echo "$kv" >> $B/nr_port.ini; done
rm -f $B/dlssnr_nr.log
env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_FW=${FW:-960} NR_TEST_FH=${FH:-540} NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_VX=${VX:-1.3} NR_TEST_VY=${VY:-0.45} NR_TEST_PERFRAME=1 $XENV \
  /e/dlss-runtime/gpu2.sh $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $B)" $N > /e/dlss-runtime/taudit/gpu_$T.txt 2>&1
echo "== $T rc $? : $(grep -E '^EDITF' /e/dlss-runtime/taudit/gpu_$T.txt | awk '{printf "%s ", $3}' | cut -c1-400)"
grep -E "^EDIT |^HF|^FLICKER steady|FAIL" /e/dlss-runtime/taudit/gpu_$T.txt | cut -c1-200
grep -E "history:" $B/dlssnr_nr.log | tail -2
