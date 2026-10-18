#!/bin/bash
# pipe.sh <build dir> <model WxH> <tag> [extra ini lines]: flicker harness (frame 2560x1440 from color_1920x1080.bin) with the reference NR of every frame (NR_TEST_REF):
# PIPE = |shown - (live + reference edit of the same frame)|. Sliders: SL="style,tone,structure,skin,auto" (ini + reference), INT=intensity. Base: Standard preset.
B=$1; s=$2; D=E:/dlss-runtime/v042/nr_data; X=$(printf '%b' "$4"); SL=${SL:-0,1,1,-1,1}; IFS=, read st t ls sk am <<< "$SL"
tr -d '\r' < /e/dlss5-v0.4.3/presets/Standard.ini | sed -e "s#^data_dir=.*#data_dir=$D#" -e "s#^model_res=.*#model_res=$s#" > $B/nr_port.ini
printf 'style=%s\nlocal_tone=%s\nlocal_structure=%s\nskin_structure=%s\nauto_mask=%s\nintensity=%s\n' $st $t $ls $sk $am ${INT:-1} >> $B/nr_port.ini
for kv in $X; do sed -i "/^${kv%%=*}=/d" $B/nr_port.ini; echo "$kv" >> $B/nr_port.ini; done
env NR_TEST_W=${s%x*} NR_TEST_H=${s#*x} NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_REF=1 NR_TEST_COND=$SL NR_TEST_INT=${INT:-1} \
  /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $B)" ${NF:-24} 2>&1 | grep -E "PIPE|EDIT|steady|network-alone|FAIL|rror"; r=${PIPESTATUS[0]}
echo "pipe $3 model $s sl=$SL int=${INT:-1} ${4//\\n/ } rc $r"
