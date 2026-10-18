#!/bin/bash
# flk2.sh: flk.sh with more output lines (steady flicker, EDIT, HF, LISTMAX). Args as flk.sh: <build dir> <preset> <model WxH> [extra ini lines]
B=$1; p=$2; s=$3; D=E:/dlss-runtime/v042/nr_data; X=$(printf '%b' "$4")
tr -d '\r' < /e/dlss5-v0.4.3/presets/$p.ini | sed -e "s#^data_dir=.*#data_dir=$D#" -e "s#^model_res=.*#model_res=$s#" > $B/nr_port.ini
for kv in $X; do sed -i "/^${kv%%=*}=/d" $B/nr_port.ini; echo "$kv" >> $B/nr_port.ini; done
env NR_TEST_W=${s%x*} NR_TEST_H=${s#*x} NR_TEST_FW=${FW:-2560} NR_TEST_FH=${FH:-1440} NR_TEST_SW=1920 NR_TEST_SH=1080 \
  /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $B)" ${NF:-60} 2>&1 | grep -E "visible|OUTHASH|FAIL|rror|LISTMAX|EDIT|HF mean" | grep -v "FLICKER frame"; r=${PIPESTATUS[0]}
echo "flk2 ${B##*/} $p $s ${4//\\n/ } ${NR_TEST_DROP_PART:+drop=$NR_TEST_DROP_PART} rc $r"
grep -E "ERROR|sync: shader|PSO|edit_temporal|dropped|pass-through" $B/dlssnr_nr.log | tail -3
