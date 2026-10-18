#!/bin/bash
# flk.sh <build dir> <preset name> <model WxH> [extra ini lines (\n separated)]: flicker harness (test_sync, source color_1920x1080.bin,
# frame FWxFH (default 2560x1440), 60 frames) with the package preset as nr_port.ini (data_dir -> v042/nr_data = the v0.4.3 package data,
# model_res overridden; extra lines replace the preset's keys of the same name)
B=$1; p=$2; s=$3; D=E:/dlss-runtime/v042/nr_data; X=$(printf '%b' "$4")
tr -d '\r' < /e/dlss5-v0.4.3/presets/$p.ini | sed -e "s#^data_dir=.*#data_dir=$D#" -e "s#^model_res=.*#model_res=$s#" > $B/nr_port.ini
for kv in $X; do sed -i "/^${kv%%=*}=/d" $B/nr_port.ini; echo "$kv" >> $B/nr_port.ini; done
env NR_TEST_W=${s%x*} NR_TEST_H=${s#*x} NR_TEST_FW=${FW:-2560} NR_TEST_FH=${FH:-1440} NR_TEST_SW=1920 NR_TEST_SH=1080 \
  /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $B)" 60 2>&1 | grep -E "visible|OUTHASH|FAIL|rror" | tail -3; r=${PIPESTATUS[0]}
echo "flk ${B##*/} $p $s ${4//\\n/ } rc $r"
