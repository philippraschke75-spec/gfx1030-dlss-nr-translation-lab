#!/bin/bash
# flk.sh <build dir> <ini file> <model WxH> : flicker harness (as v044/src/flk2.sh) with a given package ini; data -> v042/nr_data; watchdog via run.sh
B=$1; I=$2; s=$3; D=E:/dlss-runtime/v042/nr_data
tr -d '\r' < "$I" | sed -e "s#^data_dir=.*#data_dir=$D#" -e "s#^model_res=.*#model_res=$s#" > $B/nr_port.ini
env NR_TEST_W=${s%x*} NR_TEST_H=${s#*x} NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 \
  /e/dlss-runtime/gpu2.sh --exclusive bash /e/dlss-runtime/v044fix/run.sh $B/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $B)" ${NF:-60} 2>&1 | grep -E "visible|OUTHASH|FAIL|rror|EDIT|HF mean|WATCHDOG|exit" | grep -v "FLICKER frame"
echo "flk $(basename $B) $(basename "$I") $s"
grep -E "ERROR|edit_temporal|history" $B/dlssnr_nr.log | tail -2
