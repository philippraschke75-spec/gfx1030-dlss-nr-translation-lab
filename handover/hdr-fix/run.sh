#!/bin/bash
# run.sh <build dir> <extra ini lines> <frames> [VAR=val ...]: one guarded test_sync run (960x540 model, c32fix data), GPU lock + 150 s timeout.
B=$1; INI=$2; N=$3; shift 3
printf '[nr]\ndata_dir=E:\\dlss-runtime\\c32fix\\nr_data\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\nmax_inflight=3\n%b' "$INI" > "$B/nr_port.ini"
while tasklist | grep -qi fallout4.exe; do echo "Fallout4 running, waiting"; sleep 30; done
until mkdir /e/dlss-runtime/gpu.lock 2>/dev/null; do sleep 20; done; trap 'rmdir /e/dlss-runtime/gpu.lock' EXIT
env NR_TEST_W=960 NR_TEST_H=540 "$@" timeout 150 "$B/test_sync.exe" "${SRC:-E:/dlss-runtime/color_960x540.bin}" "$(cygpath -w "$B")" "$N"
echo "exit $?"
