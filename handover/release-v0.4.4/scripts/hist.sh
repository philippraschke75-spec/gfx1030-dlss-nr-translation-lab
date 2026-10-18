#!/bin/bash
# hist.sh <build> <WxH> <DX> [extra ini]: NR_TEST_HIST temporal test (frame = model, exact MVs, moving object), history on; ghost = "uncovered" vs no-history NR
B=$1; s=$2; D=E:/dlss-runtime/v042/nr_data
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\nmodel=auto\nvit_every=1\nmax_inflight=3\nhistory=1\nnr_split=0\nedit_temporal=0\n%b\n' "$D" $s "$4" > $B/nr_port.ini
env NR_TEST_W=${s%x*} NR_TEST_H=${s#*x} NR_TEST_HIST=${N:-24} NR_TEST_DX=$3 ${OBJ:+NR_TEST_OBJ=1} /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_$s.bin "$(cygpath -w $B)" 2>&1 | grep -E "^HIST|FAIL"
echo "hist $s DX=$3 ${OBJ:+obj} ${4//\\n/ } rc ${PIPESTATUS[0]}"
