#!/bin/bash
# hash.sh <build dir> <WxH> [model auto|large] [extra ini lines (\n separated)]: strict test_sync, all result lines incl. OUTHASH (env passes through, e.g. NR_TEST_V1=1)
B=$1; s=$2; MW=${s%x*}; MH=${s#*x}; D=E:/dlss-runtime/v042/nr_data
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\nmodel=%s\nvit_every=1\n%b\n' "$D" $s ${3:-auto} "$4" > $B/nr_port.ini
NR_TEST_W=$MW NR_TEST_H=$MH /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_$s.bin "$(cygpath -w $B)" ${NF:-60} | grep -E "v1:|frames [0-9]+:|PASS|FAIL|OUTHASH|NR off|MISMATCH|EDIT"
echo "hash $(basename $B) $s ${3:-auto} ${NR_TEST_V1:+V1} rc ${PIPESTATUS[0]}"
