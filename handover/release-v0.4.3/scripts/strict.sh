#!/bin/bash
# strict.sh <build dir> <WxH> [model auto|large] [extra ini lines (\n separated)]: strict test_sync (vit_every=1, frame = model size, 60 frames)
# data v042/nr_data (= the v0.4.3 package data); prints PASS/FAIL + OUTHASH
B=$1; s=$2; MW=${s%x*}; MH=${s#*x}; D=E:/dlss-runtime/v042/nr_data
printf '[nr]\ndata_dir=%s\nmodel_res=%s\nlog=dlssnr_nr.log\nsync=1\nmodel=%s\nvit_every=1\n%b\n' "$D" $s ${3:-auto} "$4" > $B/nr_port.ini
NR_TEST_W=$MW NR_TEST_H=$MH /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_$s.bin "$(cygpath -w $B)" 60 | grep -E "frames 60|PASS|FAIL|OUTHASH|NR off|MISMATCH" | head -4
echo "strict $(basename $B) $s ${3:-auto} rc ${PIPESTATUS[0]}"
grep -E "model:|sync: shader|PSO|compose" $B/dlssnr_nr.log | tail -2
