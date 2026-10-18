#!/bin/bash
# recreate.sh <build dir> [race 0|1] [cycles]: recreate stress (VRAM spread) [+ eviction during evaluate], exclusive GPU
B=$1; printf '[nr]\ndata_dir=E:/dlss-runtime/v042/nr_data\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\nvit_every=1\n' > $B/nr_port.ini
R=; [ "${2:-0}" != 0 ] && R=$2
env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_RECREATE=${3:-12} ${R:+NR_TEST_EVICTRACE=$R} /e/dlss-runtime/gpu2.sh --exclusive $B/test_sync.exe E:/dlss-runtime/color_960x540.bin "$(cygpath -w $B)" 2>&1 | grep -E "cycle|evict race|PASS|FAIL|after all"; echo "recreate ${B##*/} race=${2:-0} rc ${PIPESTATUS[0]}"
grep -E "evict |ERROR" $B/dlssnr_nr.log | tail -4
