#!/bin/bash
# spike.sh: pace spikes, v0.3 DLL+runtime (c32fix data) vs this build, and this build with hdr_mode=off / log off. Same test_sync (prints spikes).
cd /e/dlss-runtime/sweep; R=src/run.sh; mkdir -p cfg/v03
v03() { d=cfg/v03; cp /e/dlss-runtime/v031rc/base/build/nr_runtime.dll /e/dlss-runtime/v031rc/base/build/nvngx.dll_dlssnr.dll $d/; cp build/test_sync.exe $d/
  printf '[nr]\ndata_dir=E:\\dlss-runtime\\c32fix\\nr_data\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\nmax_inflight=3\nhistory=1\nnr_split=1\nnr_split_k=0\nvit_every=2\nreuse=bottleneck\nbusy_mode=wait\nbusy_wait_ms=100\nhdr_mode=auto\npass2_gain=0.5\n' > $d/nr_port.ini
  env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_FW=2560 NR_TEST_FH=1440 NR_TEST_SW=1920 NR_TEST_SH=1080 NR_TEST_PACE=250 NR_TEST_CPU_MS=8 NR_TEST_GPU_MS=12 \
    bash /e/dlss-runtime/gpu2.sh --exclusive $d/test_sync.exe E:/dlss-runtime/color_1920x1080.bin "$(cygpath -w $d)" 250 > $d/pace_$1.txt 2>&1; grep -h "spikes\|pace (" $d/pace_$1.txt | cut -c1-200; }
for i in 1 2 3; do
  echo "-- v0.3 #$i"; v03 $i
  echo "-- base #$i"; bash $R sp_base 960x540 "" pace | grep -o "pace spikes.*rc=0" | cut -c1-260
  echo "-- hdr off #$i"; bash $R sp_hdroff 960x540 "hdr_mode=off" pace | grep -o "pace spikes.*rc=0" | cut -c1-260
  echo "-- log off #$i"; bash $R sp_logoff 960x540 "log=" pace | grep -o "pace spikes.*rc=0" | cut -c1-260
done
