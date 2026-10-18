#!/bin/bash
# test_amdnr round: each GPU step its own gpu2.sh call (<150 s)
B=/e/dlss-runtime/amdnr/build; D=E:/dlss-runtime/amdnr/build; G=/e/dlss-runtime/gpu2.sh
for m in hist passes2 passes3 abandon refuse refuse_dll; do echo "== $m"; $G $B/test_amdnr.exe $D $m 960 540 2>&1 | grep -E "PASS|FAIL|AbandonJob|adapter|WARP|refused|mismatch"; done
echo "== refuse LUID"; NR_HIP_DEVICE=-1 $G $B/test_amdnr.exe $D refuse 960 540 2>&1 | grep -E "PASS|FAIL"
echo "== resample 1280x720"; $G $B/test_amdnr.exe $D resample 1280 720 2>&1 | grep -E "PASS|FAIL"
echo "== leak 640x360"; $G --exclusive $B/test_amdnr.exe $D leak 640 360 50 2>&1 | grep -E "cycle|PASS|FAIL"
echo "== exact 1920x1080"; $G $B/test_amdnr.exe $D exact 1920 1080 2>&1 | grep -E "PASS|FAIL"
for s in "960 540" "1280 720" "1920 1080"; do echo "== bench $s"; $G --exclusive $B/test_amdnr.exe $D bench $s 2>&1 | grep -E "BENCH|load|FAIL"; done
echo "== bench 960 540 passes 2/3"; for p in 2 3; do $G --exclusive $B/test_amdnr.exe $D bench 960 540 $p 2>&1 | grep -E "BENCH|FAIL"; done
echo "== done"
