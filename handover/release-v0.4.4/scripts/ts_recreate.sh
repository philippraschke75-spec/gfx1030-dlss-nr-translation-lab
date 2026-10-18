#!/bin/bash
# ts_recreate.sh <build dir> [race] [cycles]: recreate.sh with timestamps on the DLL log and the test output (into <build>/ts.log)
B=$1; L=$B/dlssnr_nr.log; T=$B/ts.log; : > $L; : > $T
tail -n0 -F $L 2>/dev/null | while IFS= read -r l; do echo "$(date +%T.%N | cut -c1-12) $l"; done >> $T &
P=$!
bash /e/dlss-runtime/v044/src/recreate.sh $B ${2:-0} ${3:-6} | while IFS= read -r l; do echo "$(date +%T.%N | cut -c1-12) TEST $l"; done >> $T
sleep 2; kill $P 2>/dev/null; pkill -f "tail -n0 -F $L" 2>/dev/null
cat $T
