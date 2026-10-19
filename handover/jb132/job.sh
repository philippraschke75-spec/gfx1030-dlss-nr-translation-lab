#!/bin/bash
# usage: gpu2.sh [--exclusive] bash job.sh <exe-name> <cmd...>  - watchdog inside the slot: kills <exe-name> after 140 s
n=$1; shift
for g in Fallout4.exe Cyberpunk2077.exe; do tasklist | grep -qi "$g" && { echo "game $g running - abort"; exit 9; }; done
tasklist | grep -qi "tlou-ii" && { echo "tlou running - abort"; exit 9; }
( sleep 140; taskkill //F //IM "$n" >/dev/null 2>&1; echo "WATCHDOG killed $n" ) & wd=$!
"$@"; rc=$?
kill $wd 2>/dev/null; taskkill //F //IM "$n" >/dev/null 2>&1
exit $rc
