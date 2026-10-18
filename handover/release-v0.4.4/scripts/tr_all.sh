#!/bin/bash
# tr_all.sh <out file> "<WxH> <cond> <int>"...: tr.sh for each case, result lines only
S=/e/dlss-runtime/v044/src; O=$1; shift
for a in "$@"; do bash $S/tr.sh $a 2>&1 | grep -E "ours vs|^tr|Error" >> $O; done
