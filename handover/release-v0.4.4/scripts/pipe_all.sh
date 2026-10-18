#!/bin/bash
# pipe_all.sh <model WxH> <out>: PIPE matrix on build2
S=/e/dlss-runtime/v044/src; B=/e/dlss-runtime/v044/build2; m=$1; O=$2
C='nr_split=0\nvit_every=1\nedit_temporal=0'
SL=0,1,1,-1,1 INT=1 bash $S/pipe.sh $B $m clean-default "$C\nhistory=0" >> $O 2>&1
SL=2,2,2,-1,1 INT=1 bash $S/pipe.sh $B $m clean-sl2 "$C\nhistory=0" >> $O 2>&1
SL=2,2,2,-1,1 INT=2 bash $S/pipe.sh $B $m clean-sl2-i2 "$C\nhistory=0" >> $O 2>&1
SL=2,2,2,-1,1 INT=2 bash $S/pipe.sh $B $m hist-sl2-i2 "$C\nhistory=1" >> $O 2>&1
SL=0,1,1,-1,1 INT=1 bash $S/pipe.sh $B $m hist-default "$C\nhistory=1" >> $O 2>&1
SL=2,2,2,-1,1 INT=2 bash $S/pipe.sh $B $m standard-sl2-i2 "" >> $O 2>&1
SL=0,1,1,-1,1 INT=1 bash $S/pipe.sh $B $m standard-default "" >> $O 2>&1
echo "pipe_all $m done" >> $O
