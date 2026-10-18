#!/bin/bash
S=/e/dlss-runtime/v044/src; B=/e/dlss-runtime/v044/build2; O=$S/hist.txt
for o in "" 1; do
  OBJ=$o bash $S/hist.sh $B 960x540 5 >> $O 2>&1
  OBJ=$o bash $S/hist.sh $B 960x540 13 >> $O 2>&1
  OBJ=$o bash $S/hist.sh $B 1920x1080 10 >> $O 2>&1
  OBJ=$o bash $S/hist.sh $B 2560x1440 13 >> $O 2>&1
done
OBJ=1 bash $S/hist.sh $B 2560x1440 13 'local_tone=2\nlocal_structure=2\nstyle=2\nintensity=2' >> $O 2>&1
OBJ=1 bash $S/hist.sh $B 960x540 5 'local_tone=2\nlocal_structure=2\nstyle=2\nintensity=2' >> $O 2>&1
echo "hist_all done" >> $O
