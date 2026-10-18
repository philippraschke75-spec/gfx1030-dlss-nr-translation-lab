#!/bin/bash
# all.sh: the whole look-rc check round, strictly serial (one GPU job at a time via gpu2.sh)
S=/e/dlss-runtime/look_build/src; N=/e/dlss-runtime/look_build/build; B=/e/dlss-runtime/look_build/before
OFF='nr_split=0\nvit_every=1\nstyle=0\nintensity=1\nhistory_gate=1\nhistory_filter=bilinear\nhistory_seed=0\nedit_temporal=0\nhistory_pad=zero'
echo "== strict defaults"; bash $S/strict.sh $N 960x540 auto; bash $S/strict.sh $N 1920x1080 auto
echo "== exact: features off, look vs v0.4.3"
for b in $N $B; do bash $S/flk.sh $b Standard 960x540 "$OFF"; done
for b in $N $B; do bash $S/flk.sh $b Standard 1920x1080 "$OFF"; done
bash $S/t4.sh
SC='style=2\nlocal_tone=2\nlocal_structure=1.88\nintensity=2\nnr_split=0\nedit_temporal=0.5'
echo "== after (look)"
bash $S/flk2.sh $N Standard 960x540 ''
bash $S/flk2.sh $N Standard 960x540 "$SC\nauto_mask=1"
bash $S/flk2.sh $N Standard 960x540 "$SC\nauto_mask=0"
bash $S/flk2.sh $N Standard 960x540 "$SC\nauto_mask=1\nintensity_rule=ratio"
bash $S/flk2.sh $N Standard 960x540 "$SC\nauto_mask=1\nint8_swin=0"
bash $S/flk2.sh $N Standard 960x540 "int8_swin=0"
echo "== flicker (fl) + ghost/gobj, before vs after"
for b in $B $N; do for m in fl ghost gobj; do echo "-- ${b##*/} $m"; bash $S/tt.sh $b $m 3; done; done
echo "== all done"
