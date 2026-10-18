#!/bin/bash
# exact.sh: v0.4.4 vs v0.4.3 DLL+runtime (base), deterministic settings (nr_split=0, vit_every=1, edit_temporal=0, detail=1, intensity 1), OUTHASH
S=/e/dlss-runtime/v044/src; N=/e/dlss-runtime/v044/build; O=/e/dlss-runtime/v044/base; X='nr_split=0\nvit_every=1\nedit_temporal=0'
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do FW=960 FH=540 bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 1920x1080 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 "$X\nmodel=large"; done
echo "== exact done"
