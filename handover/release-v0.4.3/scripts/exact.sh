#!/bin/bash
# exact.sh: v0.4.3 vs v0.4.2-hires DLL, deterministic settings (nr_split=0, vit_every=1; intensity 1, compose add), OUTHASH of the
# flicker sequence. Run each build twice at 540p first (run-to-run determinism), then 540p full size, 1080p, old model 540p.
S=/e/dlss-runtime/v043/src; N=/e/dlss-runtime/v043/build; O=/e/dlss-runtime/v043/base; X='nr_split=0\nvit_every=1'
for b in $N $N $O $O; do bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do FW=960 FH=540 bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 1920x1080 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 "$X\nmodel=large"; done
echo "== exact done"
