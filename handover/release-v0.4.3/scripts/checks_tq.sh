#!/bin/bash
# checks_tq.sh: final v0.4.3 round on build_tq (iq-guides + community-look + temporal-quality)
S=/e/dlss-runtime/v043/src; N=/e/dlss-runtime/v043/build_tq; O=/e/dlss-runtime/v043/base; X='nr_split=0\nvit_every=1\nedit_temporal=0'
echo "== strict (vit_every=1, default sliders, edit_temporal off)"
bash $S/strict.sh $N 960x540 auto; bash $S/strict.sh $N 1920x1080 auto; bash $S/strict.sh $N 960x540 large
echo "== bit-exact vs v0.4.2-hires DLL (deterministic: nr_split=0, vit_every=1, edit_temporal=0, intensity 1)"
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do FW=960 FH=540 bash $S/flk.sh $b Standard 960x540 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 1920x1080 "$X"; done
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 "$X\nmodel=large"; done
echo "== flicker presets (edit_temporal=0.7 = the package)"
for s in 960x540 1920x1080; do for p in Standard Strong Cinematic; do bash $S/flk.sh $N $p $s; done; done
echo "== same, edit_temporal=0 (for the comparison)"
for s in 960x540 1920x1080; do for p in Standard Strong Cinematic; do bash $S/flk.sh $N $p $s 'edit_temporal=0'; done; done
echo "== compose smoke with edit_temporal on"
bash $S/flk.sh $N Community 960x540 'intensity=1.3'
bash $S/flk.sh $N Standard 960x540 'compose=ratio\nintensity=1.3'
echo "== done"
