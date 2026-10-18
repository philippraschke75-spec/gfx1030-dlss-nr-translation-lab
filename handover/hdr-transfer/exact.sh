#!/bin/bash
# exact.sh: transfer_strength=0 (absent) must be bit-exact to v0.4.3-rc (28d1475): OUTHASH of the flicker sequence, our build (bx) vs the
# v0.4.3-rc DLL built from git (base), same preset each, deterministic settings. Then strict test_sync (t=0 PASS; t=1 runs, mismatch by design).
R=$(dirname "$0")/../release-v0.4.3/scripts; N=E:/dlss-runtime/hdr_transfer/bx; O=E:/dlss-runtime/hdr_transfer/base; X='nr_split=0\nvit_every=1'
for b in $N $O; do bash $R/flk.sh $b Standard 960x540 "$X"; done                     # resampled + edit_temporal (preset)
for b in $N $O; do FW=960 FH=540 bash $R/flk.sh $b Standard 960x540 "$X\nedit_temporal=0"; done   # full size
for b in $N $O; do bash $R/flk.sh $b Standard 1920x1080 "$X\nedit_temporal=0"; done   # resampled, 1080p model
for b in $N $O; do bash $R/flk.sh $b Cinematic 960x540; done                          # nr_split=3 + edit_temporal + intensity 1.3 (boost)
for b in $N $O; do bash $R/flk.sh $b Standard 960x540 "edit_temporal=0\nnr_split=0\nmv_reuse=1\nnr_every=2"; done   # mv_reuse
bash $R/strict.sh $N 960x540
bash $R/strict.sh $N 960x540 auto 'transfer_strength=1'
echo "== exact done"
