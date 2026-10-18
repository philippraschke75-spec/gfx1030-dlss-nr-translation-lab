#!/bin/bash
# exact2.sh: run-to-run check of the one config whose hash differed once (resampled + edit_temporal 0.7, Standard 540p)
R=$(dirname "$0")/../release-v0.4.3/scripts; N=E:/dlss-runtime/hdr_transfer/bx; O=E:/dlss-runtime/hdr_transfer/base; X='nr_split=0\nvit_every=1'
for b in $O $N $O $N; do bash $R/flk.sh $b Standard 960x540 "$X"; done
echo "== exact2 done"
