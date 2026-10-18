#!/bin/bash
# tests2.sh: v0.4.4 additions - v1 export, self-check hook, bit-exactness vs v0.4.3 with history + edit_temporal (deterministic: nr_split=0, vit_every=1)
S=/e/dlss-runtime/v044/src; N=/e/dlss-runtime/v044/build; O=/e/dlss-runtime/v044/base
echo "== v1 export (AMDNR forwarder path): strict + flicker hash equal to v2"
NR_TEST_V1=1 bash $S/hash.sh $N 960x540
for v in "" 1; do NR_TEST_V1=$v bash $S/flk.sh $N Standard 960x540 'nr_split=0\nvit_every=1'; done
echo "== Standard with history + edit_temporal 0.7 (nr_split=0, vit_every=1): v0.4.4 vs v0.4.3 OUTHASH"
for b in $N $O; do bash $S/flk.sh $b Standard 960x540 'nr_split=0\nvit_every=1'; done
for b in $N $O; do bash $S/flk.sh $b Standard 1920x1080 'nr_split=0\nvit_every=1'; done
echo "== self-check hook (NR_SELFCHECK_TEST=1: output corrupted -> NR off, native frames, evaluate still 1)"
NR_SELFCHECK_TEST=1 bash $S/flk2.sh $N Standard 960x540 'nr_split=0\nvit_every=1'
grep -E "selfcheck" $N/dlssnr_nr.log | tail -2
echo "== tests2 done"
