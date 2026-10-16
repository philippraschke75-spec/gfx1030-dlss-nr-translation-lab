#!/bin/bash
# repeat.sh: 2 more pace runs per config (earlier runs kept as pace_1/2.txt), for run-to-run noise
cd /e/dlss-runtime/sweep; R=src/run.sh
bash $R r468v4 832x468 "vit_every=4"
for i in 1 2; do
  while read -r n s x; do
    cp cfg/$n/pace.txt cfg/$n/pace_$i.txt; bash $R $n $s "$x" pace > /dev/null
  done <<'L'
base 960x540
base_noi8 960x540 int8_swin=0
v3 960x540 vit_every=3
v4 960x540 vit_every=4
r468 832x468
r468v3 832x468 vit_every=3
r468v4 832x468 vit_every=4
r432 768x432
r432v3 768x432 vit_every=3
r432v4 768x432 vit_every=4
r360 640x360
r360v4 640x360 vit_every=4
nr3 960x540 nr_split=0\nnr_every=3\nmv_reuse=1
r432nr3 768x432 nr_split=0\nnr_every=3\nmv_reuse=1
L
done
