#!/bin/bash
# sweep.sh: all configs (name res "ini extras"); then nr_bench per res x vit_every
R=/e/dlss-runtime/sweep/src/run.sh
while read -r n s x; do [ -n "$n" ] && bash $R $n $s "$x"; done <<'L'
ref 960x540 nr_split=0\nvit_every=1\nint8_swin=0
base_noi8 960x540 int8_swin=0
base 960x540
v3 960x540 vit_every=3
v4 960x540 vit_every=4
r468 832x468
r432 768x432
r360 640x360
nr3 960x540 nr_split=0\nnr_every=3\nmv_reuse=1
r468v3 832x468 vit_every=3
r432v3 768x432 vit_every=3
r432v4 768x432 vit_every=4
r360v4 640x360 vit_every=4
L
