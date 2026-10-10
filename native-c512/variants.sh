#!/bin/sh
# variants.sh OUTDIR name:"CFLAGS" ...  - builds one c512.co per variant and prints VGPR/LDS of the C=512 kernels
D="$(cd "$(dirname "$0")" && pwd)"; O="$1"; shift
for v in "$@"; do
  n=${v%%:*}; f=${v#*:}
  echo "== $n ($f)"
  CFLAGS="$f" sh "$D/build.sh" "$O/$n.co" 2>&1 | grep -A3 "k_conv2_n\|k_qkv_attn_n\|k_ffwd2_n" | grep -v "^--" | paste - - - - | sed 's/  */ /g'
done
