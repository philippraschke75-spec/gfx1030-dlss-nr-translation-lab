#!/bin/bash
# scan_all.sh: NVIDIA weight byte runs (>= 64 B) in every binary of the v0.4.3 package, both resources of 310.9.16 + 310.8's WEIGHTS_HT
P=/e/dlss5-v0.4.3; S=/e/dlss-runtime/v042/src/scan.py
mapfile -t F < <(find $P/nr_data -type f; echo $P/nr_runtime.dll; echo $P/nvngx.dll_dlssnr.dll)
W=(); for f in "${F[@]}"; do W+=("$(cygpath -w "$f")"); done
for spec in "newmodel WEIGHTS_HT" "newmodel WEIGHTS_CC_SMALL_FP8" "c32fix WEIGHTS_HT"; do set -- $spec
  echo "== $1 $2"; RES=$2 py -I $S E:/dlss-runtime/$1/nvngx_dlssnr.dll "${W[@]}" | awk '/records,/ {print; next} {n++; if ($2 != "0") {bad++; print}} END {print n, "files scanned,", bad+0, "with runs"}'
done
