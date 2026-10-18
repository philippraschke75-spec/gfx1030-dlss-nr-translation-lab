#!/bin/bash
# gpu_rel.sh: release preset (Standard: nr_split, edit_temporal 0.7) on the audit pan + the 2560x1440 flicker harness + ghost tests, base vs new.
H="$(dirname "$0")"; B0=/e/dlss-runtime/build_taudit_base; B1=/e/dlss-runtime/build_taudit
bash "$H/gpu_all.sh" rel 2>&1 | grep -v "^EDITF"
for B in $B0 $B1; do echo "=== ${B##*/} flicker harness (Standard, 2560x1440)"; bash /e/dlss-runtime/v043/src/flk.sh $B Standard 960x540
  echo "=== ${B##*/} ghost N=3"; bash /e/dlss-runtime/tq/t.sh $B ghost 3; echo "=== ${B##*/} moving object N=3"; bash /e/dlss-runtime/tq/t.sh $B gobj 3; done
