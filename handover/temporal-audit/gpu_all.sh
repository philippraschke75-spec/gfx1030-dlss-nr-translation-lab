#!/bin/bash
# gpu_all.sh: the temporal-audit GPU round (each line one gpu2.sh job). base = v0.4.3-rc build, new = this branch.
S="$(dirname "$0")/gpu_run.sh"; s() { bash "$S" "$@"; }; B0=/e/dlss-runtime/build_taudit_base; B1=/e/dlss-runtime/build_taudit
cp $B1/test_sync.exe $B0/   # EDITF print (test-only change)
X='nr_split=0\nedit_temporal=0\nvit_every=1'
case ${1:-net} in
net)
  s $B0 base_h1 "$X"
  s $B1 new_h0 "$X\nhistory=0"
  s $B1 new_h1 "$X"
  s $B1 new_gate "$X\nhistory_gate=1"
  s $B1 new_bil "$X\nhistory_filter=bilinear"
  s $B1 new_seed0 "$X\nhistory_seed=0" ;;
rel)   # release Standard preset (nr_split=3, edit_temporal=0.7, reuse=bottleneck)
  s $B0 base_rel ""
  s $B1 new_rel ""
  s $B1 new_rel_et0 "edit_temporal=0" ;;
esac
