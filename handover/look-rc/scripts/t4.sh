#!/bin/bash
# before numbers: v0.4.3 DLLs + the new test_sync (EDIT line); int8 on real Cyberpunk captures (network alone, iq_run)
S=/e/dlss-runtime/look_build/src; B=/e/dlss-runtime/look_build/before
echo "== before (v0.4.3)" >/dev/null
SC='style=2\nlocal_tone=2\nlocal_structure=1.88\nintensity=2\nnr_split=0\nedit_temporal=0.5'
echo "== before (v0.4.3)"
bash $S/flk2.sh $B Standard 960x540 ''
bash $S/flk2.sh $B Standard 960x540 "$SC\nauto_mask=1"
bash $S/flk2.sh $B Standard 960x540 "$SC\nauto_mask=0"
echo "== int8 on iqg captures (hall + 3 club), settings: standard | showcase mask1 | showcase mask0"
R="E:/dlss 5/.claude/worktrees/agent-a04a35591f1b63ebb/tools/iq_run.py"
for i in 1 0; do NR_INT8_SWIN=$i /e/dlss-runtime/gpu2.sh py -I "$R" ours "E:\\dlss-runtime\\look_build\\build" E:/dlss-runtime/v042/nr_data/ca6f64f1dd88b965/960x540 960 540 \
  E:/dlss-runtime/iqg/in_960x540.npy E:/dlss-runtime/look_build/src/i8_$i.npy "1,1,-1,1,0;2,1.88,-1,1,2;2,1.88,-1,0,2" 2>&1 | tail -2; done
py -I -c "
import numpy as np; x=np.load('E:/dlss-runtime/iqg/in_960x540.npy')
for i in (1,0):
  o=np.load('E:/dlss-runtime/look_build/src/i8_%d.npy'%i); print('int8_swin',i,'edit',[round(float(np.abs(o[s]-x).mean()),5) for s in range(len(o))])
"

