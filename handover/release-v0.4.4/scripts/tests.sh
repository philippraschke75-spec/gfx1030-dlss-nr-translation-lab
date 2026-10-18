#!/bin/bash
# tests.sh: v0.4.4 new tests + default-flicker comparison vs v0.4.3 (base). Each GPU step its own gpu2.sh call (< 150 s).
S=/e/dlss-runtime/v044/src; N=/e/dlss-runtime/v044/build; O=/e/dlss-runtime/v044/base
echo "== defaults (package presets) v0.4.4 vs v0.4.3"
for s in 960x540 1920x1080; do for b in $N $O; do bash $S/flk2.sh $b Standard $s; done; done
for b in $N $O; do bash $S/flk2.sh $b Cinematic 960x540; done
echo "== compose=nv (Community) with edit_temporal: detail=1 must equal v0.4.3; detail=2 must differ"
for b in $N $O; do bash $S/flk2.sh $b Community 960x540; done
bash $S/flk2.sh $N Community 960x540 'detail=2'
bash $S/flk2.sh $O Community 960x540 'detail=2'
echo "== dropped nr_split part (runtime NR_TEST_DROP_PART: job 10 / 20 never started)"
NR_TEST_DROP_PART=10 bash $S/flk2.sh $N Standard 960x540
NR_TEST_DROP_PART=20 bash $S/flk2.sh $N Performance 960x540 'model_res=960x540'
echo "== edit_temporal without nr_split / history (0.7 vs 0), NR every frame and nr_every=2 (reuse frames); v0.4.3 for comparison"
for b in $N $O; do for et in 0.7 0; do bash $S/flk2.sh $b Standard 960x540 "nr_split=0\nhistory=0\nedit_temporal=$et"; done; done
for b in $N $O; do for et in 0.7 0; do bash $S/flk2.sh $b Standard 960x540 "nr_split=0\nhistory=0\nnr_every=2\nedit_temporal=$et"; done; done
echo "== fg_mode=1 compose=nv smoke (fg reuse branch with compose)"
bash $S/flk2.sh $N Standard 960x540 'nr_split=0\nfg_mode=1\ncompose=nv\nedit_temporal=0'
bash $S/flk2.sh $N Standard 960x540 'nr_split=0\nfg_mode=1\ncompose=add\nedit_temporal=0'
echo "== format-converting pass-through (model guard: NR off) R11G11B10 / R9G9B9E5 colour -> RGBA16F output, and same-format"
printf '[nr]\ndata_dir=E:/nonexistent_nr_data\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\n' > $N/nr_port.ini
for fm in r11 e5 f16; do
  env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_FMT=$fm NR_TEST_OFMT=f16 /e/dlss-runtime/gpu2.sh $N/test_sync.exe E:/dlss-runtime/color_960x540.bin "$(cygpath -w $N)" 5 2>&1 | grep -E "NR off|frame [0-9]:|PASS|FAIL|device"; echo "passthrough $fm -> f16 rc ${PIPESTATUS[0]}"
done
grep -E "pass-through|model" $N/dlssnr_nr.log | tail -3
echo "== recreate stress (VRAM) + eviction during evaluate"
printf '[nr]\ndata_dir=E:/dlss-runtime/v042/nr_data\nmodel_res=960x540\nlog=dlssnr_nr.log\nsync=1\nvit_every=1\n' > $N/nr_port.ini
env NR_TEST_W=960 NR_TEST_H=540 NR_TEST_RECREATE=12 NR_TEST_EVICTRACE=1 /e/dlss-runtime/gpu2.sh $N/test_sync.exe E:/dlss-runtime/color_960x540.bin "$(cygpath -w $N)" 2>&1 | grep -E "cycle|evict race|PASS|FAIL|after all"; echo "recreate rc ${PIPESTATUS[0]}"
grep -E "evict " $N/dlssnr_nr.log | tail -3
echo "== done"
