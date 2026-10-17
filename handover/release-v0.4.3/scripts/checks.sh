#!/bin/bash
# checks.sh: the v0.4.3 test round, each GPU step its own gpu2.sh --exclusive call (<150 s). Output: stdout.
S=/e/dlss-runtime/v043/src; N=/e/dlss-runtime/v043/build; O=/e/dlss-runtime/v043/base
echo "== strict (vit_every=1, default sliders)"
bash $S/strict.sh $N 1920x1080 auto
bash $S/strict.sh $N 960x540 large
echo "== bit-exact vs v0.4.2-hires (OUTHASH, intensity 1)"
bash $S/flk.sh $O Standard 960x540   # build: OUTHASH cd87310eb504a383, visible 1.706 % (first run)
bash $S/flk.sh $N Standard 1920x1080; bash $S/flk.sh $O Standard 1920x1080
FW=960 FH=540 bash $S/flk.sh $N Standard 960x540; FW=960 FH=540 bash $S/flk.sh $O Standard 960x540
bash $S/flk.sh $N Standard 960x540 'model=large'; bash $S/flk.sh $O Standard 960x540 'model=large'
echo "== flicker presets"
for s in 960x540 1920x1080; do for p in Strong Cinematic; do bash $S/flk.sh $N $p $s; done; done
echo "== compose smoke (nv + intensity 1.3, ratio + intensity 1.3)"
bash $S/flk.sh $N Community 960x540 'intensity=1.3'
bash $S/flk.sh $N Standard 960x540 'compose=ratio\nintensity=1.3'
echo "== bench (new model, real colour, history on)"
for s in 640x360 960x540 1280x720 1920x1080 2560x1440; do for ve in 1 3; do
  n=57; [ ${s#*x} -ge 1080 ] && n=25
  r=$(NR_BENCH_IN="E:\\dlss-runtime\\color_$s.bin" /e/dlss-runtime/gpu2.sh --exclusive $N/nr_bench.exe "$(cygpath -w $N)" "$(cygpath -w /e/dlss-runtime/v042/nr_data/ca6f64f1dd88b965/$s)" ${s%x*} ${s#*x} $n $ve 1 2>&1 | grep -E "BENCH|rror|nr_create")
  echo "$s $r"
done; done
echo "== done"
