#!/bin/bash
S=/e/dlss-runtime/v044/src
bash $S/tests.sh > $S/tests_v044.txt 2>&1
bash $S/tests2.sh > $S/tests2_v044.txt 2>&1
for s in 960x540 1920x1080 2560x1440; do bash $S/strict.sh /e/dlss-runtime/v044/build $s >> $S/strict_v044.txt 2>&1; done
bash $S/strict.sh /e/dlss-runtime/v044/build 960x540 large >> $S/strict_v044.txt 2>&1
echo "run_tests done" >> $S/strict_v044.txt
