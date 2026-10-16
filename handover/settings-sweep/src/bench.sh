#!/bin/bash
# bench.sh: nr_bench network ms (int8_swin on, history on) per res x vit_every (bottleneck reuse = negative)
export NR_INT8_SWIN=1
for s in 960x540 832x468 768x432 640x360; do for v in 1 -2 -3 -4; do
  bash /e/dlss-runtime/gpu2.sh --exclusive /e/dlss-runtime/sweep/build/nr_bench.exe 'E:\dlss-runtime\sweep\build' "E:\dlss-runtime\sweep\nr_data/$s" ${s%x*} ${s#*x} 40 $v 1 2>&1 | grep BENCH
done; done
