#!/bin/sh
# gpu_split.sh : NOT RUN YET (GPU blocked). Builds swin_var.hip and checks/times k_swin_c256s/c128s (grid z = 2) against
# k_swin_c256/c128 of the same build (their ISA is unchanged by the split code) on all g_A steps; lock + 150 s timeout, 3 repeats.
D="$(dirname "$0")"; sh "$D/bsv.sh" /tmp/swin_split.co > /dev/null || exit 1
SWIN_SPLIT_TEST=256,128 sh "$D/gpu_time_swin.sh" /tmp/swin_split.co /tmp/swin_split.co
