#!/bin/bash
# Bounded GPU validation + timing of build_asm (global GPU lock, hard 150 s timeout). Not run by the author.
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done
trap 'rmdir /e/gpu.lock' EXIT
timeout 150 python -B "$(dirname "$0")/gpu_validate.py"
echo "exit $?"
