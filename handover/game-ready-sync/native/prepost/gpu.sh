#!/bin/bash
# gpu.sh <cmd...>: machine-wide GPU lock + 150 s timeout
until mkdir /e/gpu.lock 2>/dev/null; do sleep 20; done
timeout 150 "$@"; rc=$?
rmdir /e/gpu.lock
exit $rc
