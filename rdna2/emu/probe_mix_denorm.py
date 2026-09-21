"""Host-only hypothesis test against a saved GPU arena. Never launches GPU work.

Usage: probe_mix_denorm.py saved_arena.bin [flags=32]
Fixed reproduction: 32,true; seed 1; grid 2x2. The caller must supply an arena
from precisely that fixture. Report hashes so that evidence is identifiable.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import difftest_var as D
import gfx11emu as E
import run_var as V


def main():
    path = Path(sys.argv[1])
    flags = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if len(sys.argv) > 3:
        V.ARENA = int(sys.argv[3], 0)
    data = path.read_bytes()
    output = np.frombuffer(data, np.uint8)
    sym, lds = D.SYMS['32_1']
    rows = []
    try:
        for flush in (False, True):
            E.MIX_F16_INPUT_FLUSH = flush
            ka, init, ref, steps, secs = D.emulate(sym, lds, flags, 1, (2, 2))
            if output.shape != ref.shape:
                raise ValueError('saved arena size differs from fixture')
            rows.append(dict(flush_mix_f16_inputs=flush,
                             first_diffs=[dict(offset=hex(int(i)), ref=int(ref[i]), gpu=int(output[i]))
                                          for i in np.flatnonzero(output != ref)[:12]],
                             mismatching_bytes=int(np.count_nonzero(output != ref)),
                             steps=steps, seconds=secs,
                             reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest()))
    finally:
        E.MIX_F16_INPUT_FLUSH = False
    print(json.dumps(dict(arena_sha256=hashlib.sha256(data).hexdigest(),
                          flags=flags, seed=1, grid=[2, 2],
                          scope='hypothesis only; no gfx1100 silicon validation',
                          results=rows), indent=2))


if __name__ == '__main__':
    main()
