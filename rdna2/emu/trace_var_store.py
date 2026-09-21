"""Host-only trace of the instructions preceding writes to a selected arena byte.

Usage: trace_var_store.py [flags=32] [arena_offset=0xb00046]
Uses the seed-1 2x2 synthetic fixture. Does not launch or modify GPU code.
"""
import collections
import json
import re
import sys
import numpy as np
import gfx11emu as E
import run_var as V
import run_emu as R


def main():
    flags = int(sys.argv[1], 0) if len(sys.argv) > 1 else 32
    offset = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0xb00046
    if '--device-base' in sys.argv:
        V.ARENA = int(sys.argv[sys.argv.index('--device-base') + 1], 0)
    def option(name, default):
        return int(sys.argv[sys.argv.index(name) + 1], 0) if name in sys.argv else default
    seed = option('--seed', 1)
    height, width = option('--height', 16), option('--width', 16)
    gx, gy = option('--gx', 2), option('--gy', 2)
    length = option('--length', 1)
    prog = E.load_program(R.DIS, {'_Z10k_swin_varILi32ELb1EEv9VarParams'})
    g, ka = V.build(seed, V.make_kernarg(H=height, W=width, flags=flags, grid=(gx, gy)), len(V.PTR_FIELDS))
    active = {}
    histories = {}
    events = []
    changes = []
    original_compile = E.Exec._compile
    original_write = g.write

    def compile_traced(ex, i):
        fn = original_compile(ex, i)
        addr, op, args, _ = ex.prog[i]
        regs = set(int(x) for x in re.findall(r'\bv(\d+)\b', args))
        for lo, hi in re.findall(r'\bv\[(\d+):(\d+)\]', args):
            regs.update(range(int(lo), int(hi) + 1))
        def run(w):
            key = (*active['group'], w.wid)
            h = histories.setdefault(key, collections.deque(maxlen=20))
            if regs:
                h.append(dict(pc=hex(addr), op=op, args=args,
                              s13=hex(w.S[13]),
                              exec=hex(w.S[E.EXEC]),
                              vectors={r: w.V[r].copy() for r in regs}))
            active['wave'] = w
            active['pc'] = hex(addr)
            before = int(w.V[3, 0])
            inputs = {'v%d' % r: hex(int(w.V[r, 0])) for r in regs}
            fn(w)
            if active['group'] == (1, 0) and w.wid == 1 and before != int(w.V[3, 0]):
                changes.append(dict(pc=hex(addr), op=op, args=args,
                                    before=hex(before), after=hex(int(w.V[3, 0])), inputs=inputs))
        return run

    def write(addr, data):
        target = V.ARENA + offset
        for j, a in enumerate(addr):
            if int(a) < target + length and int(a) + data.shape[1] > target:
                w = active['wave']
                lane = int(np.flatnonzero(w.em)[j])
                trace = []
                for item in histories[(*active['group'], w.wid)]:
                    trace.append({**item, 'vectors': {
                        'v%d' % r: hex(int(vals[lane])) for r, vals in item['vectors'].items()}})
                events.append(dict(group=active['group'], wave=w.wid, lane=lane,
                                   store_pc=active['pc'], arena_offset=hex(int(a)-V.ARENA),
                                   bytes=data[j].tolist(), preceding=trace))
        original_write(addr, data)

    E.Exec._compile = compile_traced
    g.write = write
    try:
        for y in range(gy):
            for x in range(gx):
                if '--group-only' in sys.argv and (x, y) != (0, 1):
                    continue
                active['group'] = (x, y)
                E.run_workgroup(prog, g, 15616, 256,
                                {0: ka & 0xffffffff, 1: ka >> 32, 14: x, 15: y},
                                max_steps=8_000_000)
    finally:
        E.Exec._compile = original_compile
        g.write = original_write
    print(json.dumps(dict(flags=flags, seed=seed, height=height, width=width,
                          grid=[gx, gy], target=hex(offset), length=length, events=events,
                          watched_v3_lane0_changes=changes[-25:]), indent=2))


if __name__ == '__main__':
    main()
