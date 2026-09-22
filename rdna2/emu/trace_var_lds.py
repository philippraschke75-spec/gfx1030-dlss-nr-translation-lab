"""Host-only provenance for the eight LDS bytes consumed by flag-16 output.

Run with the arena base reported by the corresponding GPU checkpoint.
Tracks changes, not identical-value writes; does not claim race detection.
"""
import json
import sys
import gfx11emu as E
import run_emu as R
import run_var as V


def main():
    V.ARENA = int(sys.argv[1], 0)
    offset = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x400
    prog = E.load_program(R.DIS, {'_Z10k_swin_varILi32ELb1EEv9VarParams'})
    g, ka = V.build(1, V.make_kernarg(flags=16, grid=(2, 2)), len(V.PTR_FIELDS))
    original = E.Exec._compile
    events = []

    def compile_traced(ex, i):
        fn = original(ex, i)
        pc, op, args, _ = ex.prog[i]
        if not op.startswith(('ds_', 'flat_')):
            return fn
        def run(w):
            before = w.lds[offset:offset + 8].copy()
            mask = int(w.S[E.EXEC])
            fn(w)
            after = w.lds[offset:offset + 8]
            if (before != after).any():
                events.append(dict(pc=hex(pc), op=op, args=args, wave=w.wid,
                                   exec=hex(mask), before=before.tolist(),
                                   after=after.tolist()))
        return run

    E.Exec._compile = compile_traced
    try:
        E.run_workgroup(prog, g, 15616, 256,
                        {0: ka & 0xffffffff, 1: ka >> 32, 14: 0, 15: 0},
                        stop_at=0xc07a0, snap={}, poison=True)
    finally:
        E.Exec._compile = original
    print(json.dumps(dict(arena_base=hex(V.ARENA), offset=hex(offset),
                          flags=16, seed=1, workgroup=[0, 0], events=events), indent=2))


if __name__ == '__main__':
    main()
