"""One bounded checkpoint dispatch; use via sandbox.py run.

Preserves the original seed-1, 2x2 grid. Optional arguments: PC flags wave lane.
Snapshots are FIRST arrival, which must not be confused with later loop visits.
"""
import hashlib
import json
import sys
import statebisect as B


def main():
    addr = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0xc1eac
    flags = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    wave, lane, wg = (4, 0, (0, 0)) if flags == 16 else (0, 23, (0, 1))
    if len(sys.argv) > 3:
        wave = int(sys.argv[3])
    if len(sys.argv) > 4:
        lane = int(sys.argv[4])
    if not 0 <= wave < 8 or not 0 <= lane < 32:
        raise ValueError('wave/lane outside the 256-thread workgroup')
    ctx = B.Ctx('32_1', flags, 1, wg)
    ctx.grid = (2, 2)
    ctx.rebuild_kernarg()
    preflight = ctx.emu_snap(addr)
    if wave not in preflight:
        raise RuntimeError('target wave does not reach checkpoint; no GPU launch')
    tag = 'operands_%x' % addr
    rc, log, vec, sca = ctx.gpu_dump(addr, tag)
    if vec is None or int(vec[wave, ctx.nreg, 0]) != B.MARK:
        raise RuntimeError('missing GPU checkpoint: %s %s' % (rc, log))
    snap = ctx.emu_snap(addr)
    v, s, scc = snap[wave]
    regs = list(range(ctx.nreg))
    scalars = dict(zip(B.SG_LIST, sca[wave]))
    co = B.W / (tag + '.co')
    report = dict(pc=hex(addr), occurrence='first arrival', grid=[2, 2],
                  workgroup=list(wg), wave=wave, lane=lane, flags=flags, seed=1,
                  module_sha256=hashlib.sha256(co.read_bytes()).hexdigest(),
                  runtime=log, emulator_exec=hex(s[B.E.EXEC]),
                  gpu_exec=hex(int(scalars[100])),
                  vectors={str(r): dict(emu=hex(int(v[r, lane])),
                                       gpu=hex(int(vec[wave, r, lane]))) for r in regs},
                  s13=dict(emu=hex(s[13]), gpu=hex(int(scalars[13]))))
    (B.W / (tag + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
