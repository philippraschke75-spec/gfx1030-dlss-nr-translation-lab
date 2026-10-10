"""CPU-emulator bit-exactness check: deployed translated module vs rewritten module(s).

Both modules run on the same fixture under gfx10emu (gfx1100 emulator + gfx1030 front end); the full
arena, the kernarg block and final LDS of every workgroup must be byte-identical, and the run must
actually write output. No GPU is used.

  emu_check.py case  BUILD SYMBOL CASE_DIR [LDS]   fixture = astra bounded-tests case (kernarg.bin + initial.bin)
  emu_check.py frame BUILD SYMBOL N [WG ...]       fixture = g_A captured frame, dispatch N of that symbol,
                                                   workgroups 'x,y' (default: first, middle, last)
"""
import sys, json, hashlib, struct, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx10emu as G

DEPLOYED = Path('E:/dlss5-experimental/nr_data/kernels')
KA = 0x7000_0000_0000
ARENA = 0x7500_0000_0000
GA = Path('E:/dlss-vit/g_A')
import os
LDS = os.environ.get('EMU_LDS')        # frame mode: None, 'hw' or a byte count
LDS = LDS if LDS in (None, 'hw') else int(LDS)


def variants(build, sym):
    d = Path(build) / sym
    return [('deployed', DEPLOYED / (sym + '.co')), ('direct', d / 'direct.co'), ('final', d / 'final.co')]


def compare(sym, build, make_mem, wgs, nthreads, lds=None):
    ref, out = None, {}
    for tag, co in variants(build, sym):
        mem, initial = make_mem()
        t = time.monotonic(); steps = 0; ldss = []
        for wg in wgs:
            r, d = G.run(co, sym, mem, KA, wg, nthreads, lds=lds)
            steps += r['steps']; ldss.append(r['lds'].tobytes())
        arena = mem.regions[1].arr
        changed = int(np.count_nonzero(arena != initial))
        assert changed, 'no output written: an empty run is not a pass'
        h = hashlib.sha256()
        for reg in mem.regions[:2]:
            h.update(reg.arr.tobytes())
        for x in ldss:
            h.update(x)
        state = h.hexdigest()
        out[tag] = dict(sha256=state, steps=steps, changed_bytes=changed, seconds=round(time.monotonic() - t, 1))
        for name, f in getattr(make_mem, 'crosscheck', {}).items():   # front-end fidelity, informational
            if f.exists():
                other = np.fromfile(f, np.uint8)
                out[tag]['bytes_differing_from_' + name] = int(np.count_nonzero(other != arena)) if other.size == arena.size else 'size'
        if ref is None:
            ref = (state, arena.copy(), ldss)
        elif state != ref[0]:
            diff = np.flatnonzero(arena != ref[1])
            raise SystemExit('MISMATCH %s %s: %d arena bytes differ, first %s; lds equal=%s'
                             % (sym, tag, diff.size, diff[:8].tolist(), ldss == ref[2]))
        print(' ', tag, out[tag], flush=True)
    return out


def case(build, sym, cdir, lds=None):
    cdir = Path(cdir)
    ka, init = (cdir / 'kernarg.bin').read_bytes(), np.fromfile(cdir / 'initial.bin', np.uint8)

    def mk():
        g = G.E.GMem()
        g.add('kernarg', KA, np.frombuffer(ka, np.uint8).copy())
        g.add('arena', ARENA, init.copy())
        return g, init
    # gfx1100-emulator reference of the original kernel, and astra's RX 6900 XT output of the deployed module
    mk.crosscheck = {'gfx1100_emulator': cdir / 'reference.bin', 'gpu_deployed': cdir / 'c32-deployed.bin'}
    nthreads = struct.unpack_from('<H', ka, 0xB4)[0]
    return compare(sym, build, mk, [(0, 0)], nthreads, lds)


def frame(build, sym, n, wgs=None):
    man = [l.split('|') for l in (GA / 'manifest.txt').read_text().splitlines() if l and not l.startswith('#')]
    rows = [m for m in man if m[1] == sym]
    m = rows[n]
    off, size, gx, gy, nthreads = int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])
    ka = (GA / 'kernargs.bin').read_bytes()[off:off + size]
    if not wgs:
        wgs = sorted({(0, 0), (gx // 2, gy // 2), (gx - 1, gy - 1)})
    arena = np.fromfile(GA / 'arena.bin', np.uint8)

    class Rec(G.E.GMem):
        written = []

        def write(s, addr, data):
            a = np.asarray(addr, np.uint64)
            Rec.written.append(np.unique((a[:, None] + np.arange(data.shape[1], dtype=np.uint64)).ravel()))
            return super().write(addr, data)

    def mk(cls=G.E.GMem):
        g = cls()
        g.add('kernarg', KA, np.frombuffer(ka, np.uint8).copy())
        g.add('arena', ARENA, arena.copy())        # private copy of the 1.4 GB captured frame
        return g, arena
    # The capture already holds this dispatch's results, so a rerun rewrites identical bytes. Probe the
    # deployed module once for the bytes it writes and poison them (0xA5) so real output is visible.
    g, _ = mk(Rec)
    for wg in wgs:
        G.run(DEPLOYED / (sym + '.co'), sym, g, KA, wg, nthreads, lds=LDS)
    w = np.unique(np.concatenate(Rec.written)) - np.uint64(ARENA)
    w = w[w < arena.size].astype(np.int64)
    arena[w] = 0xA5
    del g
    # the captured kernargs point at ARENA-based addresses; kernarg pointer itself is ours
    return compare(sym, build, mk, wgs, nthreads, LDS), dict(lds=LDS, dispatch=n, grid=[gx, gy], workgroups=wgs)


if __name__ == '__main__':
    mode, build, sym = sys.argv[1:4]
    if mode == 'case':
        lds = sys.argv[5] if len(sys.argv) > 5 else None
        res = case(build, sym, sys.argv[4], lds if lds in (None, 'hw') else int(lds))
    else:
        wgs = [tuple(int(v) for v in a.split(',')) for a in sys.argv[5:]]
        res = frame(build, sym, int(sys.argv[4]), wgs)
    print('PASS', json.dumps(res))
