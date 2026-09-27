"""Impulse-response footprint of k_pre_block (emu only, no GPU, no kernel changes).

Sets a single input pixel to 1.0 (rest 0) and diffs the result against an all-zero
baseline run: the set of output bytes that differ is the kernel's spatial filter
footprint for that impulse position.

usage: impulse_pre.py [H W] [y x] ...     default: 16 16 with impulses (8,8) (4,4) (1,1) (0,0)
"""
import sys
from pathlib import Path
import numpy as np

_myargs = sys.argv[1:]
sys.argv = sys.argv[:1]                     # keep helper modules from parsing our args
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, kernelspec as K, difftest_var as D

SYM = '_Z21k_pre_block_1h_32_fp89PreParams'
SWIN = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'
seed = 1

# CLI: bare ints come in pairs (y x). H W via env H= W=, default 16x16.
H = int(__import__('os').environ.get('H', '16'))
W = int(__import__('os').environ.get('W', '16'))
nums = [int(x) for x in _myargs if x.lstrip('-').isdigit()]
pos = [(nums[k], nums[k + 1]) for k in range(0, len(nums) - 1, 2)]
if not pos:
    pos = [(8, 8), (4, 4), (1, 1), (0, 0)]
grid = ((W + 7) // 8, (H + 7) // 8)


def kernarg():
    explicit, ksize, lds, hid = K.kernel_meta(SYM)
    ka = bytearray(ksize)
    S = lambda k: V.ARENA + k * V.SLOT
    import struct, os
    struct.pack_into('<QQQ', ka, 0x00, S(0), S(1), S(2))
    struct.pack_into('<ii', ka, 0x18, H, W)
    struct.pack_into('<fi', ka, 0x20, float(os.environ.get('PRE_F20', '0.0625')), 0)
    struct.pack_into('<ff', ka, 0x28, 0.0, 0.0)
    struct.pack_into('<i', ka, 0x30, 0)
    p40 = os.environ.get('PRE_40', 'null')
    struct.pack_into('<QQ', ka, 0x38, S(3), 0 if p40 == 'null' else S(4))
    struct.pack_into('<ff', ka, 0x48, *[float(x) for x in os.environ.get('PRE_F48', '1.0,1.0').split(',')])
    for name, val in (('block_count_x', grid[0]), ('block_count_y', grid[1]), ('block_count_z', 1),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1), ('grid_dims', 2)):
        o, sz = hid[name]
        struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return ka


def run_once(pixels):
    """pixels: list of (y, x) set to 1.0 in all three channels; returns region-1 bytes."""
    import struct
    ka = kernarg()
    p = E.load_program(R.DIS, {SYM, SWIN})
    entry = E.load_program(R.DIS, {SYM})[0][0]
    k = min(idx for idx, x in enumerate(p) if x[0] == entry)
    prog = p[k:] + p[:k]
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    a[:H * W * 12] = 0
    for (y, x) in pixels:
        off = (y * W + x) * 12
        a[off:off + 12] = np.ones(3, np.float32).view(np.uint8)
    lds = D.group_size(SYM)
    DP = 0x7100_0000_0000
    pkt = bytearray(64)
    struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, grid[0] * 256, grid[1], 1)
    struct.pack_into('<II', pkt, 24, 64, lds)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256,
                            {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy},
                            max_steps=8_000_000)
    return a.copy()


def compress(off):
    off = sorted(set(int(x) for x in off))
    out, s, p = [], off[0], off[0]
    for x in off[1:]:
        if x == p + 1:
            p = x
        else:
            out.append((s, p)); s = p = x
    out.append((s, p))
    return out


base = run_once([])
nfin = int(np.isfinite(base.view(np.float32)[:H * W * 3]).sum())
print('baseline: %d bytes, finite-f32 in input area: %d/%d' % (len(base), nfin, H * W * 3))

for (y, x) in pos:
    out = run_once([(y, x)])
    d = np.nonzero(out != base)[0]
    if not len(d):
        print('impulse (%d,%d): NO CHANGE' % (y, x)); continue
    slots = {}
    for idx in d:
        slots.setdefault(int(idx // V.SLOT), []).append(int(idx % V.SLOT))
    print('impulse (%d,%d): %d changed bytes, slots %s' % (y, x, len(d), sorted(slots)))
    for sl in sorted(slots):
        rr = compress(slots[sl])
        shown = ' '.join('%d-%d' % r if r[0] != r[1] else str(r[0]) for r in rr[:24])
        print('  slot %d: %d ranges, first bytes %d..%d: %s%s' %
              (sl, len(rr), rr[0][0], rr[-1][1], shown, ' ...' if len(rr) > 24 else ''))
    # value previews at the first few changed elements (4-byte aligned view)
    els = [int(idx) - int(idx) % 4 for idx in d[:400]]
    seen = []
    for e in els:
        if e in seen:
            continue
        seen.append(e)
        if len(seen) > 8:
            break
        b = out[e:e + 4].tobytes(); b0 = base[e:e + 4].tobytes()
        f32 = np.frombuffer(b, np.float32)[0] if len(b) == 4 else float('nan')
        f320 = np.frombuffer(b0, np.float32)[0] if len(b0) == 4 else float('nan')
        print('    off %d (+%d slot): %s -> %s  f32 %g -> %g' %
              (e, e % V.SLOT, b0.hex(), b.hex(), f320, f32))
