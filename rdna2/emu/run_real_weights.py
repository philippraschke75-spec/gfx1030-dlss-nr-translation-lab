"""Host-only: run k_swin_var<32,true> in the emulator with a real WEIGHTS_HT record placed in the weight slot (kernarg +0x10).
usage: run_real_weights.py <dll> <record-name> <flags> ; compares against the synthetic-weight run. Emulator only - no GPU."""
import sys, json, mmap
import numpy as np
import gfx11emu as E, run_var as V, difftest_var as D
dll, name, flags = sys.argv[1], sys.argv[2], int(sys.argv[3], 0)
fh = open(dll, 'rb'); m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
r = {x['name']: x for x in json.load(open('emu/logs/weights_ht_index.json'))}[name]
w = np.frombuffer(m[r['file_offset']:r['file_offset'] + r['bytes']], np.uint8)
sym, lds = D.SYMS['32_1']
def go(weights):
    ka = V.make_kernarg(flags=flags, grid=(2, 2), offy=0, offx=0)
    prog = E.load_program(D.R.DIS, {sym}); g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    if weights is not None: a[2 * V.SLOT:2 * V.SLOT + len(weights)] = weights
    init = a.copy(); steps = 0
    for wy in range(2):
        for wx in range(2):
            steps += E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)['steps']
    return init, a.copy(), steps
i0, o0, s0 = go(None); i1, o1, s1 = go(w)
def lut(b):
    s = -1. if b & 128 else 1.; e = (b >> 3) & 15; mm = b & 7
    return np.nan if (e == 15 and mm == 7) else s * (mm / 8 * 2**-6 if e == 0 else (1 + mm / 8) * 2**(e - 7))
L = np.array([lut(i) for i in range(256)])
for tag, i, o, s in (('synthetic', i0, o0, s0), ('real', i1, o1, s1)):
    ch = np.nonzero(i != o)[0]; slots = np.unique(ch // V.SLOT)
    print(tag, 'steps', s, 'changed', len(ch), 'slots', {int(k): hex(V.PTR_FIELDS[k]) for k in slots})
    for k in slots:
        seg = o[k * V.SLOT:(k + 1) * V.SLOT]; msk = (i[k * V.SLOT:(k + 1) * V.SLOT] != seg)
        v = L[seg[msk]]; print('   slot', int(k), 'changed', int(msk.sum()), 'distinct', len(np.unique(seg[msk])), 'as-e4m3 nan %.3f absmean %.4g' % (np.isnan(v).mean(), np.nanmean(np.abs(v))))
print('outputs differ between synthetic and real weights:', int((o0 != o1).sum()), 'bytes')
