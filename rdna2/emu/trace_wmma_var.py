"""Trace WMMA operands/results in k_swin_var using the existing VarParams fixture.

Usage (inside the project Python environment):
  trace_wmma_var.py sites [KEY FLAGS]
  trace_wmma_var.py trace SITE [VISIT [KEY FLAGS]]
  trace_wmma_var.py sweep SITE [KEY FLAGS]

KEY is one of 32_1, 32_0, 64_0, 128_0, 256_0. The tracer uses run_var's
synthetic buffers and random e4m3-shaped data, unless DLSSNR_WEIGHT_BLOB is set.
It stops immediately before or after a WMMA instruction and does not alter the
kernel, translator, or GPU artifacts.
"""
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import gfx11emu as E
import run_emu as R
import run_var as V
import translate_kernels as T
import translate_final_head as F

SYMS = {
    "32_1": "_Z10k_swin_varILi32ELb1EEv9VarParams",
    "32_0": "_Z10k_swin_varILi32ELb0EEv9VarParams",
    "64_0": "_Z10k_swin_varILi64ELb0EEv9VarParams",
    "128_0": "_Z10k_swin_varILi128ELb0EEv9VarParams",
    "256_0": "_Z10k_swin_varILi256ELb0EEv9VarParams",
}
HELPER = "_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti"


def lds_size(sym):
    blob, sections, syms = F.kd.parse_elf(str(F.INPUT))
    for name, off in F.kd.find_kernel_kds(blob, sections, syms):
        if name == sym:
            return T.dec.dec(F.kd.dump_kd(blob, off))["group"]
    raise KeyError("kernel metadata not found: " + sym)


def prepare(key, flags, impulse=None, amp=1.0, H=16, W=16, seed=1,
            offset_x=0, offset_y=0, grid=(1, 1), input_file=None,
            weight_file=None):
    sym = SYMS[key]
    ka = V.make_kernarg(H=H, W=W, offx=offset_x, offy=offset_y,
                        flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym, HELPER})
    entry = E.load_program(R.DIS, {sym})[0][0]
    start = min(i for i, ins in enumerate(prog) if ins[0] == entry)
    prog = prog[start:] + prog[:start]
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    # The generic run_var fixture assigns a separate arena slot to each pointer
    # field. Input location/encoding is deliberately explicit: the host contract
    # does not say every encoder path uses FP32 RGB at +0x00.
    input_field = int(os.environ.get('TRACE_INPUT_FIELD', '0'), 0)
    input_encoding = os.environ.get('TRACE_INPUT_ENCODING', 'e4m3')
    if input_field not in V.PTR_FIELDS:
        raise ValueError('TRACE_INPUT_FIELD must be an arena pointer field')
    slot = V.PTR_FIELDS.index(input_field)
    inp = g.regions[1].arr[slot * V.SLOT:(slot + 1) * V.SLOT]
    channels = int(key.split('_')[0])
    input_file = input_file or os.environ.get('TRACE_INPUT_FILE')
    if input_file:
        payload = np.fromfile(input_file, np.uint8)
        # Checkpoints are whole arena slots (1 MiB); consume only the current
        # kernel's activation footprint, which may be smaller by channel size.
        if input_encoding == 'e4m3':
            activation_bytes = channels * ((H + 3) // 4) * ((W + 3) // 4) * 16
        elif input_encoding == 'rgb32':
            activation_bytes = H * W * 12
        else:
            raise ValueError('TRACE_INPUT_ENCODING must be e4m3 or rgb32')
        if len(payload) < activation_bytes:
            raise ValueError('TRACE_INPUT_FILE has %d bytes, needs %d' %
                             (len(payload), activation_bytes))
        inp[:activation_bytes] = payload[:activation_bytes]
    elif input_encoding == 'rgb32':
        inp[:H * W * 12] = 0
    elif input_encoding == 'e4m3':
        activation_bytes = channels * ((H + 3) // 4) * ((W + 3) // 4) * 16
        inp[:activation_bytes] = 0
    else:
        raise ValueError('TRACE_INPUT_ENCODING must be e4m3 or rgb32')
    if impulse is not None:
        y, x, channel = impulse
        if input_encoding == 'rgb32':
            off = (y * W + x) * 12 + channel * 4
            inp[off:off + 4] = np.asarray([amp], np.float32).view(np.uint8)
        else:
            # Contract activation layout: 4x4 tile-major, C channels, 16
            # positions per channel. amp is an e4m3 byte code (0x38 = +1.0).
            tile = (y // 4) * ((W + 3) // 4) + (x // 4)
            inner = (y % 4) * 4 + (x % 4)
            off = tile * channels * 16 + channel * 16 + inner
            inp[off] = int(amp) & 0xff
    weight_file = weight_file or os.environ.get('TRACE_WEIGHT_FILE')
    if weight_file:
        weight = np.fromfile(weight_file, np.uint8)
        weight_slot = V.PTR_FIELDS.index(0x10)
        if len(weight) > V.SLOT:
            raise ValueError('TRACE_WEIGHT_FILE exceeds one arena slot')
        base = weight_slot * V.SLOT
        g.regions[1].arr[base:base + len(weight)] = weight
    lds = lds_size(sym)
    # k_swin_var uses its explicit kernarg pointer in SGPRs 0:1 (unlike kernels
    # that opt into the dispatch_ptr kernarg convention).
    sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: 0, 15: 0}
    return prog, g, lds, sg, entry


def wmma_sites(prog):
    return [(i, addr, args) for i, (addr, op, args, _) in enumerate(prog)
            if op == 'v_wmma_f32_16x16x16_f16']


def run_to(key, flags, site, visit=1, after=False, impulse=None, amp=1.0,
           H=16, W=16, seed=1):
    prog, g, lds, sg, entry = prepare(key, flags, impulse, amp, H, W, seed)
    sites = [(i, addr, args) for i, addr, args in wmma_sites(prog) if addr >= entry]
    if site < 0 or site >= len(sites):
        raise IndexError('site %d outside 0..%d' % (site + 1, len(sites)))
    idx, addr, args = sites[site]
    if after:
        addr = prog[idx + 1][0]
    snaps = {}
    counts = {}
    result = E.run_workgroup(prog, g, lds, 256, sg,
                             max_steps=50_000_000, stop_at=addr,
                             snap=snaps, stop_nth=visit, counts=counts)
    regs = [tuple(map(int, m)) for m in re.findall(r'v\[(\d+):(\d+)\]', args)]
    result['wmma_counts'] = {hex(a): n for a, n in counts.items()
                             if any(op == 'v_wmma_f32_16x16x16_f16' and pc == a
                                    for pc, op, _, _ in prog)}
    return snaps, regs, result, args, len(sites)


def run_all_sites(key, flags, impulse=None, amp=0x38, H=8, W=8, seed=1,
                  input_file=None, weight_file=None, offset_x=0, offset_y=0,
                  grid=None):
    """Run the whole kernel once while snapshotting first visits before/after WMMA."""
    grid = grid or (max(1, (W - offset_x + 7) // 8),
                    max(1, (H - offset_y + 7) // 8))
    prog, g, lds, sg, entry = prepare(key, flags, impulse, amp, H, W, seed,
                                      offset_x, offset_y, grid,
                                      input_file, weight_file)
    sites = [(addr, args) for _, addr, args in wmma_sites(prog) if addr >= entry]
    site_addrs = {addr for addr, _ in sites}
    captures = {}
    visits = {}
    old_compile = E.Exec._compile

    def compile_with_capture(ex, pc):
        fn = old_compile(ex, pc)
        addr, op, args, _ = ex.prog[pc]
        if op != 'v_wmma_f32_16x16x16_f16' or addr not in site_addrs:
            return fn
        def captured(w):
            key_ = (addr, w.wid)
            visits[key_] = visits.get(key_, 0) + 1
            if visits[key_] == 1:
                before = w.V.copy()
                fn(w)
                captures[key_] = (before, w.V.copy())
            else:
                fn(w)
        return captured

    E.Exec._compile = compile_with_capture
    counts = {}
    try:
        result = E.run_workgroup(prog, g, lds, 256, sg,
                                 max_steps=50_000_000, counts=counts)
    finally:
        E.Exec._compile = old_compile
    return captures, visits, counts, result, sites


def delta_summary(a, b, lo, hi):
    changed = {}
    for wave in sorted(set(a) & set(b)):
        x, y = a[wave][0], b[wave][0]
        hits = []
        for reg in range(lo, hi + 1):
            lane = np.nonzero(x[reg] != y[reg])[0]
            if len(lane):
                hits.extend((reg, int(l)) for l in lane)
        if hits:
            changed[wave] = hits
    return changed


def matrix(snap, wave, reg_range):
    lo, hi = reg_range
    vm = snap[wave][0]
    out = np.empty((16, 16), dtype=np.float32)
    for r in range(8):
        words = vm[lo + r]
        halves = np.stack((words & 0xffff, words >> 16), axis=1).astype(np.uint16)
        vals = halves.view(np.float16).astype(np.float32)
        out[2 * r:2 * r + 2] = vals[:16].T
    return out


def list_sites(key, flags):
    prog, _, _, _, entry = prepare(key, flags)
    sites = [(i, addr, args) for i, addr, args in wmma_sites(prog) if addr >= entry]
    print('%s flags=0x%x: %d static WMMA sites' % (key, flags, len(sites)))
    for n, (_, addr, args) in enumerate(sites, 1):
        print('%2d 0x%x  %s' % (n, addr, args))


def trace_site(key, flags, site, visit=1, pos=None, amp=0x38):
    H = int(os.environ.get('TRACE_HEIGHT', '16'))
    W = int(os.environ.get('TRACE_WIDTH', '16'))
    if pos is None:
        pos = (int(os.environ.get('TRACE_POS_Y', str(H // 2))),
               int(os.environ.get('TRACE_POS_X', str(W // 2))))
    y, x = pos
    base, regs, result, args, total = run_to(key, flags, site, visit, H=H, W=W)
    if not base:
        print('stop not reached; steps=%d dynamic_wmma=%s' %
              (result.get('steps', -1), result.get('wmma_counts', {})))
        return
    feature = int(os.environ.get('TRACE_FEATURE', '0'))
    imp, _, _, _, _ = run_to(key, flags, site, visit,
                             impulse=(y, x, feature), amp=amp, H=H, W=W)
    base_after, _, _, _, _ = run_to(key, flags, site, visit, after=True, H=H, W=W)
    imp_after, _, _, _, _ = run_to(key, flags, site, visit, after=True,
                                  impulse=(y, x, feature), amp=amp, H=H, W=W)
    d, a, b, _ = regs
    print('%s flags=0x%x site=%d/%d visit=%d %s' %
          (key, flags, site + 1, total, visit, args))
    cin = (regs[3][0], regs[3][1])
    for name, pair, left, right in (
            ('D_result', d, base_after, imp_after),
            ('C_input', cin, base, imp),
            ('A_input', a, base, imp),
            ('B_input', b, base, imp)):
        diff = delta_summary(left, right, *pair)
        print(' %s impulse(%d,%d,component0): %s' %
              (name, y, x, {w: sorted(set(l for _, l in hits)) for w, hits in diff.items()}))
    print(' snapshots=%d waves' % len(base))


def analyze_all(key='256_0', flags=1, H=None, W=None, pos=None, amp=0x38,
                input_file=None, weight_file=None, offset_x=0, offset_y=0,
                grid=None, feature=None):
    H = H or int(os.environ.get('TRACE_HEIGHT', '8'))
    W = W or int(os.environ.get('TRACE_WIDTH', '8'))
    if pos is None:
        pos = (int(os.environ.get('TRACE_POS_Y', str(H // 2))),
               int(os.environ.get('TRACE_POS_X', str(W // 2))))
    feature = int(os.environ.get('TRACE_FEATURE', '0')) if feature is None else feature
    print('baseline: full run %s flags=0x%x HxW=%dx%d' % (key, flags, H, W), flush=True)
    base, bv, bc, br, sites = run_all_sites(
        key, flags, H=H, W=W, input_file=input_file, weight_file=weight_file,
        offset_x=offset_x, offset_y=offset_y, grid=grid)
    print('perturbation: position=%s feature=%d code=%s' % (pos, feature, hex(int(amp))), flush=True)
    imp, iv, ic, ir, _ = run_all_sites(key, flags,
                                       impulse=(pos[0], pos[1], feature), amp=amp,
                                       H=H, W=W, input_file=input_file,
                                       weight_file=weight_file,
                                       offset_x=offset_x, offset_y=offset_y,
                                       grid=grid)
    for n, (addr, args) in enumerate(sites, 1):
        regs = [tuple(map(int, m)) for m in re.findall(r'v\[(\d+):(\d+)\]', args)]
        d, a, b, c = regs
        sections = [('A', a, 0), ('B', b, 0), ('C', c, 0), ('D', d, 1)]
        row = []
        for name, (lo, hi), phase in sections:
            waves_changed = 0
            words_changed = 0
            lanes = set()
            for wave in range(8):
                xb = base.get((addr, wave)); xi = imp.get((addr, wave))
                if xb is None or xi is None:
                    continue
                X = xb[phase]; Y = xi[phase]
                diff = X[lo:hi + 1] != Y[lo:hi + 1]
                count = int(np.count_nonzero(diff))
                if count:
                    waves_changed += 1
                    words_changed += count
                    lanes.update(int(x) for x in np.nonzero(diff)[1])
            if words_changed:
                row.append('%s:%dw/%dwords/lanes%s' %
                           (name, waves_changed, words_changed, sorted(lanes)))
        dyn = sum(1 for (pc, wid), count in iv.items() if pc == addr and count)
        print('site %2d 0x%x waves_reached=%d %s' %
              (n, addr, dyn, ' '.join(row) if row else 'no captured operand/output delta'), flush=True)
    print('run steps baseline=%d perturb=%d' % (br['steps'], ir['steps']), flush=True)


def sweep_site(key, flags, site, H=16, W=16):
    base, regs, _, args, total = run_to(key, flags, site, H=H, W=W)
    print('%s flags=0x%x site=%d/%d %s' % (key, flags, site + 1, total, args))
    a0, a1 = regs[1]
    for y in range(H):
        for x in range(W):
            hit_union = {}
            for channel in range(3):
                imp, _, _, _, _ = run_to(key, flags, site,
                                         impulse=(y, x, channel), H=H, W=W)
                delta = delta_summary(base, imp, a0, a1)
                for wave, hits in delta.items():
                    hit_union.setdefault(wave, set()).update(lane for _, lane in hits)
            if hit_union:
                print('(%d,%d) %s' % (y, x,
                      {w: sorted(v) for w, v in sorted(hit_union.items())}))


def prepare_stage4_input(out_path=None):
    """Recreate block 15's real stage-4 activation from saved real-data output.

    The checked-in handover harness stores the real C=128 output of block 13.
    Run block 14 with its verified weight record and last-in-stage flags to
    materialize the fused pooled C=256, H=W=8 activation in +0x38.
    """
    import time
    checkpoint = ROOT / 'build' / 'import-real-20260922' / 'rdna2' / 'build' / 'block13_checkpoint.npy'
    weight_path = ROOT / 'build' / 'weights' / 'block14.bin'
    if not checkpoint.is_file() or not weight_path.is_file():
        raise FileNotFoundError('requires saved block13_checkpoint.npy and weights/block14.bin')
    H = W = 16
    flags = 4
    grid = (3, 3)  # host grid for origin (-4,-4): ceil((extent+4)/8)
    ka = V.make_kernarg(H=H, W=W, offy=-4, offx=-4, flags=flags, grid=grid)
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    arena = g.regions[1].arr
    activation = np.load(checkpoint)
    weights = np.fromfile(weight_path, np.uint8)
    arena[:len(activation)] = activation
    wslot = V.PTR_FIELDS.index(0x10) * V.SLOT
    arena[wslot:wslot + len(weights)] = weights
    sym = SYMS['128_0']
    prog = E.load_program(R.DIS, {sym})
    lds = lds_size(sym)
    t0 = time.time()
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256,
                            {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                            max_steps=8_000_000)
        print('block14 row %d/%d, %.1fs' % (wy + 1, grid[1], time.time() - t0), flush=True)
    pool_slot = V.PTR_FIELDS.index(0x38)
    pooled = arena[pool_slot * V.SLOT:(pool_slot + 1) * V.SLOT].copy()
    out_path = Path(out_path) if out_path else ROOT / 'build' / 'emu_var' / 'swin_var_stage4_input.bin'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pooled.tobytes())
    print('wrote %s (%d bytes), source=%s weights=%s' %
          (out_path, len(pooled), checkpoint, weight_path), flush=True)


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    mode = argv.pop(0)
    if mode == 'prepare-stage4-input':
        return prepare_stage4_input(argv[0] if argv else None)
    if mode == 'sites':
        key = argv[0] if argv else '256_0'
        flags = int(argv[1], 0) if len(argv) > 1 else 0
        return list_sites(key, flags)
    if mode == 'analyze':
        key = argv[0] if argv else '256_0'
        flags = int(argv[1], 0) if len(argv) > 1 else 1
        return analyze_all(key, flags)
    if mode in ('trace', 'sweep'):
        site = int(argv.pop(0)) - 1
        visit = int(argv.pop(0)) if mode == 'trace' and argv else 1
        key = argv.pop(0) if argv and argv[0] in SYMS else '256_0'
        flags = int(argv.pop(0), 0) if argv else 0
        if mode == 'trace':
            return trace_site(key, flags, site, visit)
        return sweep_site(key, flags, site)
    raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
