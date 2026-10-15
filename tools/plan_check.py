"""Compare a generated plan with a reference data dir (v0.3), with and without the GPU.

    py -I tools/plan_check.py struct <ref nr_data/WxH> <new nr_data/WxH>
        Dataflow equivalence: per dispatch the same effective kernel (after overrides), grid and threads, the same
        scalar bytes, weight pointers at the same record, and activation pointers related by one consistent buffer
        bijection (ref buffer -> new buffer, new size >= ref). Reference pointers into dead buffers ('spare',
        'pre-block scratch') may be null in the new plan when the kernel is native (natives never read them).
    py -I tools/plan_check.py run <nr_runtime.dll dir> <ref dir> <new dir> <W> <H> [frames] [hist_valid]
        Runs both on E:/dlss-runtime/color_WxH.bin (shifted copies for more frames; hist_valid=1 runs
        every frame after the first with the runtime's history) and reports bit-exactness / PSNR.
"""
import json, struct, sys
from pathlib import Path

DEAD = ('spare', 'pre-block scratch', 'pre-block +0x48')


def load(d):
    d = Path(d); m = json.loads((d / 'meta.json').read_text())
    ka = (d / 'kernargs.bin').read_bytes()
    ovr = {}
    if (d / 'overrides.txt').exists():
        for l in (d / 'overrides.txt').read_text().splitlines():
            if l.strip() and not l.startswith('#'): t = l.split('|'); ovr[t[0]] = t
    steps = []
    for l in (d / 'manifest.txt').read_text().splitlines():
        t = l.split('|'); o, n = int(t[2]), int(t[3])
        g = [int(t[4]), int(t[5]), int(t[7]) if len(t) > 7 else 1, int(t[6])]   # gx gy gz threads
        mod, sym, keep = t[0].split('\\')[-1], t[1], n
        if sym in ovr:
            v = ovr[sym]; mod, sym = v[1].split('\\')[-1], v[2]; keep = int(v[7], 0)
            g = [x if y == '=' else int(y) for x, y in zip(g, v[3:7])]
        steps.append((mod, sym, tuple(g), ka[o:o + keep]))
    w = {}
    for l in (d / 'weights.map').read_text().splitlines()[1:]:
        o, nm, n = l.split(); w[int(o)] = nm
    return m, steps, w


NATIVE_STRUCT = {'k_swin': 'VarParams', 'k_conv2_n': 'Conv2Params', 'k_ffwd2_n': 'Ffwd2Params', 'k_qkv_attn_n': 'AttnParams',
                 'k_vit_expand_n': 'ExpandParams', 'k_vit_contract_n': 'ConvParams1d', 'k_vit_qkv_n': 'QkvParams', 'k_dec_upsample_n': 'DecUpParams',
                 'k_pre_native': 'PreParams', 'k_post_native': 'PostParams'}


def explicit(sym):
    sys.path.insert(0, str(Path(__file__).parent)); import build_plan
    S = build_plan.parse_header()
    for k, v in NATIVE_STRUCT.items():
        if sym.startswith(k): return S[v][0]
    return next(S[n][0] for n in S if sym.endswith(n))


def owner(m, off):
    for nm, o, n in m['placements']:
        if o <= off < o + n: return nm, o, n
    return None, None, None


def struct_check(ref, new):
    (rm, rs, rw), (nm_, ns, nw) = load(ref), load(new)
    rb, nb = rm['base'], nm_['base']
    if len(rs) != len(ns): return print('FAIL: %d vs %d dispatches' % (len(rs), len(ns))) or False
    bij, bad = {}, 0
    for i, ((rmod, rsym, rg, rk), (nmod, nsym, ng, nk)) in enumerate(zip(rs, ns)):
        msg = []
        if (rmod, rsym, rg) != (nmod, nsym, ng): msg.append('kernel %s %s %s vs %s %s %s' % (rmod, rsym, rg, nmod, nsym, ng))
        exp = min(len(rk), explicit(nsym))   # the explicit struct only; the hidden block is rebuilt from the grid
        for o in range(0, min(exp, len(rk), len(nk)), 8):
            rv, nv = struct.unpack_from('<Q', rk, o)[0], struct.unpack_from('<Q', nk, o)[0]
            rp = rb <= rv < rb + rm['arena_size']; np_ = nb <= nv < nb + nm_['arena_size']
            if not rp and not np_:
                if rv != nv: msg.append('+0x%x scalar %#x vs %#x' % (o, rv, nv))
                continue
            if rp and (rv - rb) in rw:
                if not np_ or rw[rv - rb] != nw.get(nv - nb): msg.append('+0x%x weight %s vs %s' % (o, rw[rv - rb], nw.get(nv - nb) if np_ else nv))
                continue
            rn, ro, rsz = owner(rm, rv - rb)
            if rp and not np_:
                if nv == 0 and rn and rn.startswith(DEAD): continue
                msg.append('+0x%x ref -> %s, new %#x' % (o, rn, nv)); continue
            if not rp: msg.append('+0x%x ref scalar %#x, new pointer' % (o, rv)); continue
            nn, no, nsz = owner(nm_, nv - nb)
            key = (rn, rv - rb - ro); val = (nn, nv - nb - no)
            if bij.setdefault(key, val) != val or nsz < rsz: msg.append('+0x%x %s+%d -> %s+%d (size %s/%s, seen %s)' % (o, rn, key[1], nn, val[1], rsz, nsz, bij[key]))
        if msg: bad += 1; print('step %d %s: %s' % (i, nsym, '; '.join(msg)))
    inv = {}
    for k, v in bij.items():
        if inv.setdefault(v, k) != k: bad += 1; print('two ref buffers share %s: %s, %s' % (v, inv[v], k))
    for k in ('vit_lo', 'vit_hi', 'n_steps', 'pad_w', 'pad_h', 'src_size', 'dst_size'):
        if k in rm and rm[k] != nm_[k]: bad += 1; print('meta %s %s vs %s' % (k, rm[k], nm_[k]))
    print('%s: %d dispatches, %d buffers mapped, %d steps differ; arena %.1f -> %.1f MB' % ('PASS' if not bad else 'FAIL', len(rs), len(bij), bad, rm['arena_size'] / 1e6, nm_['arena_size'] / 1e6))
    return not bad


def run(dlldir, ref, new, W, H, frames=4, hist=0):
    import ctypes as C, os, numpy as np
    os.add_dll_directory(dlldir); lib = C.CDLL(os.path.join(dlldir, 'nr_runtime.dll'))
    lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]; lib.nr_destroy.argtypes = [C.c_void_p]
    base = np.fromfile('E:/dlss-runtime/color_%dx%d.bin' % (W, H), np.uint16).reshape(H, W, 4)
    outs = []
    for d in (ref, new):
        err = C.create_string_buffer(512); c = lib.nr_create(str(d).encode(), W, H, int(os.environ.get("VIT_EVERY", "1")), err, 512)   # VIT_EVERY=-2: bottleneck reuse
        if not c: sys.exit('nr_create %s: %s' % (d, err.value.decode()))
        o = []
        for k in range(frames):
            x = np.ascontiguousarray(np.roll(base, -13 * k, axis=1)); y = np.empty_like(x)
            assert lib.nr_run2(c, x.ctypes.data, y.ctypes.data, None, int(hist and k > 0)) == 0; o.append(y)
        lib.nr_destroy(c); outs.append(o)
    ok = True
    for k, (a, b) in enumerate(zip(*outs)):
        fa, fb = [np.nan_to_num(v.view(np.float16)[..., :3].astype(np.float64)) for v in (a, b)]
        mse = ((fa - fb) ** 2).mean(); same = np.array_equal(a, b); ok &= same
        print('frame %d: %s, PSNR %s dB, mean %.4f' % (k, 'bit-exact' if same else 'DIFFERS', 'inf' if mse == 0 else '%.1f' % (10 * np.log10(max(fa.max(), 1) ** 2 / mse)), fb.mean()))
    return ok


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['struct']: sys.exit(0 if struct_check(a[1], a[2]) else 1)
    if a[:1] == ['run']: sys.exit(0 if run(a[1], a[2], a[3], int(a[4]), int(a[5]), int(a[6]) if len(a) > 6 else 4, int(a[7]) if len(a) > 7 else 0) else 1)
    sys.exit(__doc__)
