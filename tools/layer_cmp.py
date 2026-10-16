"""Per-layer comparison of our GPU plan against the MIT PyTorch reference (dlssnr-pytorch-flex), one frame, no history.

    <flex venv python> tools/layer_cmp.py torch <nvngx_dlssnr.dll> <W> <H> <out_dir>
        CPU: runs the reference with capture=True, saves every layer's main output (NCHW f32) as L<index>.npy + plan.txt.
    py -I tools/layer_cmp.py ours <nr_runtime.dll dir> <nr_data/WxH> <W> <H> <out_dir> <step,step,...>   (GPU, wrap in gpu2.sh)
        Runs the first n steps (nr_dbg_steps) for each listed n and saves the step's output buffer as S<n>.bin.
    py -I tools/layer_cmp.py cmp <nr_data/WxH> <torch_dir> <ours_dir> <step:layer:C,...>
        Decodes our tiles (e4m3, act(C,h,w)) and prints PSNR (peak = ref max-abs), max-abs and relative error per pair.
Input is the same as tools/torch_ref.py: E:/dlss-runtime/color_WxH.bin, RGB clamped to [0,1], import mode 0.
"""
import json, os, struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from torch_ref import frame


def torch_dump(dll, W, H, out):
    import torch, contextlib, io
    from dlssnr.extract import extract
    from dlssnr.io import Controls, _mirror
    from dlssnr.model import NRNet, _main
    with contextlib.redirect_stdout(io.StringIO()): w = extract(dll, None)
    net = NRNet(w, W, H); ci = frame(W, H)[..., :3]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows, cols = _mirror(torch.arange(net.padding.height), H), _mirror(torch.arange(net.padding.width), W)
    img = torch.from_numpy(np.ascontiguousarray(ci.transpose(2, 0, 1)))[:, rows][:, :, cols].unsqueeze(0).contiguous()
    with torch.no_grad(): o = net.run(img, Controls(seed=0), capture=True)
    lines = []
    for s in net.plan:
        v = o[s.index]; parts = v if isinstance(v, tuple) else (v,)
        for k, t in enumerate(parts):
            if isinstance(t, torch.Tensor):
                np.save(out / ('L%d_%d.npy' % (s.index, k)), t.float().numpy()); lines.append('%d %d %s %s' % (s.index, k, s.kind, tuple(t.shape)))
    (out / 'plan.txt').write_text('\n'.join(lines))


def local(dll, data, tdir, odir, pairs):
    """Per-layer error: the reference layer on OUR previous output (valid region pasted over the reference's own
    input, so the padding is the reference's) against our output, interior only (8 px margin at the bottom/right)."""
    import torch, contextlib, io
    from dlssnr.extract import extract
    from dlssnr.model import NRNet
    m = json.loads((Path(data) / 'meta.json').read_text()); W, H = m['width'], m['height']
    with contextlib.redirect_stdout(io.StringIO()): w = extract(dll, None)
    net = NRNet(w, W, H)
    for p in pairs:
        s, l, C, f = p.split(':'); s, l, C, f = int(s), int(l), int(C), int(f)
        vh, vw = -(-H // f), -(-W // f); ld = lambda n: untile(np.fromfile(Path(odir) / ('S%d.bin' % n), np.uint8), C, m['pad_h'] // f, m['pad_w'] // f)
        x = np.load(Path(tdir) / ('L%d_0.npy' % (l - 1))).copy(); x[0, :C, :vh, :vw] = ld(s - 1)[:, :vh, :vw]
        with torch.no_grad(): y = net.layers[l](torch.from_numpy(x), None)
        y = (y[0] if isinstance(y, tuple) else y)[0].numpy()
        k = 1 if (Path(tdir) / ('L%d_1.npy' % l)).exists() and np.load(Path(tdir) / ('L%d_1.npy' % l)).shape[1] == C else 0
        y = (net.layers[l](torch.from_numpy(x), None)[k][0].numpy() if k else y)
        a, r = ld(s)[:, :vh - 8, :vw - 8], y[:C, :vh - 8, :vw - 8]
        print('local step %-4d layer %-4d %.2f dB  maxabs %.4g  rel %.4f  exact %.3f' % ((s, l) + stats(a, r) + (float((a == r).mean()),)), flush=True)
        if os.environ.get('MAP'):
            e = (a != r).mean(0); print('  mismatch by row mod 8', np.round([e[i::8].mean() for i in range(8)], 3), 'col mod 8', np.round([e[:, i::8].mean() for i in range(8)], 3))
            print('  rows 0-8 %.3f  cols 0-8 %.3f  interior %.3f' % (e[:8].mean(), e[:, :8].mean(), e[8:, 8:].mean()))
            print('  by 16-row band', np.round([e[i:i + 16].mean() for i in range(0, e.shape[0], 16)], 2))


def out_field(sym, ka):
    import plan_check, build_plan
    S = build_plan.parse_header()
    nm = next((v for k, v in plan_check.NATIVE_STRUCT.items() if sym.startswith(k)), None) or next(n for n in S if sym.endswith(n))
    f = S[nm][1]
    for nm in ('out_half', 'out', 'dst', 'pooled', 'rgb'):
        if nm in f: return struct.unpack_from('<Q', ka, f[nm][0])[0]


def ours_dump(dlldir, data, W, H, out, steps):
    import ctypes as C, plan_check
    m, st, _ = plan_check.load(data); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    os.add_dll_directory(dlldir); lib = C.CDLL(os.path.join(dlldir, 'nr_runtime.dll'))
    lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.nr_dbg_steps.argtypes = [C.c_void_p, C.c_int]; lib.nr_dbg_read.argtypes = [C.c_void_p, C.c_ulonglong, C.c_ulonglong, C.c_void_p]
    err = C.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
    x = np.ascontiguousarray(frame(W, H).astype(np.float16).view(np.uint16)); y = np.empty_like(x)
    if os.environ.get('STR'):   # conditioning tone,structure,skin,other; the reference's defaults are 1,1,1,1
        lib.nr_set_strength.argtypes = [C.c_void_p] + [C.c_float] * 4; assert lib.nr_set_strength(c, *map(float, os.environ['STR'].split(','))) == 0
    for n in steps:
        sym = st[n - 1][1]; p = out_field(sym, st[n - 1][3]) - m['base']
        nm, o, sz = plan_check.owner(m, p)
        lib.nr_dbg_steps(c, n); assert lib.nr_run2(c, x.ctypes.data, y.ctypes.data, None, 0) == 0
        b = np.empty(o + sz - p, np.uint8); assert lib.nr_dbg_read(c, p, b.size, b.ctypes.data) == 0
        b.tofile(out / ('S%d.bin' % n)); print(n, sym, nm, p - o, flush=True)
    lib.nr_dbg_steps(c, 0); assert lib.nr_run2(c, x.ctypes.data, y.ctypes.data, None, 0) == 0
    np.save(out / 'final.npy', y.view(np.float16)[..., :3].astype(np.float32))


def e4m3(b):
    e = ((b >> 3) & 15).astype(np.float32); mm = (b & 7).astype(np.float32)
    v = np.where(e == 0, mm / 8 * 2.0 ** -6, (1 + mm / 8) * np.power(2.0, e - 7))
    return np.where((e == 15) & (mm == 7), np.nan, np.where(b & 128, -v, v))


def pi32(c): return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16)


def untile(b, C, h, w):
    """act(C,h,w) -> [C,h,w]. Layout (rdna2/native/nk_common.h): tile = 4x4 px x C ch, C*16 B, byte = tile*C*16 +
    (r>>5)*512 + (p&7)*64 + 4*(p>>3) + 16*((r&15)>>2) + 8*((r>>4)&1) + (r&3), storage r -> channel (r&~31)|pi32(r&31)."""
    th, tw = -(-h // 4), -(-w // 4)
    r = np.arange(C)[:, None]; p = np.arange(16)[None, :]
    off = (r >> 5) * 512 + (p & 7) * 64 + 4 * (p >> 3) + 16 * ((r & 15) >> 2) + 8 * ((r >> 4) & 1) + (r & 3)   # [r, p]
    v = e4m3(b[:C * th * tw * 16]).reshape(th * tw, C * 16)[:, off]          # [tile, r, p]
    ch = (np.arange(C) & ~31) | np.array([pi32(i & 31) for i in range(C)])
    out = np.empty((C, th, tw, 4, 4), np.float32)
    out[ch] = v.reshape(th, tw, C, 4, 4).transpose(2, 0, 1, 3, 4)
    if os.environ.get("PT"): out = out.transpose(0, 1, 2, 4, 3)
    return out.transpose(0, 1, 3, 2, 4).reshape(C, th * 4, tw * 4)[:, :h, :w]


def untok(b, C, ntok):
    """ViT token buffers: tiles of 16 consecutive tokens with the activation tile's byte layout -> [C, ntok]."""
    t = untile(b, C, 4, 4 * (ntok // 16))                       # [C, 4, 4*T]: tile k at columns 4k..4k+3, p = 4y+x
    return t.reshape(C, 4, ntok // 16, 4).transpose(0, 2, 1, 3).reshape(C, ntok)


def unplane(b, C, h, w):
    """planes layout (pre-block pool, Ffwd2 xp): [C/16][h][w][16 storage ch], storage m -> channel pi32(m)."""
    v = e4m3(b[:C * h * w]).reshape(C // 16, h, w, 16).transpose(0, 3, 1, 2).reshape(C, h, w)
    out = np.empty_like(v); out[[(m & ~31) | pi32(m & 31) for m in range(C)]] = v
    return out


def stats(a, r):
    d = a - r; pk = float(np.abs(r).max()) or 1
    mse = float(np.nanmean(d * d)); rel = float(np.sqrt(mse / max(float(np.mean(r * r)), 1e-30)))
    return 10 * np.log10(pk * pk / max(mse, 1e-30)), float(np.nanmax(np.abs(d))), rel


def cmp(data, tdir, odir, pairs):
    for p in pairs:
        if p == 'final':   # exported frame vs the reference output, PSNR peak 1 as tools/torch_ref.py
            o = np.clip(np.load(Path(odir) / 'final.npy'), 0, 1); r = np.load(Path(tdir) / ('L%d_0.npy' % max(int(n.split()[0]) for n in (Path(tdir) / 'plan.txt').read_text().splitlines())))[0]
            r = np.clip(r[:, :o.shape[0], :o.shape[1]].transpose(1, 2, 0), 0, 1)
            print('final: ours vs PyTorch %.2f dB' % (10 * np.log10(1 / float(((o - r) ** 2).mean())))); continue
        s, l, C, f, *ext = p.split(':'); r = np.load(Path(tdir) / ('L%s.npy' % l))[0]; f = int(f)   # ext: our h:w if not pad/f
        m = json.loads((Path(data) / 'meta.json').read_text()); vh, vw = -(-m['height'] // f), -(-m['width'] // f)
        dec = unplane if C.startswith('p') else untile
        h, w = map(int, ext) if ext else (m['pad_h'] // f, m['pad_w'] // f)
        a = dec(np.fromfile(Path(odir) / ('S%s.bin' % s), np.uint8), int(C.lstrip('p')), h, w)
        c = min(a.shape[0], r.shape[0]); a = a[:c, :vh, :vw]; r = r[:c, :vh, :vw]   # valid frame region only
        print('step %-4s layer %-6s %s  %.2f dB  maxabs %.4g  rel %.4f' % ((s, l, r.shape) + stats(a, r)))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['torch']: torch_dump(a[1], int(a[2]), int(a[3]), a[4])
    elif a[:1] == ['ours']: ours_dump(a[1], a[2], int(a[3]), int(a[4]), a[5], [int(t) for t in a[6].split(',')])
    elif a[:1] == ['local']: local(a[1], a[2], a[3], a[4], a[5].split(','))
    elif a[:1] == ['cmp']: cmp(a[1], a[2], a[3], a[4].split(','))
    else: sys.exit(__doc__)
