"""Hybrid-resolution quality: edit of a candidate vs the edit of the full 2560x1440 network on game frames.

    py -I tools/hybrid_eval.py prep <dump dir> <work dir> <frame>...      writes in_<f>_<WxH>.bin (RGBA16F) at 2560x1440,
                                                                           1920x1080, 1280x720 (area downscale)
    py -I tools/hybrid_eval.py score <work dir> <frame>... -- <name>=<WxH> ...
        reads out_<f>_<name>.bin (nr_bench NR_BENCH_OUT) and out_<f>_full.bin (reference). A candidate below 2560x1440 has its
        edit (out - in at its size) bilinearly upscaled, as the DLL composes it. Prints per candidate, averaged over frames:
        edit strength (mean |edit| / mean |full edit|), PSNR of the edit vs the full edit (peak 1), correlation, and the same
        strength / correlation on the detail part of the edit (edit minus its 9x9 box blur).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SIZES = [(2560, 1440), (1920, 1080), (1280, 720)]


def t(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None]


def prep(dump, work, frames):
    work.mkdir(parents=True, exist_ok=True)
    for f in frames:
        a = np.fromfile(dump / ('dump_colour_f%s_2560x1440_4ch.f32' % f), np.float32).reshape(1440, 2560, 4)
        for w, h in SIZES:
            b = a if w == 2560 else F.interpolate(t(a), size=(h, w), mode='area')[0].permute(1, 2, 0).numpy()
            b = b.copy(); b[..., 3] = 1
            b.astype(np.float16).tofile(work / ('in_%s_%dx%d.bin' % (f, w, h)))


def load(p, w, h): return np.fromfile(p, np.float16).reshape(h, w, 4)[..., :3].astype(np.float32)


def edit(work, f, name, w, h):
    e = load(work / ('out_%s_%s.bin' % (f, name)), w, h) - load(work / ('in_%s_%dx%d.bin' % (f, w, h)), w, h)
    if w != 2560: e = F.interpolate(t(e), size=(1440, 2560), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
    return e


def score(work, frames, cands):
    print('%-12s %9s %8s %8s %8s %9s %8s' % ('candidate', 'mean|e|', 'strength', 'PSNR dB', 'corr', 'detail%', 'corr_d'))
    hp = lambda e: e - F.avg_pool2d(t(e), 9, 1, 4, count_include_pad=False)[0].permute(1, 2, 0).numpy()   # detail = edit - 9x9 box blur
    for name, (w, h) in cands:
        r = []
        for f in frames:
            ref = edit(work, f, 'full', 2560, 1440); e = edit(work, f, name, w, h)
            mse = float(((e - ref) ** 2).mean())
            hd, hr = hp(e), hp(ref)
            r.append((np.abs(e).mean(), np.abs(e).mean() / np.abs(ref).mean(), 10 * np.log10(1 / max(mse, 1e-12)), np.corrcoef(e.ravel(), ref.ravel())[0, 1],
                      np.abs(hd).mean() / np.abs(hr).mean(), np.corrcoef(hd.ravel(), hr.ravel())[0, 1]))
        m = np.mean(r, axis=0)
        print('%-12s %9.5f %7.1f%% %8.2f %8.4f %8.1f%% %8.4f' % (name, m[0], 100 * m[1], m[2], m[3], 100 * m[4], m[5]))


def selftest():   # identical edits: 100 %, corr 1; half edit: 50 %
    import tempfile
    d = Path(tempfile.mkdtemp()); x = np.random.rand(1440, 2560, 4).astype(np.float16)
    x.tofile(d / 'in_0_2560x1440.bin'); (x + np.float16(0.01)).tofile(d / 'out_0_full.bin'); (x + np.float16(0.005)).tofile(d / 'out_0_half.bin')
    ref = edit(d, 0, 'full', 2560, 1440); h = edit(d, 0, 'half', 2560, 1440)
    assert abs(np.abs(h).mean() / np.abs(ref).mean() - 0.5) < 0.05
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'prep': prep(Path(a[1]), Path(a[2]), a[3:])
    elif a[0] == 'score':
        i = a.index('--'); score(Path(a[1]), a[2:i], [(c.split('=')[0], tuple(int(v) for v in c.split('=')[1].split('x'))) for c in a[i + 1:]])
    elif a[0] == 'selftest': selftest()
