"""Analyse iq_run.py outputs: model resolution, conditioning sweep, compose (intensity), crops.

    py -I tools/iq_analyse.py <work dir> <tag> [png out dir]

Work dir holds in_<WxH>.npy, ours_<tag>_<WxH>.npy and (optional) ref_n_<WxH>.npy. Every model output is carried to
1920x1080 the way the DLL does it (frame + bilinear-upsampled edit, no gain) and compared with the 1080p frame.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq_lib import *

SET = ['1,1,-1,1', '1.5,1.5,-1,1', '2,2,-1,1', '1.5,1,-1,1', '1,1.5,-1,1', '2,1,-1,1', '1,2,-1,1', '1,1,2,1', '1,1,0,1', '0,0,-1,1']
REF_SET = {'960x540': SET, '1280x720': ['1,1,-1,1', '2,2,-1,1', '1,1,2,1'], '1920x1080': ['1,1,-1,1', '2,2,-1,1', '1,1,2,1']}
# crops in 1920x1080 coordinates: (frame, x, y, w, h)
CROPS = {'face_woman': (1, 300, 330, 220, 260), 'face_man': (1, 1050, 380, 170, 170), 'hand': (0, 940, 780, 360, 300),
         'shadow_hall': (0, 1150, 300, 560, 300), 'bar_woman': (2, 1160, 500, 400, 400), 'people': (3, 1500, 520, 400, 400)}
SIZES = ['960x540', '1280x720', '1920x1080']


def up(x, s):   # [N,h,w,3] -> 1920x1080 (bilinear-ish via PIL bicubic for the edit; same for all sizes)
    return np.stack([resize(f, 1920, 1080) for f in x]) if s != '1920x1080' else x


def compose_ours(f, nr, i):   # DLL LDR path: frame + gain*intensity*(nr - frame), max 0 (no upper clamp)
    return np.maximum(f + i * (nr - f), 0)


def compose_ref(f, nr, i):    # dlssnr-pytorch PostBlock.compose
    out = np.clip(f + min(i, 1.0) * (nr - f), 0, 1)
    if i > 1:
        w = np.array([0.2126, 0.7152, 0.0722], np.float32)
        r = ((out @ w) + 1 / 512) / ((f @ w) + 1 / 512)
        a = np.clip(np.maximum(r, 1e-6) ** i, 0.5, 2.0)
        out = np.clip(out * (a / np.maximum(r, 1e-6))[..., None], 0, 1)
    return out


def main(D, tag, png=None):
    F = np.load(os.path.join(D, 'in_1920x1080.npy'))
    ours = {s: np.load(os.path.join(D, 'ours_%s_%s.npy' % (tag, s)), mmap_mode='r') for s in SIZES if os.path.exists(os.path.join(D, 'ours_%s_%s.npy' % (tag, s)))}
    refs = {s: np.load(os.path.join(D, 'ref_n_%s.npy' % s), mmap_mode='r') for s in SIZES if os.path.exists(os.path.join(D, 'ref_n_%s.npy' % s))}
    lo = {s: np.load(os.path.join(D, 'in_%s.npy' % s)) for s in SIZES}
    full = lambda s, out: F + up(out - lo[s], s)                       # model output carried to 1080p as the DLL does
    print('== accuracy vs PyTorch at the same model size (whole frame / edit correlation), per setting')
    for s, r in refs.items():
        for k, name in enumerate(REF_SET[s]):
            if s not in ours: continue
            o = ours[s][SET.index(name)]; rr = r[k]
            print('  %-9s %-14s psnr %.2f dB  corr %.3f  |edit| ours %.4f ref %.4f' % (s, name, psnr(o, rr), np.corrcoef((o - lo[s]).ravel(), (rr - lo[s]).ravel())[0, 1],
                                                                                 np.abs(o - lo[s]).mean(), np.abs(rr - lo[s]).mean()))
    print('== model resolution (default conditioning), carried to 1080p, vs PyTorch 1080p (r) and ours 1080p (o)')
    R1080 = refs.get('1920x1080'); O1080 = ours['1920x1080'][0]
    for c, (n, x, y, w, h) in CROPS.items():
        row = []
        for s in SIZES:
            o = full(s, ours[s][0])[n, y:y + h, x:x + w]; inp = F[n, y:y + h, x:x + w]
            st = stats(inp, o, None if R1080 is None else R1080[0][n, y:y + h, x:x + w])
            row.append('%s: edit %.4f%s' % (s.split('x')[1], st['edit'], '' if R1080 is None else ' psnr %.1f corr %.2f' % (st['psnr'], st['corr'])))
        print('  %-12s %s' % (c, ' | '.join(row)))
    print('== conditioning sweep (ours 540p / 1080p, whole frames): |edit|, shadow |edit|, shadow dL, white-out share')
    for k, name in enumerate(SET):
        for s in ['960x540', '1920x1080']:
            st = stats(lo[s], ours[s][k])
            print('  %-14s %-9s edit %.4f  shadow %.4f  dLshadow %+.4f  white %+.4f' % (name, s, st['edit'], st['edit_shadow'], st['dL_shadow'], st['white']))
    print('== intensity (540p model, default conditioning, carried to 1080p): ours linear vs reference ratio compose')
    nr = np.clip(full('960x540', ours['960x540'][0]), 0, 1)
    for i in [1.0, 1.3, 1.6, 2.0]:
        for nm, fn in [('linear', compose_ours), ('ref', compose_ref)]:
            o = fn(F, nr, i); st = stats(F, np.clip(o, 0, 1))
            hue = np.abs(o / np.maximum(o.sum(-1, keepdims=True), 1e-3) - F / np.maximum(F.sum(-1, keepdims=True), 1e-3)).mean()
            print('  I=%.1f %-6s edit %.4f shadow %.4f dLshadow %+.4f white %+.4f over1 %.4f chroma-shift %.4f' % (i, nm, st['edit'], st['edit_shadow'], st['dL_shadow'], st['white'], (o.max(-1) > 1).mean(), hue))
    if png:
        os.makedirs(png, exist_ok=True)
        for c, (n, x, y, w, h) in CROPS.items():
            t = [F[n, y:y + h, x:x + w]] + [np.clip(full(s, ours[s][0])[n, y:y + h, x:x + w], 0, 1) for s in SIZES]
            if R1080 is not None: t.append(R1080[0][n, y:y + h, x:x + w])
            strip(os.path.join(png, 'res_%s.png' % c), t, scale=2 if w < 300 else 1)


if __name__ == '__main__':
    main(*sys.argv[1:])
