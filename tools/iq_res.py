"""Model resolution at a 1440p output: every model size's edit carried to 2560x1440 (bilinear, as the DLL), vs the
network run natively at 1440p (no edit upsampling) and the PyTorch reference at 1440p.

    py -I tools/iq_res.py <work dir> <tag> [png out dir]

Needs in_<WxH>.npy and ours_<tag>_<WxH>.npy for 960x540 / 1280x720 / 1920x1080 / 2560x1440, ref_n_2560x1440.npy
(settings '1,1,-1,1;2,1,-1,1', iq_run.py ref) optional.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq_lib import *
from iq_analyse import SET, CROPS, compose_ref

SIZES = ['960x540', '1280x720', '1920x1080', '2560x1440']
K = 4 / 3   # 1080p crop coordinates -> 1440p


def main(D, tag, png=None):
    F = np.load(os.path.join(D, 'in_2560x1440.npy'), mmap_mode='r')
    rp = os.path.join(D, 'ref_n_2560x1440.npy'); R = np.load(rp, mmap_mode='r') if os.path.exists(rp) else None
    out = {}
    for s in SIZES:
        lo = np.load(os.path.join(D, 'in_%s.npy' % s)); o = np.load(os.path.join(D, 'ours_%s_%s.npy' % (tag, s)), mmap_mode='r')
        for k, name in enumerate(['1,1,-1,1', '2,1,-1,1']):
            e = np.array(o[SET.index(name)]) - lo
            out[s, k] = np.clip(np.array(F) + (np.stack([resize(f, 2560, 1440) for f in e]) if s != '2560x1440' else e), 0, 1)
    for k, name in enumerate(['1,1,-1,1', '2,1,-1,1']):
        print('== conditioning %s, output 2560x1440, crops: edit |out-in|, PSNR / edit corr vs %s' % (name, 'PyTorch 1440p' if R is not None else 'ours 1440p native'))
        for c, (n, x, y, w, h) in list(CROPS.items()) + [('frame0', (0, 0, 0, 1920, 1080)), ('frame1', (1, 0, 0, 1920, 1080))]:
            x, y, w, h = (int(v * K) for v in (x, y, w, h)); cut = lambda a: np.asarray(a[n, y:y + h, x:x + w])
            ref = cut(R[k]) if R is not None else cut(out['2560x1440', k])
            row = []
            for s in SIZES:
                st = stats(cut(F), cut(out[s, k]), ref)
                row.append('%s: %.4f %.1f/%.2f' % (s.split('x')[1], st['edit'], st['psnr'], st['corr']))
            if R is not None: row.append('ref: %.4f' % np.abs(ref - cut(F)).mean())
            print('  %-12s %s' % (c, ' | '.join(row)))
    if png:
        os.makedirs(png, exist_ok=True)
        for c, (n, x, y, w, h) in CROPS.items():
            x, y, w, h = (int(v * K) for v in (x, y, w, h)); cut = lambda a: np.asarray(a[n, y:y + h, x:x + w])
            t = [cut(F), cut(out['960x540', 0]), cut(out['1280x720', 0]), cut(out['1920x1080', 0]), cut(out['2560x1440', 0])]
            if R is not None: t.append(cut(R[0]))
            strip(os.path.join(png, 'res1440_%s.png' % c), t, ['input', '540p model (today)', '720p model', '1080p model', '1440p native', 'PyTorch 1440p'][:len(t)], scale=2 if w < 300 else 1)
            # today vs best: Strong = tone 2 + intensity 1.3 (reference ratio compose), the reference with the same settings
            st = lambda a: compose_ref(cut(F), a, 1.3)
            t = [cut(F), cut(out['960x540', 0]), st(cut(out['1280x720', 1])), st(cut(out['2560x1440', 1]))] + ([st(cut(R[1]))] if R is not None else [])
            strip(os.path.join(png, 'best_%s.png' % c), t, ['input', 'today: 540p default', '720p + Strong', '1440p native + Strong', 'PyTorch 1440p + Strong'][:len(t)], scale=2 if w < 300 else 1)


if __name__ == '__main__':
    main(*sys.argv[1:])
