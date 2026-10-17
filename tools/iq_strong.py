"""Strong-preset candidates at a 540p model (edit carried to 1080p, reference ratio compose for intensity > 1, as the DLL).

    py -I tools/iq_strong.py <work dir> <tag> [png out dir]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iq_lib import *
from iq_analyse import SET, CROPS, compose_ref

CANDS = [('default', '1,1,-1,1', 1.0), ('tone1.5', '1.5,1,-1,1', 1.0), ('tone2', '2,1,-1,1', 1.0), ('I1.3', '1,1,-1,1', 1.3),
         ('tone1.5+I1.3', '1.5,1,-1,1', 1.3), ('tone2+I1.3', '2,1,-1,1', 1.3), ('tone1.5+I1.6', '1.5,1,-1,1', 1.6), ('skin2', '1,1,2,1', 1.0)]


def main(D, tag, png=None, s='960x540'):
    F = np.load(os.path.join(D, 'in_1920x1080.npy')); lo = np.load(os.path.join(D, 'in_%s.npy' % s))
    o = np.load(os.path.join(D, 'ours_%s_%s.npy' % (tag, s)), mmap_mode='r')
    out = {}
    for name, cond, i in CANDS:
        nr = np.clip(F + np.stack([resize(f, 1920, 1080) for f in np.array(o[SET.index(cond)]) - lo]), 0, 1)
        out[name] = compose_ref(F, nr, i)
        st = stats(F, out[name]); fc = [stats(F[n, y:y + h, x:x + w], out[name][n, y:y + h, x:x + w])['edit'] for n, x, y, w, h in [CROPS['face_woman'], CROPS['face_man']]]
        print('%-14s edit %.4f  shadow %.4f  dLshadow %+.4f  white %+.4f  faces %.4f %.4f' % (name, st['edit'], st['edit_shadow'], st['dL_shadow'], st['white'], *fc))
    if png:
        os.makedirs(png, exist_ok=True)
        pick = ['default', 'tone1.5+I1.3', 'tone2+I1.3', 'tone1.5+I1.6']
        for c in ['face_woman', 'face_man', 'hand', 'shadow_hall', 'people']:
            n, x, y, w, h = CROPS[c]
            strip(os.path.join(png, 'strong_%s.png' % c), [F[n, y:y + h, x:x + w]] + [out[k][n, y:y + h, x:x + w] for k in pick], ['input'] + pick, scale=2 if w < 300 else 1)


if __name__ == '__main__':
    main(*sys.argv[1:])
