# mv_check.py <dir> [W H]: checks the game's motion vectors against dump_frame output (frame_dump.h).
# dump_frames=2 (colour_f0/f1, motion_f1): warps frame 0 to frame 1 by f1's vectors with each convention and prints the
# mean |warp - colour_f1| (lowest = the game's convention; the DLL assumes "prev = cur + mv * (W,H)").
# One frame only: prints where the vector field's focus of expansion / mean flow is (sign sanity for forward walking).
import sys, os, numpy as np
d = sys.argv[1]; W, H = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2560, 1440)
ld = lambda n, c: np.fromfile(os.path.join(d, f'dump_{n}_{W}x{H}_{c}ch.f32'), np.float32).reshape(H, W, c)

def warp(img, off):   # bilinear fetch img at (x + off) in pixels
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32); x = np.clip(xs + off[..., 0], 0, W - 1.001); y = np.clip(ys + off[..., 1], 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int); tx, ty = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - tx) + img[y0, x0 + 1] * tx) * (1 - ty) + (img[y0 + 1, x0] * (1 - tx) + img[y0 + 1, x0 + 1] * tx) * ty

if os.path.exists(os.path.join(d, f'dump_colour_f1_{W}x{H}_4ch.f32')):
    c0, c1, m = ld('colour_f0', 4)[..., :3], ld('colour_f1', 4)[..., :3], ld('motion_f1', 2)
    S = np.array([W, H], np.float32); b = 64; e = lambda a: float(np.abs(a - c1)[b:-b, b:-b].mean())
    for name, off in (('zero', 0 * m), ('cur + mv*WH (DLL)', m * S), ('cur - mv*WH', -m * S), ('cur + mv (pixels)', m), ('cur + mv*WH*0.5', m * S * 0.5), ('cur + mv*WH*2', m * S * 2)):
        print(f'{name:20s} {e(warp(c0, off)):.5f}')
else:
    m = ld('motion', 2) * np.array([W, H], np.float32); mag = np.hypot(m[..., 0], m[..., 1])
    print('mean mv px', m.reshape(-1, 2).mean(0), 'p99 |mv|', np.percentile(mag, 99))
    ys, xs = np.mgrid[0:H, 0:W]; r = np.stack([xs - W / 2, ys - H / 2], -1)
    print('radial component mean (>0: prev lies outward = walking backward / zoom out, <0: forward) %.3f' % float((m * r).sum(-1).mean() / np.hypot(*r.transpose(2, 0, 1)).mean()))
