"""render_modes.py <build dir with nr_runtime.dll> <color.bin 960x540 RGBA16F> <out dir> <data m0> <data m1>
One frame through m0 and m1 data (history off, input clamped to [0,1] as the DLL's ldr_input=1), PNGs tone-mapped x/(1+x)+sRGB:
m0.png, m1.png, diff.png (|m1-m0| x8), and 2560x1440 versions as the DLL composites them (bilinear frame + upsampled (out - in)).
Prints dark-pixel / dark-16px-block counts: dark = out luma < in luma - 0.1; dark block = 16x16 block with > 50% dark pixels."""
import sys, os, ctypes, numpy as np
from PIL import Image
bd, src, od, d0, d1 = sys.argv[1:6]; W, H = 960, 540
os.add_dll_directory(bd); rt = ctypes.CDLL(os.path.join(bd, 'nr_runtime.dll'))
rt.nr_create.restype = ctypes.c_void_p; rt.nr_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
rt.nr_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]; rt.nr_destroy.argtypes = [ctypes.c_void_p]
x = np.fromfile(src, np.float16).reshape(H, W, 4).astype(np.float32); x = np.nan_to_num(x, posinf=1, neginf=0); x[..., :3] = np.clip(x[..., :3], 0, 1)
inp = x.astype(np.float16)
def run(d):
    e = ctypes.create_string_buffer(512); c = rt.nr_create(d.encode(), W, H, 1, e, 512)
    if not c: sys.exit('nr_create %s: %s' % (d, e.value))
    o = np.zeros_like(inp); r = rt.nr_run(c, inp.ctypes.data, o.ctypes.data, None); rt.nr_destroy(c)
    if r: sys.exit('nr_run %d' % r)
    return np.nan_to_num(o.astype(np.float32)[..., :3], posinf=0, neginf=0)
def tm(v): v = np.maximum(v, 0); v = v / (1 + v); return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055) * 255 + .5).clip(0, 255).astype(np.uint8)
def up(v, w, h):   # bilinear, pixel centres (as the DLL's exp)
    ys = np.clip((np.arange(h) + .5) * v.shape[0] / h - .5, 0, v.shape[0] - 1); xs = np.clip((np.arange(w) + .5) * v.shape[1] / w - .5, 0, v.shape[1] - 1)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int); y1 = np.minimum(y0 + 1, v.shape[0] - 1); x1 = np.minimum(x0 + 1, v.shape[1] - 1); ty = (ys - y0)[:, None, None]; tx = (xs - x0)[None, :, None]
    return (v[y0][:, x0] * (1 - tx) + v[y0][:, x1] * tx) * (1 - ty) + (v[y1][:, x0] * (1 - tx) + v[y1][:, x1] * tx) * ty
L = np.array([.2126, .7152, .0722], np.float32); i3 = x[..., :3]; outs = {}
for name, d in (('m0', d0), ('m1', d1)):
    o = run(d); outs[name] = o; Image.fromarray(tm(o)).save(os.path.join(od, name + '.png'))
    f = np.maximum(up(i3, 2560, 1440) + up(o - i3, 2560, 1440), 0); Image.fromarray(tm(f)).save(os.path.join(od, name + '_2560x1440.png'))
    for tag, a, b in (('960x540', o, i3), ('2560x1440', f, up(i3, 2560, 1440))):
        dk = (a @ L) < (b @ L) - 0.1; h, w = dk.shape; bl = dk[:h // 16 * 16, :w // 16 * 16].reshape(h // 16, 16, w // 16, 16).mean((1, 3)) > 0.5
        print('%s %s: dark pixels %d (%.3f%%), dark 16px blocks %d, mean out-in %.4f' % (name, tag, dk.sum(), 100 * dk.mean(), bl.sum(), float((a - b).mean())))
Image.fromarray((np.abs(outs['m1'] - outs['m0']) * 8 * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(od, 'diff.png'))
print('mean |m1 - m0| %.4f, max %.4f' % (np.abs(outs['m1'] - outs['m0']).mean(), np.abs(outs['m1'] - outs['m0']).max()))
