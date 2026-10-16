"""analyze.py: per config (sweep/cfg/*): pace / ghost / flicker numbers, 1440p PSNR vs cfg/ref (last flicker frame, display [0,1]),
PNG into <png dir>. Prints a markdown table, ranked by fps."""
import numpy as np, re, sys, os, zlib, struct
S = 'E:/dlss-runtime/sweep/cfg'; PNG = sys.argv[1] if len(sys.argv) > 1 else None


def png(path, img):
    img = (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8); h, w, _ = img.shape
    raw = b''.join(b'\0' + img[y].tobytes() for y in range(h))
    ch = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + ch(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + ch(b'IDAT', zlib.compress(raw, 6)) + ch(b'IEND', b''))


ld = lambda n, k='out': np.clip(np.fromfile('%s/%s/last_%s.bin' % (S, n, k), np.float32).reshape(1440, 2560, 3), 0, 1)
ref = ld('ref'); inp = ld('ref', 'in')
psnr = lambda a, b: 10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12))
txt = lambda n, t: open('%s/%s/%s.txt' % (S, n, t)).read() if os.path.exists('%s/%s/%s.txt' % (S, n, t)) else ''
rows = []
for n in sorted(os.listdir(S)):
    ps = [re.search(r'avg ([\d.]+) ms \(([\d.]+) fps\) sd ([\d.]+) p50 [\d.]+ p95 ([\d.]+) p99 [\d.]+ max ([\d.]+)', txt(n, t)) for t in ('pace', 'pace_1', 'pace_2')]
    ps = [m for m in ps if m]; p = None
    if ps: p = [None, '%.1f' % np.median([float(m[1]) for m in ps]), '%.1f' % (1000 / np.median([float(m[1]) for m in ps])), '%.2f' % np.median([float(m[3]) for m in ps]), '%.2f' % np.median([float(m[4]) for m in ps]), '%.1f' % max(float(m[5]) for m in ps), len(ps)]
    f = re.search(r'excess \+?([-\d.]+) \(x([\d.]+)\).*visible flicker ([\d.]+)%', txt(n, 'flick'))
    g = re.search(r'mean \|out - NR\(same frame\)\| ([\d.]+)', txt(n, 'ghost'))
    sp = [None, str(sum(int(m[1]) for m in (re.search(r'pace spikes \(> [\d.]+ ms\):.*\((\d+)\)', txt(n, t)) for t in ('pace', 'pace_1', 'pace_2')) if m))]
    q = psnr(ld(n), ref) if os.path.exists('%s/%s/last_out.bin' % (S, n)) else float('nan')
    if PNG and q == q: png('%s/%s.png' % (PNG, n), ld(n))
    rows.append((n, float(p[2]) if p else 0, p, f, g, sp, q))
if PNG: png(PNG + '/input.png', inp)
base = next(r for r in rows if r[0] == 'base'); bf = base[1]
print('| config | fps | vs base | sd | p95 | max | spikes>1.5x (3 runs) | PSNR vs ref 1440p (dB) | flicker x | visible % | ghost |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for n, fps, p, f, g, sp, q in sorted(rows, key=lambda r: -r[1]):
    print('| %s | %.1f | %+.1f%% | %s | %s | %s | %s | %s | %s | %s | %s |' % (n, fps, 100 * (fps / bf - 1), p[3] if p else '-', p[4] if p else '-', p[5] if p else '-', sp[1] if sp else '-',
          'inf' if q > 99 else '%.1f' % q, f[2] if f else '-', f[3] if f else '-', g[1] if g else '-'))
print('input vs ref: %.1f dB' % psnr(inp, ref))
