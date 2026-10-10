# One gen_runtime_data.py output -> one release data dir (what nr_data\<W>x<H>\ holds):
#   kernels\ (manifest + override modules, relative paths), overrides.txt, kernargs.bin with the fo4_1280x720_native
#   host patches (k_import/k_export mode 0, post-block +0x40 null), meta, weights.map (by record name, FNV-checked)
#   and arena.lzms with every vendor weight zeroed.
# usage: python make_release_dir.py <gen dir> <dst dir> <dir with the .co files + overrides.txt (e.g. fo4_1280x720_native)>
import sys, os, json, shutil, mmap, collections, struct, ctypes as C, numpy as np
sys.path.insert(0, r'E:/dlss 5/rdna2/emu'); import weights_ht
DLL = r'C:/Program Files (x86)/Steam/steamapps/common/Fallout 4/nvngx_dlssnr.dll'
src, dst, kref = sys.argv[1:4]
os.makedirs(dst + '/kernels', exist_ok=True)
base = lambda p: p.replace('/', '\\').rsplit('\\', 1)[-1]

man = [l.rstrip('\n').split('|') for l in open(src + '/manifest.txt') if l.strip()]
ka = bytearray(open(src + '/kernargs.bin', 'rb').read())
for t in man:
    o = int(t[2])
    if t[1] in ('_Z8k_import12ImportParams', '_Z8k_export12ExportParams'): struct.pack_into('<i', ka, o + 0x28, 0)
    if t[1] == '_Z22k_post_block_1h_32_fp810PostParams': struct.pack_into('<Q', ka, o + 0x40, 0)
    t[0] = 'kernels\\' + base(t[0])
open(dst + '/kernargs.bin', 'wb').write(ka)
open(dst + '/manifest.txt', 'w').write(''.join('|'.join(t) + '\n' for t in man))
ovr = [l.rstrip('\n').split('|') for l in open(kref + '/overrides.txt') if l.strip()]
for t in ovr: t[1] = 'kernels\\' + base(t[1])
open(dst + '/overrides.txt', 'w').write(''.join('|'.join(t) + '\n' for t in ovr))
for k in {t[0] for t in man} | {t[1] for t in ovr}: shutil.copy2(kref + '/kernels/' + base(k), dst + '/' + k.replace('\\', '/'))
for f in ('meta.txt', 'meta.json'): shutil.copy2(src + '/' + f, dst + '/' + f)

# weights: find every WEIGHTS_HT record inside the placements (wmap.py), map by name, zero them
FH = open(DLL, 'rb'); m = mmap.mmap(FH.fileno(), 0, access=mmap.ACCESS_READ); recs = weights_ht.records(m)
byk = collections.defaultdict(list)
for r in recs: byk[bytes(m[r['file_offset']:r['file_offset'] + 64])].append(r)
A = np.fromfile(src + '/arena.bin', np.uint8); hits = []
for n, o, s in json.load(open(src + '/meta.json'))['placements']:
    seg = A[o:o + s]; p = 0
    while p + 64 <= s:
        f = next((r for r in byk.get(seg[p:p + 64].tobytes(), []) if p + r['bytes'] <= s and seg[p:p + r['bytes']].tobytes() == m[r['file_offset']:r['file_offset'] + r['bytes']]), None)
        if f: hits.append((o + p, f['name'], f['bytes'])); p += f['bytes']
        else: p += 256
hits.sort(); h = 0xcbf29ce484222325
for a, nm, b in hits:
    for x in A[a:a + b].tobytes(): h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    A[a:a + b] = 0
assert all(' ' not in nm for _, nm, _ in hits)
open(dst + '/weights.map', 'w').write('fnv %d %d\n' % (h, len(hits)) + ''.join('%d %s %d\n' % t for t in hits))
d = A.tobytes(); cab = C.WinDLL('cabinet'); hc = C.c_void_p(); assert cab.CreateCompressor(5, None, C.byref(hc))
o = C.create_string_buffer(len(d) // 4); n = C.c_size_t()
assert cab.Compress(hc, d, C.c_size_t(len(d)), o, C.c_size_t(len(o)), C.byref(n))
open(dst + '/arena.lzms', 'wb').write(o.raw[:n.value])
print(dst, len(man), 'dispatches,', len(hits), 'weight records', sum(t[2] for t in hits) / 1e6, 'MB, arena.lzms', n.value)
