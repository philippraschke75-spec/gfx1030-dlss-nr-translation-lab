# arena.bin -> arena.lzms with every nvngx_dlssnr.dll weight record zeroed + weights.map (arena_off dll_off bytes).
# Header line: "fnv <hash> <n>", FNV-1a 64 over the mapped bytes in map order; the runtime checks it after patching.
import sys, json, numpy as np, ctypes as C
dd, hits, out = sys.argv[1:4]
A = np.fromfile(dd + '/arena.bin', np.uint8); h = sorted(json.load(open(hits)))
fnv = 0xcbf29ce484222325
for a, f, b, n in h:
    for x in A[a:a+b].tobytes(): fnv = ((fnv ^ x) * 0x100000001b3) & (2**64 - 1)
    A[a:a+b] = 0
open(out + '/weights.map', 'w').write('fnv %d %d\n' % (fnv, len(h)) + ''.join('%d %d %d\n' % (a, f, b) for a, f, b, n in h))
d = A.tobytes(); cab = C.WinDLL('cabinet'); hc = C.c_void_p(); assert cab.CreateCompressor(5, None, C.byref(hc))
o = C.create_string_buffer(len(d) // 4); n = C.c_size_t()
assert cab.Compress(hc, d, C.c_size_t(len(d)), o, C.c_size_t(len(o)), C.byref(n))
open(out + '/arena.lzms', 'wb').write(o.raw[:n.value]); print('arena.lzms', n.value, 'fnv', fnv)
