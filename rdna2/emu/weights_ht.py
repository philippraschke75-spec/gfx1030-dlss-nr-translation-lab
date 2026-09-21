"""Read-only parser for the WEIGHTS_HT PE resource of nvngx_dlssnr.dll.

Container layout (derived from the installer's DLL, sha256 recorded by caller; 153/153 records consistent):
  blob:    u32 magic 0x08cda732, u32 0, then records back to back, sorted by name
  record:  u64 name_len, name, u64 D+40, u64 D+40, u64 D, [A-16 bytes] = u32 1, data[D], 20-byte trailer
  trailer: 8 zero bytes, u32 1, u32 0, u32 element_count (== D/2 in every record)
The element dtype is NOT proven (f16 fits: all finite, plausible magnitudes). Tensor shapes are NOT in the container.
Usage: python weights_ht.py <nvngx_dlssnr.dll> [out.json]   (weights are never copied out; only offsets/stats)
"""
import struct, sys, mmap, json
import numpy as np

def resource(m):
    e = struct.unpack_from('<I', m, 0x3c)[0]; nsec = struct.unpack_from('<H', m, e+6)[0]
    opt = struct.unpack_from('<H', m, e+20)[0]; o = e+24; so = o+opt
    rva0, _ = struct.unpack_from('<II', m, o+112+16)
    secs = [struct.unpack_from('<8sIIII', m, so+40*i) for i in range(nsec)]
    r2o = lambda r: next(rp+r-va for n, vs, va, rs, rp in secs if va <= r < va+max(vs, rs))
    root = r2o(rva0); found = []
    def name(nm):
        if not nm & 0x80000000: return 'id%d' % nm
        a = root+(nm & 0x7fffffff); return m[a+2:a+2+2*struct.unpack_from('<H', m, a)[0]].decode('utf-16le')
    def walk(off, path):
        nn, ni = struct.unpack_from('<HH', m, off+12)
        for i in range(nn+ni):
            nm, od = struct.unpack_from('<II', m, off+16+8*i); p = path+[name(nm)]
            if od & 0x80000000: walk(root+(od & 0x7fffffff), p)
            else:
                dr, sz, _, _ = struct.unpack_from('<IIII', m, root+od); found.append(('/'.join(p), r2o(dr), sz))
    walk(root, []); return found

def records(m):
    (_, base, size), = [r for r in resource(m) if 'WEIGHTS_HT' in r[0]]
    b = m[base:base+size]
    assert struct.unpack_from('<I', b, 0)[0] == 0x08cda732
    out = []; o = 8
    while o < size:
        L = struct.unpack_from('<Q', b, o)[0]; nm = b[o+8:o+8+L].decode(); p = o+8+L
        A, A2, D = struct.unpack_from('<QQQ', b, p)
        d0 = p+24+4; t = b[d0+D:d0+D+20]
        assert struct.unpack_from('<I', b, p+24)[0] == 1 and A-16 == 4+D+20, (nm, A, D)
        cnt = struct.unpack_from('<I', t, 16)[0]
        out.append(dict(name=nm, file_offset=base+d0, bytes=D, trailer_count=cnt)); o = d0+D+20
    assert o == size; return out

if __name__ == '__main__':
    f = open(sys.argv[1], 'rb'); m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    rs = records(m)
    for r in rs:
        h = np.frombuffer(m[r['file_offset']:r['file_offset']+r['bytes']], '<f2')
        r['f16_finite_frac'] = float(np.isfinite(h).mean())
        r['f16_absmax'] = float(np.nanmax(np.abs(h.astype(np.float32)[np.isfinite(h)]))) if np.isfinite(h).any() else None
    print(len(rs), 'records', sum(r['bytes'] for r in rs), 'data bytes; count==bytes/2 everywhere:', all(r['trailer_count']*2 == r['bytes'] for r in rs))
    print('min f16 finite frac', min(r['f16_finite_frac'] for r in rs), 'max |w|', max(r['f16_absmax'] or 0 for r in rs))
    if len(sys.argv) > 2: json.dump(rs, open(sys.argv[2], 'w'), indent=1)
