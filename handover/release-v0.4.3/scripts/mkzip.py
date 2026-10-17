import os, zipfile, fnmatch
R, Z = r'E:/dlss5-v0.4.3', r'E:/DLSS5-RDNA2-Experimental-v0.4.3.zip'
bad = lambda f: f.lower() == 'nvngx_dlssnr.dll' or f.lower() == 'arena.bin' or fnmatch.fnmatch(f, '_Z*.co')
n = 0; skipped = []
with zipfile.ZipFile(Z, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for d, _, fs in os.walk(R):
        for f in sorted(fs):
            p = os.path.join(d, f); a = os.path.relpath(p, R).replace('\\', '/')
            if bad(f): skipped.append(a); continue
            z.write(p, a); n += 1
print(n, 'files,', os.path.getsize(Z), 'bytes; skipped', skipped)
names = zipfile.ZipFile(Z).namelist()
assert not [x for x in names if bad(x.rsplit('/', 1)[-1])]
v04 = set(zipfile.ZipFile(r'E:/DLSS5-RDNA2-Experimental-v0.4.2.zip').namelist()); new = set(names)
print('only in v0.4.2 zip:', sorted(v04 - new)[:20]); print('only in v0.4.3 zip:', sorted(new - v04)[:20])
