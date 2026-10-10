import sys, json, mmap, numpy as np, collections
sys.path.insert(0, r'E:/dlss 5/rdna2/emu'); import weights_ht
D = r'C:/Program Files (x86)/Steam/steamapps/common/Fallout 4/nvngx_dlssnr.dll'
dd = sys.argv[1]
FH = open(D,'rb'); m = mmap.mmap(FH.fileno(), 0, access=mmap.ACCESS_READ); recs = weights_ht.records(m)
A = np.fromfile(dd + '/arena.bin', np.uint8); meta = json.load(open(dd + '/meta.json'))
byk = collections.defaultdict(list)
for r in recs: byk[bytes(m[r['file_offset']:r['file_offset']+64])].append(r)
hits = []
for n, o, s in meta['placements']:
    # scan placement for record starts at 256-byte alignment
    seg = A[o:o+s]; p = 0
    while p + 64 <= s:
        k = seg[p:p+64].tobytes(); found = None
        for r in byk.get(k, []):
            b = r['bytes']
            if p + b <= s and seg[p:p+b].tobytes() == m[r['file_offset']:r['file_offset']+b]: found = r; break
        if found: hits.append((o+p, found['file_offset'], found['bytes'], n)); p += found['bytes']
        else: p += 256
print(len(recs), 'records', sum(r['bytes'] for r in recs)/1e6, 'MB; hits', len(hits), sum(h[2] for h in hits)/1e6, 'MB')
print(collections.Counter(h[3].split(':')[0].split(' ')[0] for h in hits).most_common(20))
json.dump(hits, open(dd.replace('/','_').replace(':','') + '_hits.json','w'))
