import json, mmap, sys, numpy as np
p=sys.argv[1]; name=sys.argv[2]; f=open(p,'rb')
m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
idx={r['name']:r for r in json.load(open('emu/logs/weights_ht_index.json'))}
r=idx[name]; d=np.frombuffer(m[r['file_offset']:r['file_offset']+r['bytes']],np.uint8)
def e4(b):
    s=-1. if b&128 else 1.; e=(b>>3)&15; mm=b&7
    return np.nan if (e==15 and mm==7) else s*(mm/8*2**-6 if e==0 else (1+mm/8)*2**(e-7))
L=np.array([e4(i) for i in range(256)])
segs=[('A 4B',0,0x1000),('B 8B',0x1000,0x2000),('C f16 sparse',0x2000,0x2460),('D 8B',0x2460,0x3060),('E f16',0x3060,0x5060),('F',0x5060,0x5070),('G 8B',0x5070,0x5470),('H f16 tail',0x5470,0x54c0)]
for n,a,b in segs:
    s=d[a:b]; h=s[:len(s)//2*2].view('<f2').astype(np.float32); f_=np.isfinite(h); v=L[s]
    print('%-13s %#06x-%#06x  f16: finite %.3f med %.4g max %.4g | e4m3: nan %.3f med %.4g max %.4g | zeros %.3f'%(n,a,b,f_.mean(),np.median(np.abs(h[f_])),np.abs(h[f_]).max(),np.isnan(v).mean(),np.nanmedian(np.abs(v)),np.nanmax(np.abs(v)),(s==0).mean()))
