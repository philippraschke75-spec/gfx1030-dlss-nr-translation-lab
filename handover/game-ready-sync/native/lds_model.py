"""lds_model.py: CPU model of the gemm64/mma16 LDS traffic of the native ViT kernels (k_vit_expand_n, k_vit_contract_n,
k_vit_qkv_n; staging as in native-c512/c512.hip, swizzled copy in vit_swz.hip). Counts bank conflicts (gfx1030: 32 banks
x 4 B, wave32; a b32/b64/b128 access is served 32/16/8 lanes per pass, a pass costs max over banks of distinct dwords)
for the old and the swizzled layout, and proves every swizzled read returns the same logical element the old one did."""
import itertools

def ipi32(m): return (m & 1) | ((m >> 3) & 1) << 1 | ((m >> 1) & 1) << 2 | ((m >> 2) & 1) << 3 | (m & 16)

# ---- writers: per thread, 4 dwords e -> logical (row k, col) of A and of W (logical = old layout coordinates)
def a_writes(tid):
    tt, sa, q = tid >> 6, (tid >> 5) & 1, tid & 31
    return [(sa * 8 + 2 * (q & 3) + (e >> 1), tt * 16 + (q >> 2) + 8 * (e & 1)) for e in range(4)]
def w_std(tid, n0):
    q, ntl, kgl = tid & 31, tid >> 5 & 3, tid >> 7; nt = (n0 >> 4) + ntl; out = []
    for e in range(4):
        n = nt * 16 + (e >> 1) * 8 + (q >> 2)
        out.append((kgl * 8 + (e & 1) + 2 * (q & 3), ((n & ~31) | ipi32(n & 31)) - n0))
    return out
def w_qkv(tid):
    s, u = tid >> 6, tid & 63; rtl, ktl = s & 1, s >> 1; out = []
    for e in range(4):
        row = (u >> 2 & 1) | (u >> 3 & 1) << 1 | (u >> 4 & 1) << 2 | (e >> 1) << 3 | (u >> 5 & 1) << 4
        out.append((ktl * 8 + (e & 1) + 2 * ((u & 1) | (u >> 1 & 1) << 1), rtl * 32 + ipi32(row)))
    return out

# ---- layouts: logical (k, col) -> uint2 index inside a 16 x 64 block (must match vit_swz.hip sA / sW)
OLD = lambda k, c: k * 64 + c
def SWZ_A(k, c): return k * 64 + (c ^ (((k >> 1) & 3) << 2))           # XOR token bits 2-3 with k bits 1-2
def SWZ_W(k, c):                                                     # cols 4tx+e -> (e>>1)*32 + 2tx + (e&1), then XOR
    p = (c & 2) * 16 + (c >> 2) * 2 + (c & 1)                       # bits 2-3 with k bits 1-2 (16 B pairs stay intact)
    return k * 64 + (p ^ (((k >> 1) & 3) << 2))

# ---- read offsets (uint2 index of the 16 B read h of thread index j at row k) as the kernel computes them
def rd_old(k, j, h): return k * 64 + j * 4 + 2 * h
def rd_A_new(k, j, h): return k * 64 + ((j * 4 + 2 * h) ^ (((k >> 1) & 3) << 2))
def rd_W_new(k, j, h): return k * 64 + ((h * 32 + j * 2) ^ (((k >> 1) & 3) << 2))

def cycles(addrs, nbytes):   # addrs: byte address per lane (32 lanes)
    lanes = {4: 32, 8: 16, 16: 8}[nbytes]; tot = 0
    for p in range(0, 32, lanes):
        banks = {}
        for a in addrs[p:p + lanes]:
            for d in range(nbytes // 4): banks.setdefault((a // 4 + d) % 32, set()).add(a // 4 + d)
        tot += max(len(s) for s in banks.values())
    return tot, 32 // lanes

def write_cycles(lay, writer):   # one 64-K chunk, 8 waves x 4 ds_write_b64
    t = i = 0
    for w, e in itertools.product(range(8), range(4)):
        c, ii = cycles([8 * lay(*writer(w * 32 + l)[e]) for l in range(32)], 8); t += c; i += ii
    return t, i

def read_cycles(rd, who):        # one 64-K chunk, 8 waves x 16 kd x 2 ds_read_b128
    t = i = 0
    for w, k, h in itertools.product(range(8), range(16), range(2)):
        js = [(w * 32 + l) >> 4 if who == "A" else (w * 32 + l) & 15 for l in range(32)]
        c, ii = cycles([8 * rd(k, j, h) for j in js], 16); t += c; i += ii
    return t, i

def check_same_values(lay, rd):
    """layout is a bijection and each 16 B read returns exactly the old logical elements, in order"""
    inv = {lay(k, c): (k, c) for k in range(16) for c in range(64)}
    assert set(inv) == set(range(1024)), "layout is not a bijection"
    for k, j, h in itertools.product(range(16), range(16), range(2)):
        o = rd(k, j, h)
        assert o % 2 == 0, "read not 16 B aligned"
        assert [inv[o], inv[o + 1]] == [(k, j * 4 + 2 * h), (k, j * 4 + 2 * h + 1)], (k, j, h)

if __name__ == "__main__":
    check_same_values(OLD, rd_old); check_same_values(SWZ_A, rd_A_new); check_same_values(SWZ_W, rd_W_new)
    std = {write_cycles(SWZ_W, lambda t, n0=n0: w_std(t, n0)) for n0 in range(0, 4096, 64)}
    stdo = {write_cycles(OLD, lambda t, n0=n0: w_std(t, n0)) for n0 in range(0, 4096, 64)}
    assert len(std) == len(stdo) == 1
    print("per 64-K chunk, whole workgroup (8 waves): (cycles, conflict-free ideal)")
    print(f"  write A  (ds_write_b64)          old {write_cycles(OLD, a_writes)}  new {write_cycles(SWZ_A, a_writes)}")
    print(f"  write W  expand/contract (std)   old {stdo.pop()}  new {std.pop()}")
    print(f"  write W  qkv (wpiece_qkv3)       old {write_cycles(OLD, w_qkv)}  new {write_cycles(SWZ_W, w_qkv)}")
    print(f"  read  A  (ds_read_b128)          old {read_cycles(rd_old, 'A')}  new {read_cycles(rd_A_new, 'A')}")
    print(f"  read  W  (ds_read_b128)          old {read_cycles(rd_old, 'W')}  new {read_cycles(rd_W_new, 'W')}")
    print("value check OK: every swizzled read returns the same logical elements as before")
