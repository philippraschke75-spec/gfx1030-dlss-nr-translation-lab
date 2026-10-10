"""Helpers to analyse a trace32 capture against the window's inputs (step 3, wg (5,5))."""
import sys, struct, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid"); import vr
b = np.arange(256); e = (b >> 3) & 15; m = b & 7
LUT = (np.where(e == 0, m * 2.0 ** -9, (8 + m) * 2.0 ** (e - 10)) * np.where(b & 128, -1, 1)).astype(np.float32)
LUT[0x7F] = np.nan; LUT[0xFF] = np.nan
AR = np.memmap(f"{vr.G}/arena.bin", np.uint8, mode="r")
def pi32(c): return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16)
def window(arr, ptr, H, W, r0, c0, planes=False):   # [64 token][32 memory channel] e4m3 bytes of one 8x8 window
    WT = W >> 2; C = 32; x = np.zeros((64, 32), np.uint8)
    for t in range(64):
        row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3); col = c0 + ((t >> 4) & 1) * 4 + (t & 3)
        if not (0 <= row < H and 0 <= col < W): continue
        p = (row & 3) * 4 + (col & 3)
        if planes:
            for c in range(32): x[t, c] = arr[ptr + (c >> 4) * H * W * 16 + (row * W + col) * 16 + (c & 15)]
        else:
            base = ptr + ((row >> 2) * WT + (col >> 2)) * C * 16 + (p & 7) * 64 + 4 * (p >> 3)
            for c in range(32): x[t, c] = arr[base + 16 * ((c & 15) >> 2) + 8 * (c >> 4) + (c & 3)]
    return x
def load(step, wgx, wgy, arr=None):
    arr = AR if arr is None else arr
    ka = vr.kernarg(step); H, W, X0, Y0 = struct.unpack_from("<4i", ka, 0x18); flags = ka[0x28]
    pin = struct.unpack_from("<Q", ka, 0)[0] - vr.BASE; pw = struct.unpack_from("<Q", ka, 0x10)[0] - vr.BASE
    pout = struct.unpack_from("<Q", ka, 8)[0] - vr.BASE
    r0, c0 = Y0 + wgy * 8, X0 + wgx * 8
    return dict(H=H, W=W, r0=r0, c0=c0, flags=flags, x=window(arr, pin, H, W, r0, c0, bool(flags & 1)), rec=pw,
                out=window(arr, pout, H, W, r0, c0, bool(flags & 2)))
def tile(rec_off, N=16, K=32):   # one 512-B record tile -> [16 n][32 k] e4m3 bytes
    t = np.zeros((16, 32), np.uint8)
    for n in range(16):
        for k in range(32): t[n, k] = AR[rec_off + (n & 7) * 64 + ((k >> 1) & 3) * 16 + (n >> 3) * 8 + (k >> 3) * 2 + (k & 1)]
    return t
def mat(rec_off, N, K, GN):   # [N][K], tile = kb*GN + j (single group)
    out = np.zeros((N, K), np.uint8)
    for kb in range(K // 32):
        for j in range(N // 16): out[j * 16:(j + 1) * 16, kb * 32:(kb + 1) * 32] = tile(rec_off + (kb * GN + j) * 512)
    return out
def q8(x):   # e4m3 round-to-nearest-even of f32 (|x| <= 448), as float
    x = np.asarray(x, np.float32); a = np.minimum(np.abs(x), 448)
    bits = a.view(np.uint32); rn = ((bits + 0x7FFFF + ((bits >> 20) & 1)) & 0xFFF00000).view(np.float32)
    rs = (a.astype(np.float32) + np.float32(24576)).astype(np.float32) - np.float32(24576)
    return np.copysign(np.where(a < 0.015625, rs, rn), x).astype(np.float32)
def seqdot(A, B, C=None):   # [M][K] x [N][K] -> [M][N], sequential dot2 over k pairs, f32 rounding per pair
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64); M, K = A.shape; N = B.shape[0]
    D = np.zeros((M, N), np.float32) if C is None else np.asarray(C, np.float32).copy()
    for k in range(0, K, 2):
        D = (D.astype(np.float64) + A[:, k, None] * B[None, :, k] + A[:, k + 1, None] * B[None, :, k + 1]).astype(np.float32)
    return D
