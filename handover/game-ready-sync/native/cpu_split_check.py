"""cpu_split_check.py [step] : CPU proof that the SPLIT kernels (k_swin_c256s / k_swin_c128s) keep the MLP f32 order.
Emulates the kernel's MLP (expand -> mid -> Ms per head, contract chain accc over heads 0..NH-1, kp 0..15) on the
recorded g_A operands (weights from the step's record, x bytes from its input) three ways:
  ref    : one workgroup, head loop as in swin_var.hip (SPLIT = false)
  split  : heads [0,NH/2) and [NH/2,NH) computed by two "workgroups" into g_ms, the second one contracts heads 0..NH-1
  naive  : two partial accc chains (one per half) added at the end -- the order the task forbids
ref and split must match bit for bit; naive is shown to differ (so the check is sensitive to the order).
x is taken as the raw e4m3 bytes of the window in memory order (the channel permutation does not affect the order proof)."""
import sys, numpy as np
np.seterr(over="ignore", invalid="ignore")
G = "E:/dlss-vit/g_A"; BASE = 0x750000000000
step = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAN = [l.split("|") for l in open(f"{G}/manifest.txt").read().splitlines()]
KA = open(f"{G}/kernargs.bin", "rb").read()
ar = np.memmap(f"{G}/arena.bin", np.uint8, "r")
m = MAN[step]; ka = KA[int(m[2]):int(m[2]) + int(m[3])]
C = int(m[1].split("ILi")[1].split("E")[0]); NH = KB = C // 32
p64 = lambda o: int.from_bytes(ka[o:o + 8], "little") - BASE
inp, rec, flags = p64(0), p64(0x10), ka[0x28]
assert not flags & 8, "pick a step without upsample (record offsets below assume it)"
f32 = np.float32

def e4m3(b):
    b = b.astype(np.int64); s = np.where(b & 0x80, -1.0, 1.0); e = (b >> 3) & 15; mm = b & 7
    return s * np.where(e == 0, mm / 8 * 2.0 ** -6, (1 + mm / 8) * 2.0 ** (e - 7))
def wmat(off, N, K, GN):   # [N][K] from 16x32 e4m3 tiles, value * 2^-8 (as the kernel stages it)
    n = np.arange(N)[:, None]; k = np.arange(K)[None, :]; nt = n // 16; nl = n % 16
    tile = (nt // GN) * (GN * K // 32) + (k // 32) * GN + nt % GN; kl = k % 32
    byte = (nl & 7) * 64 + ((kl >> 1) & 3) * 16 + (nl >> 3) * 8 + (kl >> 3) * 2 + (kl & 1)
    return e4m3(ar[off + tile * 512 + byte]) * 2.0 ** -8
def q8(x):
    x = x.astype(f32); a = np.minimum(np.abs(x), f32(448)); b = a.view(np.uint32).astype(np.uint64)
    rn = ((b + 0x7FFFF + ((b >> 20) & 1)) & 0xFFF00000).astype(np.uint32).view(f32)
    rs = ((a + f32(24576)) - f32(24576)).astype(f32)
    return np.copysign(np.where(a < f32(0.015625), rs, rn), x).astype(f32)
f16r = lambda x: x.astype(np.float16).astype(f32)
def fma(a, b, c): return (a.astype(np.float64) * b + c).astype(f32)
def act(x):
    t = np.clip(x, f32(-4), f32(4))
    return (x * fma(t, fma(np.abs(t), f32(-0.055908203125), f32(0.447265625)), f32(0.89453125))).astype(f32)
def gemm(acc, A, B):   # acc[t][n] = dot2 chain over k pairs in order; A [T][K], B [N][K] (f16-exact values)
    for k in range(0, A.shape[1], 2):
        acc = (acc.astype(np.float64) + np.outer(A[:, k], B[:, k]) + np.outer(A[:, k + 1], B[:, k + 1])).astype(f32)
    return acc

R_MID = 4 * C * C; R_CON = R_MID + C * 128
X = e4m3(np.asarray(ar[inp:inp + 64 * C]).reshape(64, C))   # one window's worth of real activation bytes
We = wmat(0, 4 * C, C, 8); Wc = wmat(R_CON, C, C, C // 16)
Wm = [wmat(R_MID + h * 8 * 512, 32, 128, 2) for h in range(NH)]
def ms(h):   # expand + mid of head h -> Ms [64][32] (q8 f16 values)
    Hs = q8(act(gemm(np.zeros((64, 128), f32), X, We[h * 128:(h + 1) * 128]) * f32(256)))
    return q8(f16r(gemm(np.zeros((64, 32), f32), Hs, Wm[h]) * f32(256)))
def contract(acc, h, M): return gemm(acc, M, Wc[:, h * 32:(h + 1) * 32])

acc = np.zeros((64, C), f32)                                  # ref
for h in range(NH): acc = contract(acc, h, ms(h))
g_ms = {}                                                     # split: z = 0 and z = 1, then the contract workgroup
for z in (1, 0):
    for h in range(z * NH // 2, (z + 1) * NH // 2): g_ms[h] = ms(h).copy()
spl = np.zeros((64, C), f32)
for h in range(NH): spl = contract(spl, h, g_ms[h])
pa = np.zeros((64, C), f32); pb = np.zeros((64, C), f32)      # naive partial sums
for h in range(NH // 2): pa = contract(pa, h, g_ms[h])
for h in range(NH // 2, NH): pb = contract(pb, h, g_ms[h])
nv = (pa + pb).astype(f32)
same = np.array_equal(acc.view(np.uint32), spl.view(np.uint32))
ndiff = int((acc.view(np.uint32) != nv.view(np.uint32)).sum())
print(f"step {step} C{C}: split vs ref bit-identical: {same}; naive partial-add differs in {ndiff}/{acc.size} f32 values")
assert same and ndiff > 0
