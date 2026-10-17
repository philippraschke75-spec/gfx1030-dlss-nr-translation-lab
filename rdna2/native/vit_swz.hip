// Native ViT GEMM kernels (k_vit_expand_n / k_vit_contract_n / k_vit_qkv_n). Workgroup: 64 tokens x 64 outputs,
// NT = 128 threads (tools/build_plan.py launches them with 128), each thread TM x TO = 8 tokens x 4 outputs.
// Every output is still one thread's dot2 chain over k in the old order (per kd: dot2(k0k1) then dot2(k2k3)), so the
// result is byte-identical to the 256-thread 4x4 kernel of perf-tile / native-c512 (checked by output hashes).
// perf-vit2 (RX 6900 XT, ViT part of the new small model):
//  * 8x4 register tile: 6 instead of 4 ds_read_b128 per kd for twice the dot2 (LDS bytes per MAC -25 %);
//  * LDS operands prefetched PF = 2 kd ahead (the 4x4 kernel waited on every group of reads: latency-bound);
//  * contract stages 128 k per barrier (KC = 2, 64 KB LDS: it runs one workgroup per CU anyway); expand/qkv keep
//    64 k (32 KB, two workgroups per CU);
//  * qkv: one 64-output block per workgroup over the plan's full 48-block grid (was 24 x 128 outputs).
// LDS: per kd row 64 uint2 (4 f16 of k each); a thread's T columns sit as T/2 b128 at j*(128/T) + 2*lane, XOR
// ((k>>1)&3)<<2 against bank conflicts on staging (reads conflict-free, staging 1-2 way; model: perf_vit2 lds2.py).
// e4m3 -> f16 conversion while staging measured as free (f16 weight shadow: no gain), so weights stay e4m3.
#include "nk_common.h"

constexpr int TM = 8, TO = 4, DX = 64 / TO, DY = 64 / TM, NT = DX * DY, PF = 2;
constexpr int RP = 256 / NT;   // 16-byte staging pieces per thread per operand per 64 k (layout of the old 256 threads)

struct WPiece { const uint8_t* src; size_t step; int wn[4], wk[4]; };

// weight piece of (virtual) staging thread vt of the old 256-thread kernel: 16 B at src (+ step per 64 k), its 4 dwords
// go to LDS row wk[e], logical output column wn[e]
template <int K, int INNER>
__device__ __forceinline__ WPiece wpiece_std(const uint8_t* Wr, int n0, int vt) {
  const int q = vt & 31, ntl = vt >> 5 & 3, kgl = vt >> 7, nt = (n0 >> 4) + ntl;
  WPiece w;
  w.src = Wr + (size_t)(((nt / INNER) * (K / 32) + kgl) * INNER + nt % INNER) * 512 + q * 16;
  w.step = (size_t)2 * INNER * 512;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int n = nt * 16 + (e >> 1) * 8 + (q >> 2);
    w.wn[e] = ((n & ~31) | ipi32(n & 31)) - n0;
    w.wk[e] = kgl * 8 + (e & 1) + 2 * (q & 3);
  }
  return w;
}

__device__ __forceinline__ WPiece wpiece_qkv3(const uint8_t* rec, int which, int n0, int vt) {
  const int s = vt >> 6, rtl = s & 1, ktl = s >> 1, u = vt & 63;
  const int tile = ((n0 >> 5) + rtl) | (ktl << 5);
  WPiece w;
  w.src = rec + 128 + (size_t)(3 * tile + which) * 1024 + u * 16;
  w.step = (size_t)3 * 64 * 1024;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int row = (u >> 2 & 1) | (u >> 3 & 1) << 1 | (u >> 4 & 1) << 2 | (e >> 1) << 3 | (u >> 5 & 1) << 4;
    w.wn[e] = rtl * 32 + ipi32(row);
    w.wk[e] = ktl * 8 + (e & 1) + 2 * ((u & 1) | (u >> 1 & 1) << 1);
  }
  return w;
}

__device__ __forceinline__ int swz(int k) { return ((k >> 1) & 3) << 2; }
template <int T> __device__ __forceinline__ int pcol(int c) {   // LDS position of logical column c (0..63)
  return ((c / (256 / T)) * 2 + ((c >> 1) & 1)) * (128 / T) + ((c >> 2) % (64 / T)) * 2 + (c & 1);
}
template <int T> __device__ __forceinline__ int tcol(int i, int l) { return (i >> 2) * (256 / T) + l * 4 + (i & 3); }   // lane l's i-th column

__device__ __forceinline__ void step(const uint4 (&a)[TM / 2], const uint4 (&b)[TO / 2], float (&acc)[TM][TO]) {
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int e = 0; e < TO; e++) {
      const uint4 u = a[i >> 1], v = b[e >> 1];
      const uint32_t a0 = i & 1 ? u.z : u.x, a1 = i & 1 ? u.w : u.y, b0 = e & 1 ? v.z : v.x, b1 = e & 1 ? v.w : v.y;
      acc[i][e] = dot2(a1, b1, dot2(a0, b0, acc[i][e]));
    }
}

template <int KD>   // kd rows of one stage
__device__ __forceinline__ void mma(const uint2* As, const uint2* Ws, float (&acc)[TM][TO]) {
  const int ty = threadIdx.x / DX, tx = threadIdx.x % DX;
  uint4 a[PF + 1][TM / 2], b[PF + 1][TO / 2];
  auto ld = [&](int kd, uint4 (&aa)[TM / 2], uint4 (&bb)[TO / 2]) {
#pragma unroll
    for (int j = 0; j < TM / 2; j++) aa[j] = *(const uint4*)(As + kd * 64 + ((j * (128 / TM) + ty * 2) ^ swz(kd)));
#pragma unroll
    for (int j = 0; j < TO / 2; j++) bb[j] = *(const uint4*)(Ws + kd * 64 + ((j * (128 / TO) + tx * 2) ^ swz(kd)));
  };
#pragma unroll
  for (int p = 0; p < PF; p++) ld(p, a[p], b[p]);
#pragma unroll
  for (int kd = 0; kd < KD; kd++) {
    if (kd + PF < KD) ld(kd + PF, a[(kd + PF) % (PF + 1)], b[(kd + PF) % (PF + 1)]);
    step(a[kd % (PF + 1)], b[kd % (PF + 1)], acc);
  }
}

template <int KC> constexpr int BUFW = 2 * 16 * KC * 64;   // uint2 per LDS buffer: A rows (16 KC x 64), then W rows

// X: activation tiles (tile*K*16 B), tiles tile0..tile0+3 (>= T: zero); wpf(vt) -> WPiece; KC: 64-k chunks per stage
template <int K, int KC, typename WP>
__device__ __forceinline__ void gemm(const uint8_t* __restrict__ X, int tile0, int T, WP wpf, float (&acc)[TM][TO], uint2* lds) {
  constexpr int NC = K / 64 / KC, BW = BUFW<KC>;
  const uint8_t* asrc[RP]; int ao[RP][4], wo[RP][4]; WPiece wp[RP];
#pragma unroll
  for (int r = 0; r < RP; r++) {
    const int vt = threadIdx.x + NT * r, tt = vt >> 6, sa = (vt >> 5) & 1, q = vt & 31, t = tile0 + tt;
    asrc[r] = t < T ? X + (size_t)t * (K * 16) + sa * 512 + q * 16 : nullptr;
    wp[r] = wpf(vt);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int k = sa * 8 + 2 * (q & 3) + (e >> 1);
      ao[r][e] = k * 64 + (pcol<TM>(tt * 16 + (q >> 2) + 8 * (e & 1)) ^ swz(k));
      wo[r][e] = BW / 2 + wp[r].wk[e] * 64 + (pcol<TO>(wp[r].wn[e]) ^ swz(wp[r].wk[e]));
    }
  }
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int e = 0; e < TO; e++) acc[i][e] = 0.f;
  uint4 ra[KC][RP], rw[KC][RP];
  auto load = [&](int c) {
#pragma unroll
    for (int s = 0; s < KC; s++)
#pragma unroll
      for (int r = 0; r < RP; r++) {
        ra[s][r] = asrc[r] ? *(const uint4*)(asrc[r] + (size_t)(c * KC + s) * 1024) : make_uint4(0, 0, 0, 0);
        rw[s][r] = *(const uint4*)(wp[r].src + (size_t)(c * KC + s) * wp[r].step);
      }
  };
  load(0);
  for (int c = 0; c < NC; c++) {   // double-buffered: one barrier per stage
    uint2* B = lds + (c & 1) * BW;
#pragma unroll
    for (int s = 0; s < KC; s++)
#pragma unroll
      for (int r = 0; r < RP; r++) {
        const uint32_t av[4] = {ra[s][r].x, ra[s][r].y, ra[s][r].z, ra[s][r].w}, wv[4] = {rw[s][r].x, rw[s][r].y, rw[s][r].z, rw[s][r].w};
#pragma unroll
        for (int e = 0; e < 4; e++) { B[ao[r][e] + s * 1024] = cvt4(av[e]); B[wo[r][e] + s * 1024] = cvt4(wv[e]); }
      }
    __syncthreads();
    if (c + 1 < NC) load(c + 1);
    mma<16 * KC>(B, B + BW / 2, acc);
  }
}

// epilogue walk: f(m = token in the 64-token block, r = first of 4 consecutive outputs, a = their 4 accumulators)
template <class F> __device__ __forceinline__ void epi(int n0, const float (&acc)[TM][TO], F f) {
  const int ty = threadIdx.x / DX, tx = threadIdx.x % DX;
#pragma unroll
  for (int go = 0; go < TO / 4; go++)
#pragma unroll
    for (int i = 0; i < TM; i++) f(tcol<TM>(i, ty), n0 + tcol<TO>(4 * go, tx), &acc[i][4 * go]);
}

// ---- kernels: epilogues verbatim from c512.hip (token m -> tile tile0 + m/16, output row r..r+3)

__device__ __forceinline__ float vit_act(float x) {
  const float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}

extern "C" __global__ void __launch_bounds__(NT) k_vit_expand_n(ExpandParams P) {
  __shared__ uint2 lds[2 * BUFW<1>];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4;
  if (n0 >= 4096) return;
  float acc[TM][TO];
  gemm<1024, 1>(P.x, tile0, T, [&](int vt) { return wpiece_std<1024, 256>(P.w, n0, vt); }, acc, lds);
  epi(n0, acc, [&](int m, int r, const float* a) {
    const int tile = tile0 + (m >> 4); uint32_t o = 0;
    for (int e = 0; e < 4; e++) o |= to_e4m3(store8(vit_act(a[e] * 65536.f))) << (8 * e);
    *(uint32_t*)(P.out + (size_t)tile * 65536 + ch_off(r) + tok_off(m & 15)) = o;
  });
}

template <int K>
__device__ __forceinline__ void vit_contract(const ContractParams& P, uint2* lds) {
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4;
  float acc[TM][TO];
  gemm<K, 2>(P.x, tile0, T, [&](int vt) { return wpiece_std<K, 64>(P.w, n0, vt); }, acc, lds);
  const _Float16* gain = (const _Float16*)(P.w + 1024 * K);
  epi(n0, acc, [&](int m, int r, const float* a) {
    const int tile = tile0 + (m >> 4);
    const size_t off = (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15);
    const uint32_t sd = *(const uint32_t*)(P.skip + off); uint32_t o = 0;
    for (int e = 0; e < 4; e++) {
      const float g = (float)gain[(r & ~31) | pi32((r + e) & 31)];
      o |= to_e4m3(store8(__fadd_rn(a[e] * 65536.f, __fmul_rn(e4f(sd >> (8 * e)), g)))) << (8 * e);
    }
    *(uint32_t*)(P.out + off) = o;
  });
}
extern "C" __global__ void __launch_bounds__(NT) k_vit_contract_n(ContractParams P) {
  __shared__ uint2 lds[2 * BUFW<2>];
  if (blockIdx.y >= 16) return;
  if (P.K == 4096) vit_contract<4096>(P, lds);
  else if (P.K == 1024) vit_contract<1024>(P, lds);
}

__device__ __forceinline__ f16 xorh(f16 v, int m) { return (f16)__shfl_xor((float)v, m, 32); }

// q/k/v: 16 blocks of 64 outputs each (grid y = 48). The q/k head norm sums x^2 over a head's 32 outputs = 8 lanes
// (tx bits 0-2, same lane order as before: identical f16 sums).
extern "C" __global__ void __launch_bounds__(NT) k_vit_qkv_n(QkvParams P) {
  __shared__ uint2 lds[2 * BUFW<1>];
  const int tile0 = blockIdx.x * 4, T = gridDim.x * 4;
  if (blockIdx.y >= 48) return;
  const int which = blockIdx.y / 16, n0 = (blockIdx.y % 16) * 64;
  float acc[TM][TO];
  gemm<1024, 1>(P.x, tile0, T, [&](int vt) { return wpiece_qkv3(P.w, which, n0, vt); }, acc, lds);
  epi(n0, acc, [&](int m, int r, const float* a) {
    const int tile = tile0 + (m >> 4);
    const f16 temp = (f16)((const float*)P.w)[r >> 5];
    float y[4];
    if (which == 2) {
      for (int e = 0; e < 4; e++) y[e] = store8(a[e] * 65536.f);
    } else {
      f16 x[4], s1[4];
      for (int e = 0; e < 4; e++) { x[e] = (f16)(a[e] * 65536.f); s1[e] = x[e] * x[e]; }
      for (int e = 0; e < 4; e++) s1[e] = s1[e] + xorh(s1[e], 4);
      f16 ta = s1[0] + s1[2], tb = s1[1] + s1[3];
      ta = ta + xorh(ta, 2); tb = tb + xorh(tb, 2);
      ta = ta + xorh(ta, 1); tb = tb + xorh(tb, 1);
      f16 tot = ta + tb;
      if (tot < (f16)0.000062f) tot = (f16)0.000062f;
      const f16 inv = (f16)(1.f / sqrtf((float)tot));
      for (int e = 0; e < 4; e++) {
        const f16 v = which == 0 ? (f16)((f16)(x[e] * (f16)5.65625f) * inv) * temp : x[e] * inv;
        y[e] = q8((float)v);
      }
    }
    uint32_t o = 0;
    for (int e = 0; e < 4; e++) o |= to_e4m3(y[e]) << (8 * e);
    *(uint32_t*)(P.out[which] + (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15)) = o;
  });
}
