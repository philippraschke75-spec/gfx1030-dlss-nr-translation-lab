// Native ViT GEMM kernels (k_vit_expand_n / k_vit_contract_n / k_vit_qkv_n) of native-c512/c512.hip with a
// bank-conflict-free LDS layout. Only where the staged halves sit in LDS changes; every thread reads the same values and
// runs the same dot2 sequence, so the output is byte-identical (proof + conflict counts: lds_model.py).
//  A[k][tok]: tok ^ (((k>>1)&3)<<2)                          write 4-way -> 1-way, read unchanged (broadcast)
//  W[k][n]  : 4tx+e -> (e>>1)*32 + 2tx + (e&1), ^ (((k>>1)&3)<<2)   write 4-way -> 1-way, read b128 2-way -> 1-way
// Register tile (perf-tile): a workgroup runs TI x TN (per kernel: VIT_TN_*) of the old 64x64 blocks (A halves: 4 tiles each, W halves:
// 64 outputs each), each staged exactly as before into its own LDS region; a thread keeps 4*TI x 4*TN accumulators and
// per kd reads 2*(TI+TN) b128 instead of 4 per 16 outputs. Per-output dot2 sequence unchanged: byte-identical output.
// The plan's grids are unchanged; workgroups past the end return at once.
#include "nk_common.h"
#ifndef VIT_TN_EXPAND
#define VIT_TN_EXPAND 1
#endif
#ifndef VIT_TN_CONTRACT
#define VIT_TN_CONTRACT 1
#endif
#ifndef VIT_TN_QKV
#define VIT_TN_QKV 2
#endif
constexpr int TI = 1;   // TI = 2 (128 tokens per workgroup) halves the already small ViT grid (3-4 x 64 tokens)

struct WPiece { const uint8_t* src; size_t step; int wn[4], wk[4]; };

template <int K, int INNER>
__device__ __forceinline__ WPiece wpiece_std(const uint8_t* Wr, int n0) {
  const int tid = threadIdx.x, q = tid & 31, ntl = tid >> 5 & 3, kgl = tid >> 7, nt = (n0 >> 4) + ntl;
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

__device__ __forceinline__ WPiece wpiece_qkv3(const uint8_t* rec, int which, int n0) {
  const int tid = threadIdx.x, s = tid >> 6, rtl = s & 1, ktl = s >> 1, u = tid & 63;
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

__device__ __forceinline__ int lin_tile(int tile0, int T) { const int t = tile0 + (threadIdx.x >> 6); return t < T ? t : -1; }
__device__ __forceinline__ int swz(int k) { return ((k >> 1) & 3) << 2; }
__device__ __forceinline__ int sA(int k, int c) { return k * 64 + (c ^ swz(k)); }
__device__ __forceinline__ int sW(int k, int c) { return k * 64 + (((c & 2) * 16 + (c >> 2) * 2 + (c & 1)) ^ swz(k)); }

// LDS per buffer: TI A regions then TN W regions of 1024 uint2 (8 KB) each
template <int TN> constexpr int BUF = (TI + TN) * 1024;

template <int TN>
__device__ __forceinline__ void mma16s(const uint2* As, const uint2* Ws, float (&acc)[4 * TI][4 * TN]) {
  const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
#pragma unroll 4
  for (int kd = 0; kd < 16; kd++) {
    const int ao = kd * 64 + ((ty * 4) ^ swz(kd)), wo = kd * 64 + ((tx * 2) ^ swz(kd));
    uint32_t A[4 * TI][2], B[4 * TN][2];
#pragma unroll
    for (int a = 0; a < TI; a++) {
      const uint4 a01 = *(const uint4*)(As + a * 1024 + ao), a23 = *(const uint4*)(As + a * 1024 + ao + 2);
      A[4*a][0] = a01.x; A[4*a][1] = a01.y; A[4*a+1][0] = a01.z; A[4*a+1][1] = a01.w;
      A[4*a+2][0] = a23.x; A[4*a+2][1] = a23.y; A[4*a+3][0] = a23.z; A[4*a+3][1] = a23.w;
    }
#pragma unroll
    for (int h = 0; h < TN; h++) {
      const uint4 w01 = *(const uint4*)(Ws + h * 1024 + wo), w23 = *(const uint4*)(Ws + h * 1024 + wo + 32);
      B[4*h][0] = w01.x; B[4*h][1] = w01.y; B[4*h+1][0] = w01.z; B[4*h+1][1] = w01.w;
      B[4*h+2][0] = w23.x; B[4*h+2][1] = w23.y; B[4*h+3][0] = w23.z; B[4*h+3][1] = w23.w;
    }
#pragma unroll
    for (int i = 0; i < 4 * TI; i++)
#pragma unroll
      for (int e = 0; e < 4 * TN; e++) acc[i][e] = dot2(A[i][1], B[e][1], dot2(A[i][0], B[e][0], acc[i][e]));
  }
}

// tile0: first of the 4*TI tiles; wp[h]: weight piece of outputs n0 + 64h
template <int K, int TN>
__device__ __forceinline__ void gemm64(const uint8_t* __restrict__ X, int tile0, int T, const WPiece (&wp)[TN],
                                       float (&acc)[4 * TI][4 * TN], uint2* lds) {
  constexpr int NC = K / 64;
  const int tid = threadIdx.x, tt = tid >> 6, sa = (tid >> 5) & 1, q = tid & 31;
  int atile[TI];
#pragma unroll
  for (int a = 0; a < TI; a++) atile[a] = lin_tile(tile0 + 4 * a, T);
  auto loadA = [&](int a, int c) -> uint4 {
    if (atile[a] < 0) return make_uint4(0, 0, 0, 0);
    return *(const uint4*)(X + (size_t)atile[a] * (K * 16) + sa * 512 + q * 16 + (size_t)c * 1024);
  };
  int ao[4], wo[TN][4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    ao[e] = sA(sa * 8 + 2 * (q & 3) + (e >> 1), tt * 16 + (q >> 2) + 8 * (e & 1));
#pragma unroll
    for (int h = 0; h < TN; h++) wo[h][e] = sW(wp[h].wk[e], wp[h].wn[e]);
  }
#pragma unroll
  for (int i = 0; i < 4 * TI; i++)
#pragma unroll
    for (int e = 0; e < 4 * TN; e++) acc[i][e] = 0.f;
  uint4 ra[TI], rw[TN];
#pragma unroll
  for (int a = 0; a < TI; a++) ra[a] = loadA(a, 0);
#pragma unroll
  for (int h = 0; h < TN; h++) rw[h] = *(const uint4*)wp[h].src;
  for (int c = 0; c < NC; c++) {
    uint2* As = lds + (c & 1) * BUF<TN>;
    uint2* Ws = As + TI * 1024;
#pragma unroll
    for (int a = 0; a < TI; a++) {
      const uint32_t av[4] = {ra[a].x, ra[a].y, ra[a].z, ra[a].w};
#pragma unroll
      for (int e = 0; e < 4; e++) As[a * 1024 + ao[e]] = cvt4(av[e]);
    }
#pragma unroll
    for (int h = 0; h < TN; h++) {
      const uint32_t wv[4] = {rw[h].x, rw[h].y, rw[h].z, rw[h].w};
#pragma unroll
      for (int e = 0; e < 4; e++) Ws[h * 1024 + wo[h][e]] = cvt4(wv[e]);
    }
    __syncthreads();
    if (c + 1 < NC) {
#pragma unroll
      for (int a = 0; a < TI; a++) ra[a] = loadA(a, c + 1);
#pragma unroll
      for (int h = 0; h < TN; h++) rw[h] = *(const uint4*)(wp[h].src + (size_t)(c + 1) * wp[h].step);
    }
    mma16s<TN>(As, Ws, acc);
  }
}

// ---- kernels: epilogues verbatim from c512.hip, looped over the TI x TN sub-blocks (token m of A half a = tile
// tile0 + 4a + m/16; output row of W half h = n0 + 64h + 4tx + e)

__device__ __forceinline__ float vit_act(float x) {
  const float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}

extern "C" __global__ void __launch_bounds__(256) k_vit_expand_n(ExpandParams P) {
  constexpr int TN = VIT_TN_EXPAND;
  __shared__ uint2 lds[2 * BUF<TN>];
  const int tile0 = blockIdx.x * 4 * TI, n0 = blockIdx.y * 64 * TN, T = gridDim.x * 4;
  if (n0 >= 4096 || tile0 >= T) return;
  float acc[4 * TI][4 * TN];
  WPiece wp[TN];
#pragma unroll
  for (int h = 0; h < TN; h++) wp[h] = wpiece_std<1024, 256>(P.w, n0 + 64 * h);
  gemm64<1024, TN>(P.x, tile0, T, wp, acc, lds);
  const int ty = threadIdx.x >> 4;
#pragma unroll
  for (int h = 0; h < TN; h++) {
    const int r = n0 + 64 * h + (threadIdx.x & 15) * 4;
#pragma unroll
    for (int i = 0; i < 4 * TI; i++) {
      const int m = ty * 4 + (i & 3), tile = tile0 + 4 * (i >> 2) + (m >> 4);
      uint32_t o = 0;
#pragma unroll
      for (int e = 0; e < 4; e++) o |= to_e4m3(store8(vit_act(acc[i][4 * h + e] * 65536.f))) << (8 * e);
      *(uint32_t*)(P.out + (size_t)tile * 65536 + ch_off(r) + tok_off(m & 15)) = o;
    }
  }
}

template <int K, int TN>
__device__ __forceinline__ void vit_contract(const ContractParams& P, uint2* lds) {
  const int tile0 = blockIdx.x * 4 * TI, n0 = blockIdx.y * 64 * TN, T = gridDim.x * 4;
  if (n0 >= 1024 || tile0 >= T) return;
  float acc[4 * TI][4 * TN];
  WPiece wp[TN];
#pragma unroll
  for (int h = 0; h < TN; h++) wp[h] = wpiece_std<K, 64>(P.w, n0 + 64 * h);
  gemm64<K, TN>(P.x, tile0, T, wp, acc, lds);
  const int ty = threadIdx.x >> 4;
  const _Float16* gain = (const _Float16*)(P.w + 1024 * K);
#pragma unroll
  for (int h = 0; h < TN; h++) {
    const int r = n0 + 64 * h + (threadIdx.x & 15) * 4;
    float g[4];
#pragma unroll
    for (int e = 0; e < 4; e++) g[e] = (float)gain[(r & ~31) | pi32((r + e) & 31)];
#pragma unroll
    for (int i = 0; i < 4 * TI; i++) {
      const int m = ty * 4 + (i & 3), tile = tile0 + 4 * (i >> 2) + (m >> 4);
      const size_t off = (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15);
      const uint32_t sd = *(const uint32_t*)(P.skip + off);
      uint32_t o = 0;
#pragma unroll
      for (int e = 0; e < 4; e++)
        o |= to_e4m3(store8(__fadd_rn(acc[i][4 * h + e] * 65536.f, __fmul_rn(e4f(sd >> (8 * e)), g[e])))) << (8 * e);
      *(uint32_t*)(P.out + off) = o;
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) k_vit_contract_n(ContractParams P) {
  constexpr int TN = VIT_TN_CONTRACT;
  __shared__ uint2 lds[2 * BUF<TN>];
  if (blockIdx.y >= 16) return;
  if (P.K == 4096) vit_contract<4096, TN>(P, lds);
  else if (P.K == 1024) vit_contract<1024, TN>(P, lds);
}

__device__ __forceinline__ f16 xorh(f16 v, int m) { return (f16)__shfl_xor((float)v, m, 32); }

extern "C" __global__ void __launch_bounds__(256) k_vit_qkv_n(QkvParams P) {
  constexpr int TN = VIT_TN_QKV;
  __shared__ uint2 lds[2 * BUF<TN>];
  constexpr int NB = 16 / TN;   // workgroups per q / k / v
  const int tile0 = blockIdx.x * 4 * TI, T = gridDim.x * 4;
  if (blockIdx.y >= 3 * NB || tile0 >= T) return;
  const int which = blockIdx.y / NB, n0 = (blockIdx.y % NB) * 64 * TN;
  float acc[4 * TI][4 * TN];
  WPiece wp[TN];
#pragma unroll
  for (int h = 0; h < TN; h++) wp[h] = wpiece_qkv3(P.w, which, n0 + 64 * h);
  gemm64<1024, TN>(P.x, tile0, T, wp, acc, lds);
  const int ty = threadIdx.x >> 4;
#pragma unroll
  for (int h = 0; h < TN; h++) {
    const int r = n0 + 64 * h + (threadIdx.x & 15) * 4;
    const f16 temp = (f16)((const float*)P.w)[r >> 5];
#pragma unroll
    for (int i = 0; i < 4 * TI; i++) {
      const int m = ty * 4 + (i & 3), tile = tile0 + 4 * (i >> 2) + (m >> 4);
      const float* ac = &acc[i][4 * h];
      float y[4];
      if (which == 2) {
#pragma unroll
        for (int e = 0; e < 4; e++) y[e] = store8(ac[e] * 65536.f);
      } else {
        f16 x[4], s1[4];
#pragma unroll
        for (int e = 0; e < 4; e++) { x[e] = (f16)(ac[e] * 65536.f); s1[e] = x[e] * x[e]; }
#pragma unroll
        for (int e = 0; e < 4; e++) s1[e] = s1[e] + xorh(s1[e], 4);
        f16 ta = s1[0] + s1[2], tb = s1[1] + s1[3];
        ta = ta + xorh(ta, 2); tb = tb + xorh(tb, 2);
        ta = ta + xorh(ta, 1); tb = tb + xorh(tb, 1);
        f16 tot = ta + tb;
        if (tot < (f16)0.000062f) tot = (f16)0.000062f;
        const f16 inv = (f16)(1.f / sqrtf((float)tot));
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const f16 v = which == 0 ? (f16)((f16)(x[e] * (f16)5.65625f) * inv) * temp : x[e] * inv;
          y[e] = q8((float)v);
        }
      }
      uint32_t o = 0;
#pragma unroll
      for (int e = 0; e < 4; e++) o |= to_e4m3(y[e]) << (8 * e);
      *(uint32_t*)(P.out[which] + (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15)) = o;
    }
  }
}
