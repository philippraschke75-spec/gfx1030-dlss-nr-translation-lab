// Native ViT GEMM kernels (k_vit_expand_n / k_vit_contract_n / k_vit_qkv_n) of native-c512/c512.hip with a
// bank-conflict-free LDS layout. Only where the staged halves sit in LDS changes; every thread reads the same values and
// runs the same dot2 sequence, so the output is byte-identical (proof + conflict counts: lds_model.py).
//  A[k][tok]: tok ^ (((k>>1)&3)<<2)                          write 4-way -> 1-way, read unchanged (broadcast)
//  W[k][n]  : 4tx+e -> (e>>1)*32 + 2tx + (e&1), ^ (((k>>1)&3)<<2)   write 4-way -> 1-way, read b128 2-way -> 1-way
#include "nk_common.h"

// thread index within a 256-thread GEMM group (k_vit_contract_n runs two groups, one per K half)
#define LT ((int)(threadIdx.x & 255))

struct WPiece { const uint8_t* src; size_t step; int wn[4], wk[4]; };

template <int K, int INNER>
__device__ __forceinline__ WPiece wpiece_std(const uint8_t* Wr, int n0) {
  const int tid = LT, q = tid & 31, ntl = tid >> 5 & 3, kgl = tid >> 7, nt = (n0 >> 4) + ntl;
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
  const int tid = LT, s = tid >> 6, rtl = s & 1, ktl = s >> 1, u = tid & 63;
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

__device__ __forceinline__ int lin_tile(int tile0, int T) { const int t = tile0 + (LT >> 6); return t < T ? t : -1; }
__device__ __forceinline__ int swz(int k) { return ((k >> 1) & 3) << 2; }
__device__ __forceinline__ int sA(int k, int c) { return k * 64 + (c ^ swz(k)); }
__device__ __forceinline__ int sW(int k, int c) { return k * 64 + (((c & 2) * 16 + (c >> 2) * 2 + (c & 1)) ^ swz(k)); }

__device__ __forceinline__ void mma16s(const uint2* As, const uint2* Ws, float acc[4][4]) {
  const int ty = LT >> 4, tx = LT & 15;
#pragma unroll 4
  for (int kd = 0; kd < 16; kd++) {
    const int ao = kd * 64 + ((ty * 4) ^ swz(kd)), wo = kd * 64 + ((tx * 2) ^ swz(kd));
    const uint4 a01 = *(const uint4*)(As + ao), a23 = *(const uint4*)(As + ao + 2);
    const uint4 w01 = *(const uint4*)(Ws + wo), w23 = *(const uint4*)(Ws + wo + 32);
    const uint32_t A[4][2] = {{a01.x, a01.y}, {a01.z, a01.w}, {a23.x, a23.y}, {a23.z, a23.w}};
    const uint32_t B[4][2] = {{w01.x, w01.y}, {w01.z, w01.w}, {w23.x, w23.y}, {w23.z, w23.w}};
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[i][e] = dot2(A[i][1], B[e][1], dot2(A[i][0], B[e][0], acc[i][e]));
  }
}

// chunks c0..c1-1 of the K/64 64-wide k chunks (default: all)
template <int K>
__device__ __forceinline__ void gemm64(const uint8_t* __restrict__ X, int atile, const WPiece& wp, float acc[4][4], uint2* lds,
                                       int c0 = 0, int c1 = K / 64) {
  const int NC = c1;
  const int tid = LT, tt = tid >> 6, sa = (tid >> 5) & 1, q = tid & 31;
  auto loadA = [&](int c) -> uint4 {
    if (atile < 0) return make_uint4(0, 0, 0, 0);
    return *(const uint4*)(X + (size_t)atile * (K * 16) + sa * 512 + q * 16 + (size_t)c * 1024);
  };
  int ao[4], wo[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    ao[e] = sA(sa * 8 + 2 * (q & 3) + (e >> 1), tt * 16 + (q >> 2) + 8 * (e & 1));
    wo[e] = sW(wp.wk[e], wp.wn[e]);
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int e = 0; e < 4; e++) acc[i][e] = 0.f;
  uint4 ra = loadA(c0);
  uint4 rw = *(const uint4*)(wp.src + (size_t)c0 * wp.step);
  for (int c = c0; c < NC; c++) {
    uint2* As = lds + (c & 1) * 2048;
    uint2* Ws = As + 1024;
    const uint32_t av[4] = {ra.x, ra.y, ra.z, ra.w}, wv[4] = {rw.x, rw.y, rw.z, rw.w};
#pragma unroll
    for (int e = 0; e < 4; e++) { As[ao[e]] = cvt4(av[e]); Ws[wo[e]] = cvt4(wv[e]); }
    __syncthreads();
    if (c + 1 < NC) {
      ra = loadA(c + 1);
      rw = *(const uint4*)(wp.src + (size_t)(c + 1) * wp.step);
    }
    mma16s(As, Ws, acc);
  }
}

// ---- kernels: epilogues verbatim from c512.hip
struct ExpandParams { const uint8_t* x; uint8_t* out; const uint8_t* w; };
struct ContractParams { const uint8_t* x; const uint8_t* skip; uint8_t* out; const uint8_t* w; int K, KN, four; };
struct QkvParams { const uint8_t* x; uint8_t* out[3]; const uint8_t* w; };

__device__ __forceinline__ float vit_act(float x) {
  const float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}

extern "C" __global__ void __launch_bounds__(256) k_vit_expand_n(ExpandParams P) {
  __shared__ uint2 lds[4096];
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4;
  if (n0 >= 4096) return;
  float acc[4][4];
  gemm64<1024>(P.x, lin_tile(tile0, T), wpiece_std<1024, 256>(P.w, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) o |= to_e4m3(store8(vit_act(acc[i][e] * 65536.f))) << (8 * e);
    *(uint32_t*)(P.out + (size_t)tile * 65536 + ch_off(r) + tok_off(m & 15)) = o;
  }
}

// 512 threads: group h = tid >> 8 runs K half h on its own 32 KB of LDS; group 1 hands its f32 sums to group 0
// through LDS, which adds them (lo + hi) and runs the epilogue. Not bit-exact (one f32 reassociation per output),
// but deterministic. Why: the 64x64-tile grid is only 3 x 16 = 48 workgroups on 80 CUs, one per CU, each a
// 64-chunk serial chain; two groups per workgroup halve the chain and double the waves per CU.
template <int K>
__device__ __forceinline__ void vit_contract(const ContractParams& P, uint2* lds) {
  const int tile0 = blockIdx.x * 4, n0 = blockIdx.y * 64, T = gridDim.x * 4, h = threadIdx.x >> 8, ns = blockDim.x >> 8;
  float acc[4][4];   // ns = 1 (256-thread launch, the old override line): one group, all of K, bit-exact as before
  gemm64<K>(P.x, lin_tile(tile0, T), wpiece_std<K, 64>(P.w, n0), acc, lds + h * 4096, h * (K / 64) / ns, (h + 1) * (K / 64) / ns);
  if (ns > 1) {
    __syncthreads();
    float* red = (float*)lds;
    if (h) {
#pragma unroll
      for (int j = 0; j < 16; j++) red[j * 256 + LT] = acc[j >> 2][j & 3];
    }
    __syncthreads();
    if (h) return;
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j >> 2][j & 3] = __fadd_rn(acc[j >> 2][j & 3], red[j * 256 + LT]);
  }
  const int ty = LT >> 4, r = n0 + (LT & 15) * 4;
  const _Float16* gain = (const _Float16*)(P.w + 1024 * K);
  float g[4];
#pragma unroll
  for (int e = 0; e < 4; e++) g[e] = (float)gain[(r & ~31) | pi32((r + e) & 31)];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    const size_t off = (size_t)tile * 16384 + ch_off(r) + tok_off(m & 15);
    const uint32_t sd = *(const uint32_t*)(P.skip + off);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++)
      o |= to_e4m3(store8(__fadd_rn(acc[i][e] * 65536.f, __fmul_rn(e4f(sd >> (8 * e)), g[e])))) << (8 * e);
    *(uint32_t*)(P.out + off) = o;
  }
}
extern "C" __global__ void __launch_bounds__(512) k_vit_contract_n(ContractParams P) {   // 512 threads (override)
  __shared__ uint2 lds[8192];
  if (blockIdx.y >= 16) return;
  if (P.K == 4096) vit_contract<4096>(P, lds);
  else if (P.K == 1024) vit_contract<1024>(P, lds);
}

__device__ __forceinline__ f16 xorh(f16 v, int m) { return (f16)__shfl_xor((float)v, m, 32); }

extern "C" __global__ void __launch_bounds__(256) k_vit_qkv_n(QkvParams P) {
  __shared__ uint2 lds[4096];
  if (blockIdx.y >= 48) return;
  const int which = blockIdx.y >> 4, n0 = (blockIdx.y & 15) * 64, tile0 = blockIdx.x * 4;
  float acc[4][4];
  gemm64<1024>(P.x, lin_tile(tile0, gridDim.x * 4), wpiece_qkv3(P.w, which, n0), acc, lds);
  const int ty = threadIdx.x >> 4, r = n0 + (threadIdx.x & 15) * 4;
  const f16 temp = (f16)((const float*)P.w)[r >> 5];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i, tile = tile0 + (m >> 4);
    float y[4];
    if (which == 2) {
#pragma unroll
      for (int e = 0; e < 4; e++) y[e] = store8(acc[i][e] * 65536.f);
    } else {
      f16 x[4], s1[4];
#pragma unroll
      for (int e = 0; e < 4; e++) { x[e] = (f16)(acc[i][e] * 65536.f); s1[e] = x[e] * x[e]; }
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
