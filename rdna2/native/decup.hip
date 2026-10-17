// Native k_dec_upsample (block 39, ViT -> decoder): out = e4m3(f16(proj + f16(skip * gain))), proj = W[512][1024] x
// at the ViT token (nearest 2x upsample), bit-exact with the translated kernel:
//  * K runs in 32-channel slabs; inside a slab the dot2 chain goes over storage k pairs (0,1),(2,3),...,(30,31) from 0,
//    then run = f16(run + slab) (the original keeps the running sum in f16 between two WMMAs per slab).
//  * gain index = reference channel of the storage channel; skip*gain is one f32 product rounded to f16, the add is f16.
// Grid (=, 4): WG w = y*gx + x -> ViT tile w>>3 (16 tokens), 64 of the 512 outputs (w&7); extra WGs exit.
// 256 threads: thread (ty = tid>>4 token, tx = tid&15 -> 4 storage channels).
#include "nk_common.h"


extern "C" __global__ void __launch_bounds__(256) k_dec_upsample_n(DecUpParams P) {
  __shared__ uint2 As[16 * 16];   // [kd][token]
  __shared__ uint2 Ws[16 * 64];   // [kd][n]
  const int W6 = (P.W5 + 1) >> 1, H6 = (P.H5 + 1) >> 1, TW6 = (W6 + 3) >> 2, TH6 = (H6 + 3) >> 2;
  const int wg = blockIdx.y * gridDim.x + blockIdx.x, t6 = wg >> 3, n0 = (wg & 7) * 64;
  if (t6 >= TW6 * TH6) return;
  const int tid = threadIdx.x, q = tid & 31;
  const int ty6 = t6 / TW6, tx6 = t6 - ty6 * TW6;
  // A piece (threads 0..63): slab sa, 16 B at q*16; dword e -> token (q>>2) + 8*(e&1), k dword (q&3) + 4*(e>>1)
  const int sa = (tid >> 5) & 1;
  const uint8_t* asrc = P.x + (size_t)t6 * 16384 + sa * 512 + q * 16;
  bool aval[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int p = (q >> 2) + 8 * h, y6 = ty6 * 4 + (p >> 2), x6 = tx6 * 4 + (p & 3);
    aval[h] = y6 < H6 && x6 < W6;
  }
  // W piece: slice (kg*32 + nt), 16 B at q*16; dword e -> n = nt*16 + (e>>1)*8 + (q>>2), k dword (e&1)*4 + (q&3)
  const int kgl = tid >> 7, nt = (n0 >> 4) + ((tid >> 5) & 3);
  const uint8_t* wsrc = P.w + (size_t)(kgl * 32 + nt) * 512 + q * 16;
  int wn[4], wk[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const int n = nt * 16 + (e >> 1) * 8 + (q >> 2);
    wn[e] = ((n & ~31) | ipi32(n & 31)) - n0;
    wk[e] = kgl * 8 + (e & 1) * 4 + (q & 3);
  }
  const int ty = tid >> 4, tx = tid & 15;
  f16 run[4] = {0, 0, 0, 0};
  uint4 ra = tid < 64 ? *(const uint4*)asrc : make_uint4(0, 0, 0, 0);
  uint4 rw = *(const uint4*)wsrc;
  for (int c = 0; c < 16; c++) {   // 64 k per round = 2 slabs
    if (tid < 64) {
      const uint32_t av[4] = {ra.x, ra.y, ra.z, ra.w};
#pragma unroll
      for (int e = 0; e < 4; e++) As[(sa * 8 + (q & 3) + 4 * (e >> 1)) * 16 + (q >> 2) + 8 * (e & 1)] = cvt4(aval[e & 1] ? av[e] : 0u);
    }
    const uint32_t wv[4] = {rw.x, rw.y, rw.z, rw.w};
#pragma unroll
    for (int e = 0; e < 4; e++) Ws[wk[e] * 64 + wn[e]] = cvt4(wv[e]);
    __syncthreads();
    if (c + 1 < 16) {
      if (tid < 64) ra = *(const uint4*)(asrc + (size_t)(c + 1) * 1024);
      rw = *(const uint4*)(wsrc + (size_t)(c + 1) * 2 * 32 * 512);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kd = 0; kd < 16; kd++) {
      const uint2 a = As[kd * 16 + ty];
      const uint4 w01 = *(const uint4*)(Ws + kd * 64 + tx * 4), w23 = *(const uint4*)(Ws + kd * 64 + tx * 4 + 2);
      const uint32_t B[4][2] = {{w01.x, w01.y}, {w01.z, w01.w}, {w23.x, w23.y}, {w23.z, w23.w}};
#pragma unroll
      for (int e = 0; e < 4; e++) acc[e] = dot2(a.y, B[e][1], dot2(a.x, B[e][0], acc[e]));
      if (kd == 7 || kd == 15) {
#pragma unroll
        for (int e = 0; e < 4; e++) { run[e] = (f16)__fadd_rn((float)run[e], acc[e] * 65536.f); acc[e] = 0.f; }
      }
    }
    __syncthreads();
  }
  // epilogue: token ty of the tile -> 2x2 output pixels at level 5
  const int r = n0 + tx * 4;
  const f16* gain = (const f16*)(P.w + 1024 * 512);
  f16 g[4];
#pragma unroll
  for (int e = 0; e < 4; e++) g[e] = gain[(r & ~31) | pi32((r + e) & 31)];
  const int y6 = ty6 * 4 + (ty >> 2), x6 = tx6 * 4 + (ty & 3), TW5 = P.W5 / 4, T5 = gridDim.x;
#pragma unroll
  for (int d = 0; d < 4; d++) {
    const int y5 = 2 * y6 + (d >> 1), x5 = 2 * x6 + (d & 1), tx5 = x5 >> 2, t5 = (y5 >> 2) * TW5 + tx5;
    if (tx5 >= TW5 || t5 >= T5) continue;
    const size_t off = (size_t)t5 * 8192 + ch_off(r) + tok_off((y5 & 3) * 4 + (x5 & 3));
    const uint32_t sd = *(const uint32_t*)(P.skip + off);
    uint32_t o = 0;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const f16 gated = (f16)__fmul_rn(e4f(sd >> (8 * e)), (float)g[e]);
      const f16 v = run[e] + gated;
      o |= e4m3_byte((float)v) << (8 * e);
    }
    *(uint32_t*)(P.out + off) = o;
  }
}
