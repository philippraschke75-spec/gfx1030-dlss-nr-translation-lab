// Native gfx1030 multi-head Swin block (Swin2H/4H/8H, swin.py) on the arena layout of k_swin_var<C>, C = 64/128/256.
// Drop-ins for _Z10k_swin_varILi{64,128,256}ELb0EEv9VarParams: same VarParams kernarg, same grid (one 8x8 window per
// workgroup), 256 threads. A window's whole block stays in LDS; weights are read straight from the e4m3 record and
// converted to f16 (x 2^-8, undone in the epilogues) while being staged into LDS.
// Rounding points follow dlssnr/layers/swin.py. Build with -ffp-contract=off. Layouts: native-mid/RESULTS.md.
//  activation (inside a stage): e4m3, 4x4-pixel tiles of C*16 B, tile = (row/4)*(W/4) + col/4, C/32 chunks of 512 B,
//     byte = chunk*512 + (p%8)*64 + 4*(p>=8) + 16*((c%16)/4) + 8*(c>=16) + (c&3), p = (row%4)*4 + col%4,
//     reference channel = 32*chunk + PI[c], PI = c0|c2<<1|c3<<2|c1<<3|c4<<4.
//  between stages (flags 1 in, 2 out, 4 -> +0x38, 8 <- +0x30): planes [C/16][H][W][16], same interleave.
//  weight matrix [N][K]: 16x32 tiles, tile = g*(GN*K/32) + kb*GN + j (n-tile = g*GN + j),
//     byte in tile = (n&7)*64 + ((k>>1)&3)*16 + (n>>3)*8 + (k>>3)*2 + (k&1).
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef KPU   // k-loop unroll: 4 (C >= 128), 2 (C = 64, fewer live registers)
#define KPU (C == 64 ? 2 : 4)
#endif
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

struct VarParams {
  const uint8_t* in;    // +0x00
  uint8_t* out;         // +0x08
  const uint8_t* w;     // +0x10 weight record
  int H, W;             // +0x18 rows, columns (padded extent)
  int X0, Y0;           // +0x20 column origin, +0x24 row origin
  uint32_t flags;       // +0x28: 1 first (planes in), 2 last decoder (planes out), 4 downsample, 8 upsample
  uint32_t pad0;
  const uint8_t* p30;   // +0x30 coarse level (upsample)
  uint8_t* p38;         // +0x38 next level (downsample)
  uint8_t rest[0xa8 - 0x40];
};

__device__ __forceinline__ float q8(float x) {
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
}
__device__ __forceinline__ float f16r(float x) { return (float)(_Float16)x; }
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}
__device__ __forceinline__ uint32_t pk(float a, float b) {
  h2 r; r.x = (_Float16)a; r.y = (_Float16)b; return __builtin_bit_cast(uint32_t, r);
}
__device__ __forceinline__ float lo(uint32_t u) { return (float)__builtin_bit_cast(h2, u).x; }
__device__ __forceinline__ float hi(uint32_t u) { return (float)__builtin_bit_cast(h2, u).y; }
// two e4m3 bytes (low 16 bits of u) -> half2 holding value * 2^-8 (exact, incl. subnormals)
__device__ __forceinline__ uint32_t e4x2(uint32_t u) {
  return ((u & 0x7Fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7F00u) << 15) | ((u & 0x8000u) << 16);
}
__device__ __forceinline__ uint32_t x256(uint32_t v) {
  return __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, v) * (h2){(_Float16)256.f, (_Float16)256.f});
}
__device__ __forceinline__ uint32_t to_e4m3(float v) {  // v already on the e4m3 grid
  float a = fabsf(v);
  uint32_t b = a < 0.015625f ? (uint32_t)(a * 512.f) : ((__float_as_uint(a) >> 20) - 960u);
  return b | (v < 0.f ? 0x80u : 0u);
}
__device__ __forceinline__ float act(float x) {
  float t = fminf(fmaxf(x, -4.f), 4.f);
  return __fmul_rn(x, fmaf(t, fmaf(fabsf(t), -0.055908203125f, 0.447265625f), 0.89453125f));
}
__device__ __forceinline__ float norm_scale(const float* v) {  // from native-swin32
  float t[4][2] = {};
#pragma unroll
  for (int g = 0; g < 2; g++)
#pragma unroll
    for (int f = 0; f < 2; f++)
#pragma unroll
      for (int c = 0; c < 4; c++)
#pragma unroll
        for (int h = 0; h < 2; h++) { float x = v[16 * f + 8 * g + 2 * c + h]; t[c][h] = fmaf(x, x, t[c][h]); }
  _Float16 h0 = (_Float16)((t[0][0] + t[1][0]) + (t[2][0] + t[3][0]));
  _Float16 h1 = (_Float16)((t[0][1] + t[1][1]) + (t[2][1] + t[3][1]));
  _Float16 s = h0 + h1;
  const _Float16 floor_ = (_Float16)0.000062f;
  if (s < floor_) s = floor_;
  return __fdiv_rn(1.f, (float)sqrt((double)(float)s));
}
__device__ __forceinline__ int pi32(int c) {   // memory channel -> reference channel (within 32)
  return (c & 1) | (((c >> 2) & 1) << 1) | (((c >> 3) & 1) << 2) | (((c >> 1) & 1) << 3) | (c & 16);
}
__device__ __forceinline__ int ipi32(int m) {  // reference channel -> memory channel
  return (m & 1) | (((m >> 3) & 1) << 1) | (((m >> 1) & 1) << 2) | (((m >> 2) & 1) << 3) | (m & 16);
}
template <int N>
__device__ __forceinline__ void ldw(const uint32_t* p, uint32_t* o) {   // N consecutive LDS words (N = 1, 2, 4, 8)
  if constexpr (N == 1) { o[0] = p[0]; }
  else if constexpr (N == 2) { uint2 u = *(const uint2*)p; o[0] = u.x; o[1] = u.y; }
  else {
#pragma unroll
    for (int i = 0; i < N; i += 4) { uint4 u = *(const uint4*)(p + i); o[i] = u.x; o[i + 1] = u.y; o[i + 2] = u.z; o[i + 3] = u.w; }
  }
}

// one 16-B piece (q = 0..31 within a 512-B tile) -> dst[kq + 4m][ncol + nh*8 + nl] as half2 k-pairs
__device__ __forceinline__ void put16(uint32_t* dst, int ldn, uint4 u, int q, int ncol) {
  const int nl = q >> 2, kq = q & 3;
  uint32_t w0[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int nh = 0; nh < 2; nh++)
#pragma unroll
    for (int m = 0; m < 4; m++) {
      uint32_t word = w0[nh * 2 + (m >> 1)];
      uint32_t v = (m & 1) ? (word >> 16) : word;
      dst[(kq + 4 * m) * ldn + ncol + nh * 8 + nl] = e4x2(v & 0xFFFFu);
    }
}
// Stage `ntiles` tiles; tile i comes from rec byte offset off(i), lands at column ncol(i), kp row kpo(i).
template <typename F>
__device__ __forceinline__ void stage(uint32_t* dst, int ldn, const uint8_t* rec, int ntiles, F where) {
  for (int i = threadIdx.x; i < ntiles * 32; i += 256) {
    int tile = i >> 5, q = i & 31, off, ncol, kpo;
    where(tile, off, ncol, kpo);
    put16(dst + kpo * ldn, ldn, *(const uint4*)(rec + off + q * 16), q, ncol);
  }
}

template <int C>
__device__ __forceinline__ void swin_block(const VarParams& P) {
  constexpr int NH = C / 32, KB = C / 32;                  // heads, 32-wide K steps over C
  constexpr int PWD = C >= 128 ? 128 : 64, NPASS = C / PWD, CPT = PWD / 32;   // C-wide outputs: tx*CPT within a pass
  // SMALL (C = 64): 28 KB of LDS so that two workgroups fit a CU. The hidden chunk is 64 instead of 128, q/k/Pm/ctx
  // are spread over R1 and R3 (see the attention phase), and the softmax reciprocals live in R1.
  constexpr bool SMALL = C == 64;
  constexpr int HC = SMALL ? 64 : 128, NS = 128 / HC, HCPT = HC / 32;   // hidden chunk, sub-chunks per head
  __shared__ uint32_t Xs[C / 2 * 64];          // [kp][tok]: x, later Yq (attention input and skip)
  __shared__ uint32_t R1[SMALL ? 2048 : 4096];  // Hs[HC/2 kp][64] | q (+k) f32 | Pm | ctx/Wo | coarse/pooled
  __shared__ uint32_t R2[1024];                // Ms[16 kp][64 tok] | Vt f16 [32 d][64 key]
  __shared__ uint32_t R3[2048];                // weight stage | qn,kn or k | Pm | ctx | Pj f16 [16][C]
  __shared__ float rec_arr[SMALL ? 1 : 64];
  float* rec_s = SMALL ? (float*)(R1 + 1024) : rec_arr;

  const int tid = threadIdx.x;
  const uint8_t* rec = P.w;
  const int H = P.H, W = P.W, WT = W >> 2;
  const int r0 = P.Y0 + blockIdx.y * 8, c0 = P.X0 + blockIdx.x * 8;
  const bool up = P.flags & 8;
  // record offsets (dlssnr extract.py _swin_layout)
  constexpr int R_EXP = 0, R_MID = 4 * C * C, R_CON = R_MID + C * 128, R_AFT = R_CON + C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT;
  const int R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C - 32 : 0);
  const int R_BIAS = R_QKV + 3 * C * C, R_HS = R_BIAS + NH * 8192, R_OUT = R_HS + (4 * NH > 16 ? 4 * NH : 16);
  const int R_ARS = R_OUT + C * C, R_DSRS = R_ARS + 2 * C;
  const _Float16* rs = (const _Float16*)(rec + R_RS);
  const uint4 rs8 = *(const uint4*)rs;
  const int rso = (rs8.x | rs8.y | rs8.z | rs8.w) == 0 ? 8 : 0;   // zero-padded table: MLP scales start at 8
  _Float16* Pj = (_Float16*)R3;
  const int tx = tid & 31, ty = tid >> 5;   // C-wide GEMM tiles: tokens ty*8..+7, channels chan(pass, c)
  auto chan = [&](int pass, int c) { return pass * PWD + tx * CPT + c; };

  // ---- upsample: Pj[16 pix][C] = f16(Wr . coarse) for the 4x4 coarse pixels under the window (K = 2C)
  if (up) {
    const int H2 = H >> 1, W2 = W >> 1, pr0 = r0 >> 1, pc0 = c0 >> 1;
    if (tid < C) {   // (pixel, 32-channel chunk of 2C)
      const int px = tid & 15, ch = tid >> 4;
      const int pr = pr0 + (px >> 2), pc = pc0 + (px & 3);
      uint32_t b[8] = {};
      if (pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
        const uint8_t* src = P.p30 + (size_t)(ch * 2) * H2 * W2 * 16 + (size_t)(pr * W2 + pc) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)H2 * W2 * 16 + 4 * g); }
      }
#pragma unroll
      for (int g = 0; g < 4; g++)
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            int c = 16 * s2 + 4 * g + 2 * hh;
            R1[((ch * 32 + pi32(c)) >> 1) * 16 + px] = x256(e4x2((b[2 * g + s2] >> (16 * hh)) & 0xFFFFu));
          }
    }
    constexpr int CQ = C / 64;
    const int tx6 = tid & 63, ty6 = tid >> 6;
    float acc[4][CQ] = {};
#pragma unroll 1
    for (int kb = 0; kb < C / 16; kb++) {
      __syncthreads();
      stage(Xs, C, rec, C / 16, [&](int i, int& off, int& ncol, int& kpo) {
        off = R_AFT + (kb * (C / 16) + i) * 512; ncol = i * 16; kpo = 0; });
      __syncthreads();
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a = *(const uint4*)&R1[(kb * 16 + kp) * 16 + ty6 * 4];
        uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[CQ];
        ldw<CQ>(&Xs[kp * C + tx6 * CQ], bv);
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
          for (int c = 0; c < CQ; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
      }
    }
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
      for (int c = 0; c < CQ; c++) Pj[(ty6 * 4 + r) * C + tx6 * CQ + c] = (_Float16)(acc[r][c] * 256.f);
    __syncthreads();
  }

  // ---- load the window: token t = (qy*2+qx)*16 + y*4 + x
  for (int i = tid; i < 64 * (C / 32); i += 256) {   // (token, 32-channel chunk)
    int t = i & 63, ch = i >> 6;
    int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
    bool in = row >= 0 && row < H && col >= 0 && col < W;
    uint32_t b[8] = {};
    if (in) {
      int p = (row & 3) * 4 + (col & 3);
      if (P.flags & 1) {
        const uint8_t* src = P.in + (size_t)(ch * 2) * H * W * 16 + (size_t)(row * W + col) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)H * W * 16 + 4 * g); }
      } else {
        const uint8_t* src = P.in + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + ch * 512 + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 16 * g); b[2 * g + 1] = *(const uint32_t*)(src + 16 * g + 8); }
      }
    }
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          int c = 16 * s + 4 * g + 2 * hh;
          uint32_t v = x256(e4x2((b[2 * g + s] >> (16 * hh)) & 0xFFFFu));
          const int rc = ch * 32 + pi32(c);
          if (up) {   // x = q8(f16(skip * gain + P(coarse))), zero outside the frame
            const int lr = ((t >> 5) & 1) * 4 + ((t >> 2) & 3), lc = ((t >> 4) & 1) * 4 + (t & 3);
            const int pix = (lr >> 1) * 4 + (lc >> 1);
            const _Float16* ug = (const _Float16*)(rec + R_RS + 2 * C + 32);
            float g0 = rc < 16 ? (float)rs[C + rc] : (float)ug[rc - 16];
            float g1 = rc + 1 < 16 ? (float)rs[C + 1 + rc] : (float)ug[rc - 15];
            float x0 = in ? q8(f16r((float)((double)lo(v) * g0 + (double)(float)Pj[pix * C + rc]))) : 0.f;
            float x1 = in ? q8(f16r((float)((double)hi(v) * g1 + (double)(float)Pj[pix * C + rc + 1]))) : 0.f;
            v = pk(x0, x1);
          }
          Xs[(rc >> 1) * 64 + t] = v;
        }
  }

  float accc[NPASS][8][CPT];
#pragma unroll
  for (int a = 0; a < NPASS; a++)
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
      for (int c = 0; c < CPT; c++) accc[a][r][c] = 0.f;

  // ================= MLP, one head (128 hidden values) at a time, in NS sub-chunks of HC
#pragma unroll 1
  for (int h = 0; h < NH; h++) {
    const bool ldc = tid < PWD * 2;   // contract pieces of one pass
    auto con_src = [&](int pass) {
      return *((const uint4*)(rec + R_CON + (h * (C / 16) + pass * (PWD / 16)) * 512) + (ldc ? tid : 0)); };
    uint4 wc = con_src(0);
    float macc[2][4] = {};
    const int tx8 = tid & 7, ty8 = tid >> 3;
#pragma unroll 1
    for (int sc = 0; sc < NS; sc++) {
      // ---- expand: Hs[64 tok][HC] = q8(act(X . We[h*128 + sc*HC .. +HC]^T)), K = C (GN = 8)
      const bool ldw16 = tid < HC * 2;   // HC/16 tiles of 32 pieces
      uint4 wm = *((const uint4*)(rec + R_MID + (h * 8 + sc * (HC / 16)) * 512) + (ldw16 ? tid : 0));
      {
        float acc[8][HCPT] = {};
        const uint4* wsrc = (const uint4*)(rec + R_EXP + (h * 8 * KB + sc * (HC / 16)) * 512) + (ldw16 ? tid : 0);
        uint4 wn = wsrc[0];
#pragma unroll 1
        for (int kb = 0; kb < KB; kb++) {
          __syncthreads();
          if (ldw16) put16(R3, HC, wn, tid & 31, (tid >> 5) * 16);
          __syncthreads();
          if (kb + 1 < KB) wn = wsrc[(kb + 1) * 256];
#pragma unroll KPU
          for (int kp = 0; kp < 16; kp++) {
            uint4 a0 = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty * 8];
            uint4 a1 = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty * 8 + 4];
            uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[HCPT];
            ldw<HCPT>(&R3[kp * HC + tx * HCPT], bv);
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
              for (int c = 0; c < HCPT; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
          }
        }
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < HCPT; c += 2)
            R1[((tx * HCPT + c) >> 1) * 64 + ty * 8 + r] = pk(q8(act(acc[r][c] * 256.f)), q8(act(acc[r][c + 1] * 256.f)));
      }
      __syncthreads();
      // ---- mid partial: macc += Hs . Wm_h[:, sc*HC .. +HC]^T (GN = 2); local tile i = tid>>5 -> kb = i>>1, j = i&1
      if (ldw16) put16(R3 + (tid >> 6) * 16 * 32, 32, wm, tid & 31, ((tid >> 5) & 1) * 16);
      __syncthreads();
#pragma unroll 8
      for (int kp = 0; kp < HC / 2; kp++) {
        uint2 a = *(const uint2*)&R1[kp * 64 + ty8 * 2];
        uint4 bb = *(const uint4*)&R3[kp * 32 + tx8 * 4];
        uint32_t av[2] = {a.x, a.y}, bv[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int c = 0; c < 4; c++) macc[r][c] = dot2(av[r], bv[c], macc[r][c]);
      }
    }
    // Ms[64 tok][32] = q8(f16(mid)); the previous head's contract finished reading R2 long ago
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int c = 0; c < 4; c += 2)
        R2[((tx8 * 4 + c) >> 1) * 64 + ty8 * 2 + r] = pk(q8(f16r(macc[r][c] * 256.f)), q8(f16r(macc[r][c + 1] * 256.f)));
    // ---- contract partial: accc += Ms . Wc[:, h*32 .. +32]^T (GN = C/16), passes of PWD output channels
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++) {
      __syncthreads();
      if (ldc) put16(R3, PWD, wc, tid & 31, (tid >> 5) * 16);
      __syncthreads();
      if (pass + 1 < NPASS) wc = con_src(pass + 1);
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a0 = *(const uint4*)&R2[kp * 64 + ty * 8];
        uint4 a1 = *(const uint4*)&R2[kp * 64 + ty * 8 + 4];
        uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bv[CPT];
        ldw<CPT>(&R3[kp * PWD + tx * CPT], bv);
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < CPT; c++) accc[pass][r][c] = dot2(av[r], bv[c], accc[pass][r][c]);
      }
    }
  }

  // ---- y = rs * x + mlp ; Yq = q8(f16(y)) replaces x in Xs (multi-head: it is both attention input and skip)
  {
    uint32_t yq[NPASS][8][CPT / 2];
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < CPT; c += 2) {
          int ch = chan(pass, c), t = ty * 8 + r;
          uint32_t xv = Xs[(ch >> 1) * 64 + t];
          float y0 = __fadd_rn(__fmul_rn(lo(xv), (float)rs[rso + ch]), accc[pass][r][c] * 256.f);
          float y1 = __fadd_rn(__fmul_rn(hi(xv), (float)rs[rso + ch + 1]), accc[pass][r][c + 1] * 256.f);
          yq[pass][r][c >> 1] = pk(q8(f16r(y0)), q8(f16r(y1)));
        }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < CPT / 2; c++) Xs[(chan(pass, 0) / 2 + c) * 64 + ty * 8 + r] = yq[pass][r][c];
  }

  // ================= attention, one head at a time; out_proj accumulated per head
  float acco[NPASS][8][CPT];
#pragma unroll
  for (int a = 0; a < NPASS; a++)
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
      for (int c = 0; c < CPT; c++) acco[a][r][c] = 0.f;
  // buffers per head; SMALL moves k, kn, Pm into R3 and qn, ctx, Wo, the reciprocals into R1
  float* qb = (float*)R1;
  float* kbuf = SMALL ? (float*)R3 : (float*)R1 + 2048;
  uint32_t* qn = SMALL ? R1 : R3;
  uint32_t* kn = SMALL ? R3 : R3 + 1024;
  uint32_t* Pm = SMALL ? R3 : R1;
  uint32_t* ctx = SMALL ? R1 : R3;
  uint32_t* wo = SMALL ? R1 + 1024 : R1;
  _Float16* Vt = (_Float16*)R2;
  const float* hsc = (const float*)(rec + R_HS);

#pragma unroll 1
  for (int h = 0; h < NH; h++) {
    // ---- qkv_h: [64 tok][96] = Yq . Wqkv[h*96 .. +96]^T, K = C (GN = 3C/16)
    {
      const int tx16 = tid & 15, ty16 = tid >> 4;
      float acc[4][6] = {};
      const bool ld = tid < 192;
      const uint4* wsrc = (const uint4*)(rec + R_QKV + h * 6 * 512) + (ld ? tid : 0);
      uint4 wn = wsrc[0];
#pragma unroll 1
      for (int kb = 0; kb < KB; kb++) {
        __syncthreads();
        if (ld) put16(R3, 96, wn, tid & 31, (tid >> 5) * 16);
        __syncthreads();
        if (kb + 1 < KB) wn = wsrc[(kb + 1) * (3 * C / 16) * 32];
#pragma unroll KPU
        for (int kp = 0; kp < 16; kp++) {
          uint4 a = *(const uint4*)&Xs[(kb * 16 + kp) * 64 + ty16 * 4];
          uint2 b0 = *(const uint2*)&R3[kp * 96 + tx16 * 6];
          uint2 b1 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 2];
          uint2 b2 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 4];
          uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[6] = {b0.x, b0.y, b1.x, b1.y, b2.x, b2.y};
#pragma unroll
          for (int r = 0; r < 4; r++)
#pragma unroll
            for (int c = 0; c < 6; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
        }
      }
      __syncthreads();   // everyone is done with R1 (Wo of the previous head) and R2 (Vt)
#pragma unroll
      for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < 6; c++) {
          int n = tx16 * 6 + c, t = ty16 * 4 + r;
          float v = acc[r][c] * 256.f;
          if (n < 32) qb[n * 64 + t] = v;
          else if (n < 64) kbuf[(n - 32) * 64 + t] = v;
          else Vt[(n - 64) * 64 + t] = (_Float16)q8(f16r(v));
        }
    }
    __syncthreads();
    // ---- L2-normalise q (with the head scale) and k, q8 -> R3 qn [16 kp][64], kn [16 kp][64]
    if (tid < 128) {
      const int t = tid & 63, w = tid >> 6;
      float v[32];
#pragma unroll
      for (int d = 0; d < 32; d++) v[d] = (w ? kbuf : qb)[d * 64 + t];
      float sc = norm_scale(v);
      if (w == 0) sc = __fmul_rn(sc, hsc[h]);
#pragma unroll
      for (int d = 0; d < 32; d += 2) (w ? kn : qn)[(d >> 1) * 64 + t] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d + 1], sc)));
    }
    __syncthreads();
    // ---- logits -> exponent-ramp numerators, Pm[key pair][query] (f16 pairs)
    {
      const int t = tid & 63, kq = tid >> 6;
      uint32_t qv[16];
#pragma unroll
      for (int i = 0; i < 16; i++) qv[i] = qn[i * 64 + t];
      const _Float16* braw = (const _Float16*)(rec + R_BIAS) + h * 4096;
      uint32_t num[8];
#pragma unroll
      for (int jj = 0; jj < 8; jj++) {
        float s2[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
          int s = kq * 16 + jj * 2 + e;
          float acc = 0.f;
#pragma unroll
          for (int i = 0; i < 16; i++) acc = dot2(qv[i], kn[i * 64 + s], acc);
          int ii = t & 15, j = s & 15;
          int lane = 4 * (ii & 7) + ((j & 7) >> 1);
          int slot = (j & 1) | ((ii >> 3) << 1) | ((j >> 3) << 2);
          float b = (float)braw[(4 * (t >> 4) + (s >> 4)) * 256 + lane * 8 + slot];
          s2[e] = fmaf(acc, 0.044921875f, fmaf(b, 0.044921875f, 1.30078125f));
        }
        h2 r = (h2){(_Float16)s2[0], (_Float16)s2[1]};
        r = __builtin_elementwise_max(r, (h2){(_Float16)1.03125f, (_Float16)1.03125f});
        r = __builtin_elementwise_min(r, (h2){(_Float16)1.5693359375f, (_Float16)1.5693359375f});
        num[jj] = (__builtin_bit_cast(uint32_t, r) & 0x03FF03FFu) << 5;
      }
      __syncthreads();   // SMALL: Pm replaces kn
#pragma unroll
      for (int jj = 0; jj < 8; jj++) Pm[(kq * 8 + jj) * 64 + t] = num[jj];
    }
    __syncthreads();
    // ---- denominator in the reference order (as native-swin32), one thread per query
    if (tid < 64) {
      h2 pp[32];
#pragma unroll
      for (int j = 0; j < 32; j++) pp[j] = __builtin_bit_cast(h2, Pm[j * 64 + tid]);
      h2 hm[2];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        h2 pr[4];
#pragma unroll
        for (int p = 0; p < 4; p++) { int b = 2 * p + m; pr[p] = ((pp[b] + pp[8 + b]) + pp[16 + b]) + pp[24 + b]; }
        hm[m] = ((pr[0] + pr[1]) + pr[2]) + pr[3];
      }
      _Float16 den = (hm[0].x + hm[0].y) + (hm[1].x + hm[1].y);
      rec_s[tid] = __fdiv_rn(1.f, fmaxf((float)den, 0.00006103515625f));
    }
    __syncthreads();
    // Normalise each P pair once for all four output-channel groups.
    // Pm holds 32 * 64 pairs; each thread owns eight disjoint pairs.
    // Denominator readers finished at the preceding workgroup barrier.
    {
      const int t = tid & 63, group = tid >> 6;
      const float rc = rec_s[t];
#pragma unroll
      for (int n = 0; n < 8; n++) {
        const int j = group * 8 + n;
        const uint32_t u = Pm[j * 64 + t];
        Pm[j * 64 + t] = pk(q8(f16r(__fmul_rn(lo(u), rc))), q8(f16r(__fmul_rn(hi(u), rc))));
      }
    }
    __syncthreads(); // all channel groups consume all 32 normalised pairs
    // ---- context = P V, q8(f16()) -> R3 ctx [16 kp][64 tok]
    {
      const int t = tid & 63, dq = tid >> 6;
      float s[8] = {};
#pragma unroll 8
      for (int j = 0; j < 32; j++) {
        uint32_t pv = Pm[j * 64 + t];
#pragma unroll
        for (int d = 0; d < 8; d++) s[d] = dot2(pv, ((const uint32_t*)Vt)[(dq * 8 + d) * 32 + j], s[d]);
      }
      __syncthreads();
#pragma unroll
      for (int d = 0; d < 8; d += 2) ctx[((dq * 8 + d) >> 1) * 64 + t] = pk(q8(f16r(s[d])), q8(f16r(s[d + 1])));
    }
    // ---- out_proj partial: acco += ctx_h . Wo[:, h*32 .. +32]^T, Wo staged into R1 [16 kp][C] (GN = C/16)
    stage(wo, C, rec, C / 16, [&](int i, int& off, int& ncol, int& kpo) {
      off = R_OUT + (h * (C / 16) + i) * 512; ncol = i * 16; kpo = 0; });
    __syncthreads();
#pragma unroll KPU
    for (int kp = 0; kp < 16; kp++) {
      uint4 a0 = *(const uint4*)&ctx[kp * 64 + ty * 8];
      uint4 a1 = *(const uint4*)&ctx[kp * 64 + ty * 8 + 4];
      uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++) {
        uint32_t bv[CPT];
        ldw<CPT>(&wo[kp * C + chan(pass, 0)], bv);
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
          for (int c = 0; c < CPT; c++) acco[pass][r][c] = dot2(av[r], bv[c], acco[pass][r][c]);
      }
    }
  }

  // ---- out = ars * skip + Wo . ctx ; wide = f16(out) ; store q8(wide) as e4m3 in the arena layout
  {
    const _Float16* ars = (const _Float16*)(rec + R_ARS);
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int t = ty * 8 + r;
      int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
      bool in = row >= 0 && row < H && col >= 0 && col < W;
      int p = (row & 3) * 4 + (col & 3);
      uint8_t* dst = P.out + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
        for (int c = 0; c < CPT; c++) {
          int ch = chan(pass, c);
          uint32_t sk = Xs[(ch >> 1) * 64 + t];
          float o = __fadd_rn(__fmul_rn((ch & 1) ? hi(sk) : lo(sk), (float)ars[ch]), acco[pass][r][c] * 256.f);
          acco[pass][r][c] = o;
          if (!in) continue;
          int mc = ipi32(ch & 31);
          uint8_t v8 = (uint8_t)to_e4m3(q8(f16r(o)));
          if (P.flags & 2)
            P.out[(size_t)((ch >> 5) * 2 + (mc >> 4)) * H * W * 16 + (size_t)(row * W + col) * 16 + (mc & 15)] = v8;
          else
            dst[(ch >> 5) * 512 + 16 * ((mc & 15) >> 2) + 8 * (mc >> 4) + (mc & 3)] = v8;
        }
    }
  }
  if (!(P.flags & 4)) return;

  // ---- downsample: pooled = q8(pool2x2(out)), next = q8(f16(Wds . pooled)) -> +0x38, planes [2C/16][H/2][W/2][16].
  //      A thread's 8 tokens are 2 rows x 4 columns = 2 pooled pixels.
  __syncthreads();   // R1 (Wo) and Xs (skip) are free from here on
  {
    const int py = (ty >> 2) * 2 + (ty & 1), qx = (ty >> 1) & 1;
#pragma unroll
    for (int px = 0; px < 2; px++)
#pragma unroll
      for (int pass = 0; pass < NPASS; pass++)
#pragma unroll
        for (int c = 0; c < CPT; c += 2) {
          float v[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            _Float16 h0 = (_Float16)(acco[pass][2 * px][c + e] + acco[pass][2 * px + 1][c + e]);
            _Float16 h1 = (_Float16)(acco[pass][4 + 2 * px][c + e] + acco[pass][4 + 2 * px + 1][c + e]);
            _Float16 tot = h0 + h1;
            v[e] = q8((float)tot * 0.25f);
          }
          R1[(chan(pass, c) >> 1) * 16 + py * 4 + qx * 2 + px] = pk(v[0], v[1]);
        }
  }
  {
    constexpr int CP2 = C / 32;   // 2C outputs over 64 threads
    const int tx6 = tid & 63, ty6 = tid >> 6;
    float acc[4][CP2] = {};
#pragma unroll 1
    for (int kb = 0; kb < KB; kb++) {
      __syncthreads();
      stage(Xs, 2 * C, rec, C / 8, [&](int i, int& off, int& ncol, int& kpo) {
        off = R_DSRS + (kb * (C / 8) + i) * 512; ncol = i * 16; kpo = 0; });
      __syncthreads();
#pragma unroll KPU
      for (int kp = 0; kp < 16; kp++) {
        uint4 a = *(const uint4*)&R1[(kb * 16 + kp) * 16 + ty6 * 4];
        uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[CP2];
        ldw<CP2>(&Xs[kp * 2 * C + tx6 * CP2], bv);
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
          for (int c = 0; c < CP2; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
      }
    }
    const int H2 = H >> 1, W2 = W >> 1;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int pix = ty6 * 4 + r, pr = (r0 >> 1) + (pix >> 2), pc = (c0 >> 1) + (pix & 3);
      if (pr < 0 || pr >= H2 || pc < 0 || pc >= W2) continue;
#pragma unroll
      for (int c = 0; c < CP2; c++) {
        int ch = tx6 * CP2 + c, mc = ipi32(ch & 31);
        P.p38[(size_t)((ch >> 5) * 2 + (mc >> 4)) * H2 * W2 * 16 + (size_t)(pr * W2 + pc) * 16 + (mc & 15)] =
            (uint8_t)to_e4m3(q8(f16r(acc[r][c] * 256.f)));
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(256) void k_swin_c256(VarParams P) { swin_block<256>(P); }
extern "C" __global__ __launch_bounds__(256) void k_swin_c128(VarParams P) { swin_block<128>(P); }
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8))) void k_swin_c64(VarParams P) { swin_block<64>(P); }
