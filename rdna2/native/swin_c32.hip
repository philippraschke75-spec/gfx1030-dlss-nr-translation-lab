// Native gfx1030 single-head Swin block (Swin1H, dlssnr/layers/swin.py) for k_swin_var<32>: drop-in for
// _Z10k_swin_varILi32ELb0EEv9VarParams (same VarParams kernarg, grid = one 8x8 window per workgroup, 256 threads).
// Structure follows native-mid/swinvar.hip (its C = 64 "SMALL" LDS plan, 24 KB, two workgroups per CU); the C = 32
// differences of the single-head block:
//  - no grouped mid projection: expand 32 -> 128 (+SiLU, q8), contract 128 -> 32, in two hidden chunks of 64;
//  - the attention skip is the unrounded f32 MLP output y (qkv reads q8(f16(y)));
//  - upsample (flags 8): blend = f16(skip * gain + P(coarse)) is the MLP residual, q8(blend) the matmul input;
//    gain = upsample_gain[0..31];
//  - record (extract.py _swin_layout, heads = 1): expand 4096 B | contract 4096 B | [resample 2048 B if up] |
//    residual_scale 96 B | [upsample_gain 64 B if up] | qkv 3072 B | bias 8192 B | scalars 16 B | out_proj 1024 B |
//    attn_residual_scale | [ds resample 2048 B].
// Build with -ffp-contract=off -mcumode (build_native.sh).
#include "swin_common.h"

#ifndef WPE
#define WPE 8
#endif
extern "C" __global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(WPE))) void k_swin_c32(VarParams P) {
  constexpr int C = 32;
  __shared__ uint32_t Xs[16 * 64];   // [kp][tok]: q8(x), later Yq; weight stage for up/ds projections
  __shared__ uint32_t R1[2176];      // coarse | Hs [32 kp][68] | q f32 | qn | ctx | Wo, reciprocals at +1024
  __shared__ uint32_t R2[1152];      // up: residual blend f16 [16 kp][64] | Vt [32 key pair][36] half2
  __shared__ uint32_t R3[2112];      // weight stage | Pj | k f32 | kn | Pm
  float* rec_s = (float*)(R1 + 1024);

  const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;   // C-wide tiles: tokens ty*8..+7, channel tx
  const uint8_t* rec = P.w;
  const int H = P.H, W = P.W, WT = W >> 2;
  const int r0 = P.Y0 + blockIdx.y * 8, c0 = P.X0 + blockIdx.x * 8;
  const bool up = P.flags & 8;
  constexpr int R_CON = 4 * C * C, R_AFT = R_CON + 4 * C * C;
  const int R_RS = up ? R_AFT + 2 * C * C : R_AFT;
  const int R_QKV = R_RS + 2 * C + 32 + (up ? 2 * C : 0);
  const int R_BIAS = R_QKV + 3 * C * C, R_HS = R_BIAS + 8192, R_OUT = R_HS + 16;
  const int R_ARS = R_OUT + C * C, R_DSRS = R_ARS + 2 * C;
  const _Float16* rs = (const _Float16*)(rec + R_RS);
  const uint4 rs8 = *(const uint4*)rs;
  const int rso = (rs8.x | rs8.y | rs8.z | rs8.w) == 0 ? 8 : 0;
  _Float16* Pj = (_Float16*)R3;

  // ---- upsample: Pj[16 pix][32] = f16(Wr . coarse) for the 4x4 coarse pixels under the window (K = 64)
  if (up) {
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3, pr0 = r0 >> 1, pc0 = c0 >> 1;
    if (tid < 32) {   // (pixel, 32-channel chunk of 64)
      const int px = tid & 15, ch = tid >> 4;
      const int pr = pr0 + (px >> 2), pc = pc0 + (px & 3);
      uint32_t b[8] = {};
      if (pr >= 0 && pr < H2 && pc >= 0 && pc < W2) {
        const uint8_t* src = P.p30 + (size_t)(ch * 2) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)Hs * Ws * 16 + 4 * g); }
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
    // Wr [32 n][64 k]: tiles kb*2 + j (GN = 2) -> Xs [32 kp][32 n]
    stage(Xs, 32, rec, 4, [&](int i, int& off, int& ncol, int& kpo) { off = R_AFT + i * 512; ncol = (i & 1) * 16; kpo = (i >> 1) * 16; });
    __syncthreads();
    const int px = tid & 15, cp = tid >> 4;   // pixel, output channel pair
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 8
    for (int kp = 0; kp < 32; kp++) {
      uint32_t a = R1[kp * 16 + px]; uint2 bb = *(const uint2*)&Xs[kp * 32 + 2 * cp];
      a0 = dot2(a, bb.x, a0); a1 = dot2(a, bb.y, a1);
    }
    Pj[px * C + 2 * cp] = (_Float16)(a0 * 256.f); Pj[px * C + 2 * cp + 1] = (_Float16)(a1 * 256.f);
    __syncthreads();
  }

  // ---- load the window: token t = (qy*2+qx)*16 + y*4 + x
  if (tid < 64) {
    const int t = tid;
    int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
    bool in = row >= 0 && row < H && col >= 0 && col < W;
    uint32_t b[8] = {};
    if (in) {
      int p = (row & 3) * 4 + (col & 3);
      if (P.flags & 1) {
        const uint8_t* src = P.in + (size_t)(row * W + col) * 16;
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 4 * g); b[2 * g + 1] = *(const uint32_t*)(src + (size_t)H * W * 16 + 4 * g); }
      } else {
        const uint8_t* src = P.in + (size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3);
#pragma unroll
        for (int g = 0; g < 4; g++) { b[2 * g] = *(const uint32_t*)(src + 16 * g); b[2 * g + 1] = *(const uint32_t*)(src + 16 * g + 8); }
      }
    }
    const _Float16* ug = (const _Float16*)(rec + R_RS + 2 * C + 32);
    const int lr = ((t >> 5) & 1) * 4 + ((t >> 2) & 3), lc = ((t >> 4) & 1) * 4 + (t & 3), pix = (lr >> 1) * 4 + (lc >> 1);
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int s = 0; s < 2; s++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          int c = 16 * s + 4 * g + 2 * hh;
          uint32_t v = x256(e4x2((b[2 * g + s] >> (16 * hh)) & 0xFFFFu));
          const int rc = pi32(c);
          if (up) {   // blend = f16(skip * gain + P(coarse)), zero outside the frame; matmuls read q8(blend)
            float x0 = in ? f16r((float)((double)lo(v) * (double)(float)ug[rc] + (double)(float)Pj[pix * C + rc])) : 0.f;
            float x1 = in ? f16r((float)((double)hi(v) * (double)(float)ug[rc + 1] + (double)(float)Pj[pix * C + rc + 1])) : 0.f;
            R2[(rc >> 1) * 64 + t] = pk(x0, x1);
            v = pk(q8(x0), q8(x1));
          }
          Xs[(rc >> 1) * 64 + t] = v;
        }
  }
  const uint32_t* Xres = up ? R2 : Xs;   // MLP residual

  // ================= MLP: two hidden chunks of 64; contract partials accumulate in registers
  float y[8] = {};
#pragma unroll 1
  for (int sc = 0; sc < 2; sc++) {
    {   // expand: Hs[64 tok][64] = q8(act(X . We[sc*64 .. +64]^T)), K = 32 (n-tile = tile)
      __syncthreads();
      if (tid < 128) put16(R3, 64, *((const uint4*)(rec + sc * 4 * 512) + tid), tid & 31, (tid >> 5) * 16);
      __syncthreads();
      float acc[8][2] = {};
#pragma unroll 4
      for (int kp = 0; kp < 16; kp++) {
        uint4 a0 = *(const uint4*)&Xs[kp * 64 + ty * 8];
        uint4 a1 = *(const uint4*)&Xs[kp * 64 + ty * 8 + 4];
        uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        uint2 bb = *(const uint2*)&R3[kp * 64 + tx * 2];
#pragma unroll
        for (int r = 0; r < 8; r++) { acc[r][0] = dot2(av[r], bb.x, acc[r][0]); acc[r][1] = dot2(av[r], bb.y, acc[r][1]); }
      }
#pragma unroll
      for (int r = 0; r < 8; r++) R1[tx * 68 + ty * 8 + r] = pk(q8(act(acc[r][0] * 256.f)), q8(act(acc[r][1] * 256.f)));
    }
    __syncthreads();
    // contract partial: y += Hs . Wc[:, sc*64 .. +64]^T; Wc [32 n][128 k] tiles kb*2 + j (GN = 2), kb = 2sc, 2sc+1 -> R3 [32 kp][32 n]
    if (tid < 128) { const int i = tid >> 5; put16(R3 + (i >> 1) * 16 * 32, 32, *((const uint4*)(rec + R_CON + (sc * 4 + i) * 512) + (tid & 31)), tid & 31, (i & 1) * 16); }
    __syncthreads();
#pragma unroll 8
    for (int kp = 0; kp < 32; kp++) {
      uint4 a0 = *(const uint4*)&R1[kp * 68 + ty * 8];
      uint4 a1 = *(const uint4*)&R1[kp * 68 + ty * 8 + 4];
      uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, b = R3[kp * 32 + tx];
#pragma unroll
      for (int r = 0; r < 8; r++) y[r] = dot2(av[r], b, y[r]);
    }
  }

  // ---- y = rs * x + mlp (f32, the attention skip); Yq = q8(f16(y)) replaces x in Xs
  {
    const float rsc = (float)rs[rso + tx];
    const _Float16* xr = (const _Float16*)Xres;
#pragma unroll
    for (int r = 0; r < 8; r++) y[r] = __fadd_rn(__fmul_rn((float)xr[((tx >> 1) * 64 + ty * 8 + r) * 2 + (tx & 1)], rsc), y[r] * 256.f);
    __syncthreads();
    _Float16* xq = (_Float16*)Xs;
#pragma unroll
    for (int r = 0; r < 8; r++) xq[((tx >> 1) * 64 + ty * 8 + r) * 2 + (tx & 1)] = (_Float16)q8(f16r(y[r]));
  }

  // ================= attention (one head)
  float* qb = (float*)R1;
  float* kbuf = (float*)R3;
  uint32_t* qn = R1;
  uint32_t* kn = R3;
  uint32_t* Pm = R3;
  uint32_t* ctx = R1;
  uint32_t* wo = R1 + 1024;
  _Float16* Vt = (_Float16*)R2;
  const float* hsc = (const float*)(rec + R_HS);
  {   // qkv: [64 tok][96] = Yq . Wqkv^T, K = 32
    const int tx16 = tid >> 4, ty16 = tid & 15;   // tokens across lanes: conflict-free stores
    float acc[4][6] = {};
    __syncthreads();
    if (tid < 192) put16(R3, 96, *((const uint4*)(rec + R_QKV) + tid), tid & 31, (tid >> 5) * 16);
    __syncthreads();
#pragma unroll 4
    for (int kp = 0; kp < 16; kp++) {
      uint4 a = *(const uint4*)&Xs[kp * 64 + ty16 * 4];
      uint2 b0 = *(const uint2*)&R3[kp * 96 + tx16 * 6];
      uint2 b1 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 2];
      uint2 b2 = *(const uint2*)&R3[kp * 96 + tx16 * 6 + 4];
      uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[6] = {b0.x, b0.y, b1.x, b1.y, b2.x, b2.y};
#pragma unroll
      for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < 6; c++) acc[r][c] = dot2(av[r], bv[c], acc[r][c]);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
      for (int c = 0; c < 6; c++) {
        int n = tx16 * 6 + c, t = ty16 * 4 + r;
        float v = acc[r][c] * 256.f;
        if (n < 32) qb[n * 65 + t] = v;   // row stride 65: no bank conflicts
        else if (n < 64) kbuf[(n - 32) * 65 + t] = v;
        else Vt[((t >> 1) * 36 + n - 64) * 2 + (t & 1)] = (_Float16)q8(f16r(v));   // [key pair][d] half2
      }
  }
  __syncthreads();
  {   // L2-normalise q (with the head scale) and k. qn/kn alias qb/kbuf across waves: read all, barrier, then write.
    const int t = tid & 63, w = (tid >> 6) & 1;
    float v[32], sc = 0.f;
    if (tid < 128) {
#pragma unroll
      for (int d = 0; d < 32; d++) v[d] = (w ? kbuf : qb)[d * 65 + t];
      sc = norm_scale(v);
      if (w == 0) sc = __fmul_rn(sc, hsc[0]);
    }
    __syncthreads();
    if (tid < 128) {
#pragma unroll
      for (int d = 0; d < 32; d += 2) (w ? kn : qn)[(d >> 1) * 64 + t] = pk(q8(__fmul_rn(v[d], sc)), q8(__fmul_rn(v[d + 1], sc)));
    }
  }
  __syncthreads();
  {   // logits -> exponent-ramp numerators, Pm[key pair][query]
    const int t = tid & 63, kq = tid >> 6;
    uint32_t qv[16];
#pragma unroll
    for (int i = 0; i < 16; i++) qv[i] = qn[i * 64 + t];
    const _Float16* braw = (const _Float16*)(rec + R_BIAS);
    uint32_t num[8];
    float lg[16] = {};   // this thread's 16 keys; 4 keys per LDS read
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int i = 0; i < 16; i++) {
        uint4 k4 = *(const uint4*)&kn[i * 64 + kq * 16 + g * 4];
        lg[4 * g] = dot2(qv[i], k4.x, lg[4 * g]); lg[4 * g + 1] = dot2(qv[i], k4.y, lg[4 * g + 1]);
        lg[4 * g + 2] = dot2(qv[i], k4.z, lg[4 * g + 2]); lg[4 * g + 3] = dot2(qv[i], k4.w, lg[4 * g + 3]);
      }
#pragma unroll
    for (int jj = 0; jj < 8; jj++) {
      float s2[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        int s = kq * 16 + jj * 2 + e;
        float acc = lg[jj * 2 + e];
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
    __syncthreads();   // Pm replaces kn
#pragma unroll
    for (int jj = 0; jj < 8; jj++) Pm[(kq * 8 + jj) * 64 + t] = num[jj];
  }
  __syncthreads();
  if (tid < 64) {   // denominator in the reference order
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
  {   // context = P V -> ctx [16 kp][64 tok]
    const int t = tid & 63, dq = tid >> 6;
    const float rc = rec_s[t];
#pragma unroll
    for (int j = dq * 8; j < dq * 8 + 8; j++) {   // probabilities, each quantised once (in place)
      uint32_t u = Pm[j * 64 + t];
      Pm[j * 64 + t] = pk(q8(f16r(__fmul_rn(lo(u), rc))), q8(f16r(__fmul_rn(hi(u), rc))));
    }
    __syncthreads();
    float s[8] = {};
#pragma unroll 8
    for (int j = 0; j < 32; j++) {
      const uint32_t pv = Pm[j * 64 + t];
      const uint4 v0 = *(const uint4*)((const uint32_t*)Vt + j * 36 + dq * 8), v1 = *(const uint4*)((const uint32_t*)Vt + j * 36 + dq * 8 + 4);
      const uint32_t vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int d = 0; d < 8; d++) s[d] = dot2(pv, vv[d], s[d]);
    }
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 8; d += 2) ctx[((dq * 8 + d) >> 1) * 64 + t] = pk(q8(f16r(s[d])), q8(f16r(s[d + 1])));
  }
  stage(wo, C, rec, 2, [&](int i, int& off, int& ncol, int& kpo) { off = R_OUT + i * 512; ncol = i * 16; kpo = 0; });
  __syncthreads();
  float o[8] = {};
#pragma unroll 4
  for (int kp = 0; kp < 16; kp++) {
    uint4 a0 = *(const uint4*)&ctx[kp * 64 + ty * 8];
    uint4 a1 = *(const uint4*)&ctx[kp * 64 + ty * 8 + 4];
    uint32_t av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, b = wo[kp * C + tx];
#pragma unroll
    for (int r = 0; r < 8; r++) o[r] = dot2(av[r], b, o[r]);
  }

  // ---- out = ars * y + Wo . ctx ; store q8(f16(out)) as e4m3
  {
    const float as = (float)((const _Float16*)(rec + R_ARS))[tx];
    const int mc = ipi32(tx);
#pragma unroll
    for (int r = 0; r < 8; r++) {
      int t = ty * 8 + r;
      int row = r0 + ((t >> 5) & 1) * 4 + ((t >> 2) & 3), col = c0 + ((t >> 4) & 1) * 4 + (t & 3);
      o[r] = __fadd_rn(__fmul_rn(y[r], as), o[r] * 256.f);
      if (!(row >= 0 && row < H && col >= 0 && col < W)) continue;
      int p = (row & 3) * 4 + (col & 3);
      uint8_t v8 = (uint8_t)to_e4m3(q8(f16r(o[r])));
      if (P.flags & 2)
        P.out[(size_t)(mc >> 4) * H * W * 16 + (size_t)(row * W + col) * 16 + (mc & 15)] = v8;
      else
        P.out[(size_t)((row >> 2) * WT + (col >> 2)) * (C * 16) + (p & 7) * 64 + 4 * (p >> 3) + 16 * ((mc & 15) >> 2) + 8 * (mc >> 4) + (mc & 3)] = v8;
    }
  }
  if (!(P.flags & 4)) return;

  // ---- downsample: pooled = q8(pool2x2(out)) -> R1 f16 [16 kp][16 pix]; next = q8(f16(Wds . pooled)) -> +0x38 planes [4][H/2][W/2][16]
  __syncthreads();
  {
    const int py = (ty >> 2) * 2 + (ty & 1), qx = (ty >> 1) & 1;
    _Float16* pl = (_Float16*)R1;
#pragma unroll
    for (int px = 0; px < 2; px++) {
      _Float16 h0 = (_Float16)(o[2 * px] + o[2 * px + 1]);
      _Float16 h1 = (_Float16)(o[4 + 2 * px] + o[4 + 2 * px + 1]);
      _Float16 tot = h0 + h1;
      pl[((tx >> 1) * 16 + py * 4 + qx * 2 + px) * 2 + (tx & 1)] = (_Float16)q8((float)tot * 0.25f);
    }
  }
  {
    const int tx6 = tid & 63, ty6 = tid >> 6;
    float acc[4] = {};
    stage(Xs, 2 * C, rec, 4, [&](int i, int& off, int& ncol, int& kpo) { off = R_DSRS + i * 512; ncol = i * 16; kpo = 0; });
    __syncthreads();
#pragma unroll 4
    for (int kp = 0; kp < 16; kp++) {
      uint4 a = *(const uint4*)&R1[kp * 16 + ty6 * 4];
      uint32_t av[4] = {a.x, a.y, a.z, a.w}, b = Xs[kp * 2 * C + tx6];
#pragma unroll
      for (int r = 0; r < 4; r++) acc[r] = dot2(av[r], b, acc[r]);
    }
    const int H2 = H >> 1, W2 = W >> 1, Hs = (H2 + 3) & ~3, Ws = (W2 + 3) & ~3;
#pragma unroll
    for (int r = 0; r < 4; r++) {
      int pix = ty6 * 4 + r, pr = (r0 >> 1) + (pix >> 2), pc = (c0 >> 1) + (pix & 3);
      if (pr < 0 || pr >= H2 || pc < 0 || pc >= W2) continue;
      int ch = tx6, mc = ipi32(ch & 31);
      P.p38[(size_t)((ch >> 5) * 2 + (mc >> 4)) * Hs * Ws * 16 + (size_t)(pr * Ws + pc) * 16 + (mc & 15)] = (uint8_t)to_e4m3(q8(f16r(acc[r] * 256.f)));
    }
  }
}
