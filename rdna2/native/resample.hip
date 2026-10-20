// k_resample: bilinear resample of a planar e4m3 activation (ResampleParams, nr_params.h) for the hybrid-resolution
// plan. One thread per destination pixel and plane (16 channels); channel interleave does not matter (same on both sides).
// A 2x downsample samples at the centre of each 2x2 block, i.e. the 2x2 average.
#include "nk_common.h"

extern "C" __global__ __launch_bounds__(256) void k_resample(ResampleParams p) {
  const int x = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y, pl = blockIdx.z;
  if (x >= p.dw || y >= p.dh) return;
  const float fx = fminf(fmaxf((x + 0.5f) * p.sx - 0.5f, 0.f), (float)(p.sw - 1));
  const float fy = fminf(fmaxf((y + 0.5f) * p.sy - 0.5f, 0.f), (float)(p.sh - 1));
  const int x0 = (int)fx, y0 = (int)fy, x1 = min(x0 + 1, p.sw - 1), y1 = min(y0 + 1, p.sh - 1);
  const float ax = fx - x0, ay = fy - y0;
  const uint8_t* s = p.src + (size_t)pl * p.sh * p.sw * 16;
  const uint4 q[4] = {*(const uint4*)(s + ((size_t)y0 * p.sw + x0) * 16), *(const uint4*)(s + ((size_t)y0 * p.sw + x1) * 16),
                      *(const uint4*)(s + ((size_t)y1 * p.sw + x0) * 16), *(const uint4*)(s + ((size_t)y1 * p.sw + x1) * 16)};
  const float wt[4] = {(1.f - ax) * (1.f - ay), ax * (1.f - ay), (1.f - ax) * ay, ax * ay};
  uint32_t o[4];
#pragma unroll
  for (int d = 0; d < 4; d++) {
    uint32_t r = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) {
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < 4; k++) v += wt[k] * e4f((&q[k].x)[d] >> (8 * b));
      r |= e4m3_byte(v) << (8 * b);
    }
    o[d] = r;
  }
  *(uint4*)(p.dst + (size_t)pl * p.dh * p.dw * 16 + ((size_t)y * p.dw + x) * 16) = make_uint4(o[0], o[1], o[2], o[3]);
}
