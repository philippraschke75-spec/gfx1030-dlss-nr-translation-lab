// k_mirror_pad: reflect the imported image into its padding (MirrorParams, nr_params.h). One thread per padded pixel;
// sources are always inside the valid rectangle, so no thread reads a pixel another writes.
#include <hip/hip_runtime.h>
#include "nr_params.h"

extern "C" __global__ __launch_bounds__(256) void k_mirror_pad(MirrorParams p) {
  const int x = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y;
  if (x >= p.pad_w || y >= p.pad_h || (x < p.w_ && y < p.h)) return;
  int sx = x < p.w_ ? x : 2 * p.w_ - x - 2, sy = y < p.h ? y : 2 * p.h - y - 2;
  sx = sx < 0 ? 0 : sx; sy = sy < 0 ? 0 : sy;   // padding wider than the image (tiny frames): clamp
  const float* s = p.rgb + ((size_t)sy * p.pad_w + sx) * 3; float* d = p.rgb + ((size_t)y * p.pad_w + x) * 3;
  d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
}
