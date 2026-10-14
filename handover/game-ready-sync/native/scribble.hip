#include <hip/hip_runtime.h>
// fills all 64 KB of LDS with a seed-dependent pattern (run with many workgroups so every CU gets it)
extern "C" __global__ __launch_bounds__(256) void scribble(unsigned seed) {
  __shared__ unsigned L[16384];
  for (int i = threadIdx.x; i < 16384; i += 256) L[i] = (seed * 2654435761u) ^ (i * 40503u) ^ blockIdx.x;
  __syncthreads();
  if (L[(threadIdx.x * 7) & 16383] == 0x12345678u) L[0] = 1;   // keep the stores
}
