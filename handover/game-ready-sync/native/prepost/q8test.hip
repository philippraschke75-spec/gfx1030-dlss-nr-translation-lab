// Exhaustive check of the cheaper e4m3 rounders against the reference q8: q8f over all 2^32 f32 inputs, q8h over
// all 2^16 f16 inputs (as f16 bits). Build: clang++ -x hip --offload-arch=gfx1030 -O3 -ffp-contract=off q8test.hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <stdint.h>
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float q8ref(float x) {
  float a = fminf(fabsf(x), 448.f);
  uint32_t b = __float_as_uint(a);
  float rn = __uint_as_float((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
  float rs = __fsub_rn(__fadd_rn(a, 24576.f), 24576.f);
  return copysignf(a < 0.015625f ? rs : rn, x);
}
__device__ __forceinline__ float q8f(float x) {
  float a = fminf(fabsf(x), 448.f);
  float e = __uint_as_float(__float_as_uint(a) & 0xFF800000u);
  float K = __fmul_rn(fmaxf(e, 0.015625f), 1048576.f);
  return copysignf(__fsub_rn(__fadd_rn(a, K), K), x);
}
__device__ __forceinline__ h2 q8h(h2 x) {
  uint32_t xb = __builtin_bit_cast(uint32_t, x);
  // |x| clamped at 448 and 2^exp floored at 2^-6 as unsigned 16-bit compares (same order as f16 for non-negative
  // values; NaN/inf sort above 448 like fminf), which avoids the canonicalising v_pk_max the f16 min/max emit
  typedef unsigned short u2 __attribute__((ext_vector_type(2)));
  u2 ab = __builtin_elementwise_min(__builtin_bit_cast(u2, xb & 0x7FFF7FFFu), (u2){0x5F00, 0x5F00});
  u2 eb = __builtin_elementwise_max(ab & (u2){0x7C00, 0x7C00}, (u2){0x2400, 0x2400});
  h2 a = __builtin_bit_cast(h2, ab);
  h2 K = __builtin_bit_cast(h2, eb) * (h2){(_Float16)128.f, (_Float16)128.f};
  h2 r = (a + K) - K;
  return __builtin_bit_cast(h2, (__builtin_bit_cast(uint32_t, r) & 0x7FFF7FFFu) | (xb & 0x80008000u));
}
__global__ void kf(unsigned* bad, unsigned* first) {
  uint32_t i = blockIdx.x * 256 + threadIdx.x;
  for (uint32_t k = 0; k < 16; k++) {
    uint32_t x = i * 16 + k;
    uint32_t r = __float_as_uint(q8ref(__uint_as_float(x))), c = __float_as_uint(q8f(__uint_as_float(x)));
    if (r != c) { if (atomicAdd(bad, 1) == 0) { first[0] = x; first[1] = r; first[2] = c; } }
  }
}
__global__ void kh(unsigned* bad, unsigned* first) {
  uint32_t x = blockIdx.x * 256 + threadIdx.x;   // f16 bits
  _Float16 h = __builtin_bit_cast(_Float16, (uint16_t)x);
  _Float16 r = (_Float16)q8ref((float)h);
  h2 c2 = q8h((h2){h, h});
  uint16_t rb = __builtin_bit_cast(uint16_t, r), cb = __builtin_bit_cast(uint16_t, c2.x), cb2 = __builtin_bit_cast(uint16_t, c2.y);
  if (rb != cb || rb != cb2) { if (atomicAdd(bad, 1) == 0) { first[0] = x; first[1] = rb; first[2] = cb; } }
}
int main() {
  unsigned *bad, *first; hipMalloc(&bad, 8); hipMalloc(&first, 24); hipMemset(bad, 0, 8); hipMemset(first, 0, 24);
  kf<<<(1u << 28) / 256, 256>>>(bad, first);
  kh<<<65536 / 256, 256>>>(bad + 1, first + 3);
  hipDeviceSynchronize();
  unsigned hb[2], hf[6]; hipMemcpy(hb, bad, 8, hipMemcpyDeviceToHost); hipMemcpy(hf, first, 24, hipMemcpyDeviceToHost);
  printf("q8f  mismatches over 2^32 f32: %u (first x=%08x ref=%08x got=%08x)\n", hb[0], hf[0], hf[1], hf[2]);
  printf("q8h  mismatches over 2^16 f16: %u (first x=%04x ref=%04x got=%04x)\n", hb[1], hf[3], hf[4], hf[5]);
  return hb[0] || hb[1];
}
