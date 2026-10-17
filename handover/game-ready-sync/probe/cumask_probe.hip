// cumask_probe: does hipExtStreamCreateWithCUMask restrict where workgroups run on this driver (Windows/PAL)?
// For each single-bit mask: launch blocks, record HW_ID1 (gfx10: SIMD 9:8, WGP 13:10, SA 16, SE 19:18). Then time a busy kernel at 100/50/20/10 %.
// build: hipcc --offload-arch=gfx1030 -O2 cumask_probe.hip -o cumask_probe.exe   run: cumask_probe [nbits]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include <chrono>
__global__ void hwid(unsigned* out, int iters) {
  float x = threadIdx.x; for (int i = 0; i < iters; i++) x = x * 0.9999f + 0.5f;
  if (threadIdx.x == 0) out[blockIdx.x] = __builtin_amdgcn_s_getreg((31 << 11) | 23) | (x == 12345.f);
}
static unsigned cuOf(unsigned v) { return ((v >> 18) & 3) << 8 | ((v >> 16) & 1) << 7 | ((v >> 10) & 15) << 1 | ((v >> 9) & 1); }   // se | sa | wgp | cu-in-wgp (SIMD 2,3 = CU1)
int main(int argc, char** argv) {
  int ncu = 0; hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, 0); printf("multiProcessorCount %d\n", ncu);
  const int NB = 256; unsigned* d; hipMalloc(&d, NB * 4); std::vector<unsigned> h(NB), m((ncu + 31) / 32);
  const int nbits = argc > 1 ? atoi(argv[1]) : ncu;
  for (int b = 0; b < nbits; b++) {
    for (auto& w : m) w = 0; m[b / 32] |= 1u << (b % 32);
    hipStream_t s; hipError_t e = hipExtStreamCreateWithCUMask(&s, (unsigned)m.size(), m.data());
    if (e != hipSuccess) { printf("bit %d: hipExtStreamCreateWithCUMask -> %s\n", b, hipGetErrorString(e)); return 1; }
    hipMemsetAsync(d, 0xff, NB * 4, s); hipLaunchKernelGGL(hwid, NB, 64, 0, s, d, 2000); hipMemcpyAsync(h.data(), d, NB * 4, hipMemcpyDeviceToHost, s); hipStreamSynchronize(s);
    std::set<unsigned> cus; for (unsigned v : h) cus.insert(cuOf(v));
    printf("bit %2d ->", b); int n = 0; for (unsigned c : cus) if (n++ < 6) printf(" se%u.sa%u.wgp%u.cu%u", c >> 8, (c >> 7) & 1, (c >> 1) & 15, c & 1); printf("  (%zu CUs)\n", cus.size());
    hipStreamDestroy(s);
  }
  unsigned* big; hipMalloc(&big, 4096 * 4); std::vector<unsigned> hb(4096);
  for (int pct : {100, 50, 20, 10}) {
    const int k = ncu * pct / 100; for (auto& w : m) w = 0; for (int i = 0; i < k; i++) m[i / 32] |= 1u << (i % 32);
    hipStream_t s; hipExtStreamCreateWithCUMask(&s, (unsigned)m.size(), m.data());
    hipLaunchKernelGGL(hwid, 4096, 64, 0, s, big, 20000); hipStreamSynchronize(s);
    auto t = std::chrono::steady_clock::now(); hipLaunchKernelGGL(hwid, 4096, 64, 0, s, big, 20000); hipStreamSynchronize(s);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    hipMemcpy(hb.data(), big, 4096 * 4, hipMemcpyDeviceToHost); std::set<unsigned> cus, se; for (unsigned v : hb) { cus.insert(cuOf(v)); se.insert(cuOf(v) >> 7); }
    printf("mask %3d%% (%2d low bits): %.2f ms, %zu distinct CUs over %zu SE/SA\n", pct, k, ms, cus.size(), se.size()); hipStreamDestroy(s);
  }
  return 0;
}
