#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

// Reproduce the store operation seen in GFX11 k_align_probe, assembled by
// the compiler for GFX1030. Inline assembly avoids a misaligned C++ pointer.
__global__ void k_align_probe(unsigned char* p) {
    if (threadIdx.x == 0) {
        unsigned value = 0x2211;
        asm volatile("global_store_short %0, %1, off offset:1" :: "v"(p), "v"(value) : "memory");
    }
}
__global__ void probe_pair(unsigned char* p) {
    if (threadIdx.x == 0) {
        unsigned value = 0x2211;
        asm volatile("global_store_short %0, %1, off offset:0" :: "v"(p), "v"(value) : "memory");
        asm volatile("global_store_short %0, %1, off offset:9" :: "v"(p), "v"(value) : "memory");
    }
}
#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess) { \
    std::printf("%s: %s\n", #x, hipGetErrorString(e)); return 2; } } while(0)
int main() {
    hipDeviceProp_t prop{};
    CHECK(hipGetDeviceProperties(&prop, 0));
    if (std::strcmp(prop.gcnArchName,"gfx1030") != 0 || prop.warpSize != 32) return 4;
    std::printf("Device=%s arch=%s\n", prop.name, prop.gcnArchName);
    unsigned char got[32];
    std::memset(got, 0xa5, sizeof(got));
    unsigned char* d=nullptr;
    CHECK(hipMalloc(&d, sizeof(got)));
    CHECK(hipMemcpy(d, got, sizeof(got), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe_pair, dim3(1), dim3(32), 0, 0, d);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got, d, sizeof(got), hipMemcpyDeviceToHost));
    CHECK(hipFree(d));
    std::printf("Aligned +0:   %02x %02x %02x %02x\n",got[0],got[1],got[2],got[3]);
    std::printf("Unaligned +9: %02x %02x %02x %02x\n",got[8],got[9],got[10],got[11]);
    for(int i=0;i<32;++i) {
        if(i==0 || i==1 || (i>=8 && i<=10)) continue;
        if(got[i]!=0xa5) { std::puts("Unexpected out-of-region write"); return 3; }
    }
    if(got[0]!=0x11 || got[1]!=0x22) { std::puts("Aligned store FAILED"); return 3; }
    if(got[8]==0xa5 && got[9]==0x11 && got[10]==0x22)
        std::puts("Unaligned store preserves byte address");
    else if(got[8]==0x11 && got[9]==0x22 && got[10]==0xa5)
        std::puts("Unaligned store rounds address down to 2-byte boundary");
    else { std::puts("Unaligned result unclassified"); return 3; }
    std::puts("STORE PROBE COMPLETE; this is not network output");
    return 0;
}
