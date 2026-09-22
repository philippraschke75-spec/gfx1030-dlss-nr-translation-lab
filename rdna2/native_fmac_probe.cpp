// One wave, one dispatch: exact operands from the flag-32 store trace.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){ std::printf("HIP error %s\n",hipGetErrorString(e)); return 2; } } while(0)
__global__ void probe(uint32_t* out) {
    uint32_t a=1, b=0xb61fbaa2, c=0;
    asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(c) : "v"(a), "v"(b));
    out[threadIdx.x]=c;
}
int main() {
    hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p,0));
    std::printf("GPU=%s arch=%s\n",p.name,p.gcnArchName);
    uint32_t* d=nullptr; CHECK(hipMalloc(&d,32*sizeof(uint32_t)));
    hipLaunchKernelGGL(probe,dim3(1),dim3(32),0,0,d);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    uint32_t h[32]; CHECK(hipMemcpy(h,d,sizeof h,hipMemcpyDeviceToHost));
    for(int i=0;i<32;i++) std::printf("lane=%d result=%08x\n",i,h[i]);
    CHECK(hipFree(d)); return 0;
}
