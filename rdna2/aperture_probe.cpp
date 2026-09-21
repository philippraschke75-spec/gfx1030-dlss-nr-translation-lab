#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
__global__ void k(uint64_t* o){
  __shared__ int lds[4]; uint64_t sh, pr;
  asm volatile("s_mov_b64 %0, src_shared_base" : "=s"(sh));
  asm volatile("s_mov_b64 %0, src_private_base" : "=s"(pr));
  int priv[4]; priv[threadIdx.x&3]=1; asm volatile("" :: "r"(priv[0]));
  lds[threadIdx.x&3]=1; __syncthreads();
  if(threadIdx.x==0){ o[0]=sh; o[1]=pr; o[2]=(uint64_t)(void*)&lds[0]; o[3]=(uint64_t)(void*)&priv[0]; }
}
int main(){ uint64_t* d; hipMalloc(&d,32); hipMemset(d,0,32); k<<<1,32>>>(d); hipDeviceSynchronize(); uint64_t h[4]; hipMemcpy(h,d,32,hipMemcpyDeviceToHost);
 printf("src_shared_base  = %016llx\nsrc_private_base = %016llx\ngeneric &lds     = %016llx\ngeneric &private = %016llx\n",(unsigned long long)h[0],(unsigned long long)h[1],(unsigned long long)h[2],(unsigned long long)h[3]); }
