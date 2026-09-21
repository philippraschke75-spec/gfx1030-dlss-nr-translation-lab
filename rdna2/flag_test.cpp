#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::printf("%s: %s\n",#x,hipGetErrorString(e)); return 2;} } while(0)
struct Args{ void* flag; uint32_t threshold; uint32_t spins; };
// Bounded probe: does the translated k_flag_wait run, spin a bounded count, and store a plausible realtime stamp?
int main(int argc,char**argv){
 hipModule_t m; hipFunction_t f;
 CHECK(hipModuleLoad(&m,argv[1])); CHECK(hipModuleGetFunction(&f,m,"_Z11k_flag_waitPjjj"));
 uint8_t* d; CHECK(hipMalloc(&d,256)); int rc=0;
 for(int mode=0;mode<2;mode++){
  uint8_t h[256]; std::memset(h,0xA5,256); uint32_t* w=(uint32_t*)h; w[0]=mode?7:0; // flag value; +4.. guard
  CHECK(hipMemcpy(d,h,256,hipMemcpyHostToDevice));
  struct { Args a; uint8_t hidden[256]; } k{}; k.a={d,1u,mode?50u:1000u};
  size_t sz=sizeof(k); void* cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&k,HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
  CHECK(hipModuleLaunchKernel(f,1,1,1,32,1,1,0,nullptr,nullptr,cfg)); hipEventRecord(b);
  hipError_t e=hipEventSynchronize(b); if(e!=hipSuccess){std::printf("sync: %s\n",hipGetErrorString(e));return 3;}
  float ms; hipEventElapsedTime(&ms,a,b); CHECK(hipMemcpy(h,d,256,hipMemcpyDeviceToHost));
  uint64_t stamp; std::memcpy(&stamp,h+24,8); uint32_t cnt; std::memcpy(&cnt,h+40,4);
  bool guard=true; for(int i=48;i<256;i++) guard&=h[i]==0xA5; for(int i=4;i<24;i++) guard&=h[i]==0xA5;
  std::printf("mode=%d flag=%u elapsed=%.3fms stamp=%llu count=%u guards=%s\n",mode,w[0],ms,(unsigned long long)stamp,cnt,guard?"intact":"CORRUPT");
  if(!guard) rc=4;
 }
 return rc;
}
