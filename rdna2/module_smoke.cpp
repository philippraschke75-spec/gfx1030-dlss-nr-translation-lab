#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::printf("%s: %s (%d)\n",#x,hipGetErrorString(e),int(e)); return 2;} } while(0)
int main(int argc,char** argv){
 if(argc!=3)return 1; hipDeviceProp_t p{};CHECK(hipGetDeviceProperties(&p,0));
 if(std::strcmp(p.gcnArchName,"gfx1030")||p.warpSize!=32)return 3;
 hipModule_t m=nullptr;hipFunction_t f=nullptr;CHECK(hipModuleLoad(&m,argv[1]));CHECK(hipModuleGetFunction(&f,m,argv[2]));
 int regs=0,lds=0,maxThreads=0;CHECK(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,f));CHECK(hipFuncGetAttribute(&lds,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f));CHECK(hipFuncGetAttribute(&maxThreads,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f));
 std::printf("GPU=%s arch=%s symbol=%s VGPRs=%d LDS=%d maxThreads=%d\n",p.name,p.gcnArchName,argv[2],regs,lds,maxThreads);
 bool pass=regs>0 && maxThreads>=256 && lds<=65536;CHECK(hipModuleUnload(m));std::puts(pass?"MODULE LOAD TEST: PASS":"MODULE LOAD TEST: FAIL");return pass?0:4;
}
