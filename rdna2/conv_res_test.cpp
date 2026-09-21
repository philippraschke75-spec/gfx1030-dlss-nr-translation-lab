#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) { std::printf("%s: %s\n",#x,hipGetErrorString(e));return 2; } } while(0)
constexpr size_t Guard=256,Tensor=8192,Weights=262144+1024;
constexpr unsigned char Canary=0xa5;
struct ConvParams { void *input,*residual,*output,*weights; uint64_t reserved; };
static_assert(sizeof(ConvParams)==40);
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=2) return 1;
    hipDeviceProp_t prop{};CHECK(hipGetDeviceProperties(&prop,0));
    if(std::strcmp(prop.gcnArchName,"gfx1030") || prop.warpSize!=32) return 3;
    std::printf("GPU=%s arch=%s; one 256-thread workgroup; no TDR changes or retries\n",prop.name,prop.gcnArchName);
    hipModule_t module;hipFunction_t fn;CHECK(hipModuleLoad(&module,argv[1]));
    CHECK(hipModuleGetFunction(&fn,module,"_Z10k_conv_res10ConvParams"));
    int lds=0;CHECK(hipFuncGetAttribute(&lds,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));
    if(lds!=8192) return 4;
    const size_t sizes[]={Tensor,Tensor,Tensor,Weights};
    std::vector<unsigned char> host[4],got[4];unsigned char* device[4]{};
    for(int i=0;i<4;++i) host[i].assign(sizes[i]+2*Guard,Canary);
    std::fill(host[0].begin()+Guard,host[0].end()-Guard,0x20); // 1/8
    constexpr unsigned char residuals[]={0xc8,0xc0,0xb8,0,0x38,0x40,0x48}; // -4,-2,-1,0,1,2,4
    constexpr unsigned char expectedCodes[]={0x48,0x4c,0x4e,0x50,0x51,0x52,0x54}; // 8 + residual
    std::vector<unsigned char> expected(Tensor);
    for(size_t i=0;i<Tensor;++i) { size_t j=(i*13+i/31)%7;host[1][Guard+i]=residuals[j];expected[i]=expectedCodes[j]; }
    std::fill(host[3].begin()+Guard,host[3].begin()+Guard+262144,0x20);
    for(size_t i=262144;i<Weights;i+=2) { host[3][Guard+i]=0;host[3][Guard+i+1]=0x3c; } // half 1.0 scales
    for(int i=0;i<4;++i) { CHECK(hipMalloc(&device[i],host[i].size()));CHECK(hipMemcpy(device[i],host[i].data(),host[i].size(),hipMemcpyHostToDevice)); }
    ConvParams params{device[0]+Guard,device[1]+Guard,device[2]+Guard,device[3]+Guard,0};void* args[]={&params};
    std::puts("Reference: 512*(1/8)*(1/8) + patterned residual, unit FP16 scales; 8192 expected FP8 bytes");
    CHECK(hipModuleLaunchKernel(fn,1,1,1,256,1,1,0,nullptr,args,nullptr));CHECK(hipDeviceSynchronize());
    bool guards=true,readonly=true;size_t mismatches=0;
    for(int i=0;i<4;++i) {
        got[i].resize(host[i].size());CHECK(hipMemcpy(got[i].data(),device[i],got[i].size(),hipMemcpyDeviceToHost));
        for(size_t j=0;j<Guard;++j) guards &= got[i][j]==Canary && got[i][Guard+sizes[i]+j]==Canary;
        if(i!=2) readonly &= got[i]==host[i];
    }
    for(size_t i=0;i<Tensor;++i) if(got[2][Guard+i]!=expected[i]) {
        if(mismatches<12) std::printf("offset=%zu got=%02x expected=%02x\n",i,got[2][Guard+i],expected[i]);
        ++mismatches;
    }
    std::printf("Compared=%zu mismatches=%zu guards=%s readonly_inputs=%s\n",Tensor,mismatches,guards?"intact":"FAILED",readonly?"unchanged":"FAILED");
    for(auto p:device) CHECK(hipFree(p));CHECK(hipModuleUnload(module));
    bool pass=!mismatches && guards && readonly;std::printf("CONV RES TEST: %s\n",pass?"PASS":"FAIL");return pass?0:5;
}
