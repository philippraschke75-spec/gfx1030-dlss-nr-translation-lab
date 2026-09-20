#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x);if(e!=hipSuccess){std::printf("%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);if(argc!=2)return 1;
    hipDeviceProp_t p{};CHECK(hipGetDeviceProperties(&p,0));
    if(std::strcmp(p.gcnArchName,"gfx1030") || p.warpSize!=32)return 3;
    constexpr size_t Guard=256,Size=4096+512*32;
    std::vector<unsigned char> expected(Size+2*Guard,0xa5),got=expected;
    for(uint32_t id=0;id<512;++id) {
        uint32_t table[]={id,id^0x11223344u,id+0x01020304u,~id};
        uint32_t row[]={table[id%4],(0x89abcdefu>>(8*(id%4)))&255,id+0x01020304u,id<256?31u:63u,id%32?31-id%32:32};
        std::memcpy(expected.data()+Guard+4096+id*32,row,sizeof(row));
    }
    unsigned char* d=nullptr;CHECK(hipMalloc(&d,got.size()));CHECK(hipMemcpy(d,got.data(),got.size(),hipMemcpyHostToDevice));
    hipModule_t m;hipFunction_t fn;CHECK(hipModuleLoad(&m,argv[1]));CHECK(hipModuleGetFunction(&fn,m,"lowering_probe"));
    int lds=0;CHECK(hipFuncGetAttribute(&lds,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));if(lds!=8192)return 4;
    void* dst=d+Guard;void* args[]={&dst};
    std::printf("GPU=%s; one dispatch: 2 groups x 256 threads, LDS=%d\n",p.name,lds);
    std::puts("Checks: private table isolation, dynamic word/byte loads, signed wide offsets, SCC preservation, CLZ including zero.");
    CHECK(hipModuleLaunchKernel(fn,2,1,1,256,1,1,0,nullptr,args,nullptr));CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(got.data(),d,got.size(),hipMemcpyDeviceToHost));
    size_t bad=0;for(size_t i=0;i<got.size();++i)if(got[i]!=expected[i]){if(bad<10)std::printf("byte %zu got=%02x expected=%02x\n",i,got[i],expected[i]);++bad;}
    std::printf("Checked 2560 result words plus untouched padding/guards; byte mismatches=%zu\n",bad);
    CHECK(hipFree(d));CHECK(hipModuleUnload(m));std::puts(bad?"LOWERING PROBE: FAIL":"LOWERING PROBE: PASS");return bad?5:0;
}
