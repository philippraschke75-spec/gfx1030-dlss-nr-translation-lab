#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) { std::printf("%s: %s\n",#x,hipGetErrorString(e));return 2; } } while(0)
constexpr size_t Guard=256;
constexpr unsigned char Canary=0xa5;
struct MeanParams { void* input; int stride,height,width,pad; void* output; };
static_assert(sizeof(MeanParams)==32 && offsetof(MeanParams,output)==24);
static bool close(float a,float b) { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=2e-6f*std::max(1.0f,std::abs(b)); }

int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=3) { std::puts("usage: kernels_test.exe module.co align|mean");return 1; }
    bool mean=std::strcmp(argv[2],"mean")==0,align=std::strcmp(argv[2],"align")==0;
    if(!mean && !align) return 1;
    // Negative controls run on the host before touching the device.
    if(close(NAN,1) || close(INFINITY,1) || close(2,1)) return 3;
    hipDeviceProp_t prop{};CHECK(hipGetDeviceProperties(&prop,0));
    if(std::strcmp(prop.gcnArchName,"gfx1030") || prop.warpSize!=32) return 4;
    int runtime=0,driver=0;CHECK(hipRuntimeGetVersion(&runtime));CHECK(hipDriverGetVersion(&driver));
    std::printf("GPU=%s arch=%s runtime=%d driver=%d\n",prop.name,prop.gcnArchName,runtime,driver);
    std::puts("One dispatch, no automatic retry; no watchdog/TDR changes by this test.");
    hipModule_t module=nullptr;CHECK(hipModuleLoad(&module,argv[1]));
    hipFunction_t fn=nullptr;
    CHECK(hipModuleGetFunction(&fn,module,mean?"_Z6k_mean10MeanParams":"_Z13k_align_probePh"));
    int regs=0,lds=0;CHECK(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,fn));
    CHECK(hipFuncGetAttribute(&lds,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));
    std::printf("Module: VGPRs=%d LDS=%d\n",regs,lds);
    if(lds!=(mean?1024:0)) return 5;
    constexpr int Width=19,Height=17,Stride=23;
    const size_t inputBytes=mean?Stride*Height*3*sizeof(float):8;
    std::vector<unsigned char> input(inputBytes+2*Guard,Canary), output(4+2*Guard,Canary);
    float initial=3.5f,expected=0;
    std::memcpy(output.data()+Guard,&initial,4);
    if(mean) {
        double sum=0;
        for(int y=0;y<Height;++y) for(int x=0;x<Stride;++x) {
            float rgb[3]={float((x+3*y)%17)/16.0f,float((5*x+y)%19)/8.0f,float((x+7*y)%23)/32.0f};
            if(x>=Width) rgb[0]=rgb[1]=rgb[2]=1000.0f; // detect ignored row pitch/padding
            std::memcpy(input.data()+Guard+(y*Stride+x)*12,rgb,12);
            if(x<Width) sum+=0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2];
        }
        expected=initial+float(sum/(Width*Height));
        std::printf("Mean fixture: %dx%d RGB, stride=%d, initial accumulator=%.2f, CPU expected=%.9g\n",Width,Height,Stride,initial,expected);
    }
    unsigned char *di=nullptr,*dout=nullptr;
    CHECK(hipMalloc(&di,input.size()));CHECK(hipMalloc(&dout,output.size()));
    CHECK(hipMemcpy(di,input.data(),input.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dout,output.data(),output.size(),hipMemcpyHostToDevice));
    void* p=di+Guard;MeanParams params{di+Guard,Stride,Height,Width,0,dout+Guard};
    void* args[]={mean?static_cast<void*>(&params):static_cast<void*>(&p)};
    std::printf("Launching 1 workgroup, %d threads\n",mean?256:32);
    CHECK(hipModuleLaunchKernel(fn,1,1,1,mean?256:32,1,1,0,nullptr,args,nullptr));
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned char> gotInput(input.size()),gotOutput(output.size());
    CHECK(hipMemcpy(gotInput.data(),di,gotInput.size(),hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(gotOutput.data(),dout,gotOutput.size(),hipMemcpyDeviceToHost));
    bool pass=true;
    if(mean) {
        float got;std::memcpy(&got,gotOutput.data()+Guard,4);
        std::printf("Mean got=%.9g expected=%.9g error=%.9g\n",got,expected,std::abs(got-expected));
        pass=close(got,expected) && gotInput==input;
    } else {
        input[Guard+1]=0x11;input[Guard+2]=0x22;
        pass=gotInput==input && gotOutput==output;
        std::printf("Probe bytes: %02x %02x %02x %02x\n",gotInput[Guard],gotInput[Guard+1],gotInput[Guard+2],gotInput[Guard+3]);
    }
    bool guards=true;
    for(size_t i=0;i<Guard;++i) guards &= gotInput[i]==Canary && gotInput[Guard+inputBytes+i]==Canary && gotOutput[i]==Canary && gotOutput[Guard+4+i]==Canary;
    pass &= guards;
    std::printf("Guards=%s numerical comparison=%s\n",guards?"intact":"FAILED",pass?"PASS":"FAIL");
    CHECK(hipFree(di));CHECK(hipFree(dout));CHECK(hipModuleUnload(module));
    return pass?0:6;
}
