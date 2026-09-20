#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess) { \
    std::printf("%s: %s (%d)\n", #x, hipGetErrorString(e), int(e)); return 2; } } while(0)
constexpr size_t Guard=256, WeightBytes=524288;
constexpr unsigned char Canary=0xa5;
struct HeadParams { void* input; void* output; void* weights; };
static_assert(sizeof(HeadParams)==24);

// Tensor and weight layouts reconstructed from the original kernel's address
// arithmetic. The numerical reference below is ordinary integer GEMM, not ISA
// interpretation, and never uses the replacement's lane/register functions.
static size_t tensor_offset(int row,int col) {
    return (col/32)*512+(row%8)*64+(row/8)*4+((col%16)/4)*16+col%4+((col/16)%2)*8;
}
static size_t weight_offset(int k,int col) {
    return (k/32)*32768+(col/16)*512+(col&1)*64+((col>>1)&1)*8+
           ((col>>2)&3)*128+(k&3)+((k>>2)&3)*16+((k>>4)&1)*4;
}
static unsigned char small_fp8(int units) {
    if(!units) return 0;
    return (units<0?128:0) | (std::abs(units)==1?0x18:0x20);
}
static double positive_fp8(int bits) {
    int e=bits>>3,m=bits&7;
    return e?std::ldexp(1.0+m/8.0,e-7):std::ldexp(double(m),-9);
}
static unsigned char encode_fp8(double value) {
    unsigned sign=value<0?128:0;
    value=std::min(448.0,std::abs(value));
    int best=0;double error=value;
    for(int bits=1;bits<=126;++bits) {
        double d=std::abs(value-positive_fp8(bits));
        if(d<error || (d==error && !(bits&1))) { best=bits;error=d; }
    }
    return sign|best;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=3) { std::puts("usage: final_head_test.exe module.co abi|constant|patterned|multigroup"); return 1; }
    const bool abi=std::strcmp(argv[2],"abi")==0;
    const bool multigroup=std::strcmp(argv[2],"multigroup")==0;
    const bool patterned=std::strcmp(argv[2],"patterned")==0 || multigroup;
    const unsigned groups=multigroup?2:1;
    const size_t InputBytes=8192*groups,OutputBytes=16384*groups;
    if(!abi && !patterned && std::strcmp(argv[2],"constant")!=0) return 1;
    hipDeviceProp_t prop{}; CHECK(hipGetDeviceProperties(&prop,0));
    if(std::strcmp(prop.gcnArchName,"gfx1030")!=0 || prop.warpSize!=32) return 4;
    std::printf("GPU=%s arch=%s\n",prop.name,prop.gcnArchName);
    hipModule_t module=nullptr; CHECK(hipModuleLoad(&module,argv[1]));
    hipFunction_t fn=nullptr; CHECK(hipModuleGetFunction(&fn,module,"_Z12k_final_head10HeadParams"));
    int regs=0, lds=0, max_threads=0;
    CHECK(hipFuncGetAttribute(&regs,HIP_FUNC_ATTRIBUTE_NUM_REGS,fn));
    CHECK(hipFuncGetAttribute(&lds,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));
    CHECK(hipFuncGetAttribute(&max_threads,HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,fn));
    std::printf("module loaded: VGPRs=%d LDS=%d maxThreads=%d\n",regs,lds,max_threads);
    if(lds!=8192 || max_threads<256) return 5;
    std::vector<unsigned char> in(InputBytes+2*Guard,Canary),out(OutputBytes+2*Guard,Canary),weights(WeightBytes+2*Guard,Canary);
    // E4M3 0x20 is 1/8. With both operands constant, the 512-term
    // product is 512*(1/8)*(1/8)=8, whose E4M3 encoding is 0x50.
    // This deliberately tests a simple numerical case, not full equivalence.
    std::fill(in.begin()+Guard,in.end()-Guard,0x20);
    std::fill(weights.begin()+Guard,weights.end()-Guard,0x20);
    std::vector<unsigned char> expected(OutputBytes,0x50);
    if(patterned) {
        std::vector<int> a(16*groups*512),b(512*1024);
        std::vector<bool> seen_input(InputBytes),seen_weights(WeightBytes),seen_output(OutputBytes);
        uint32_t seed=0x6900abcd;
        auto random=[&]() { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return int(seed%5)-2; };
        for(unsigned r=0;r<16*groups;++r) for(int k=0;k<512;++k) {
            int x=random();a[r*512+k]=x;
            size_t off=(r/16)*8192+tensor_offset(r%16,k);
            if(off>=InputBytes || seen_input[off]) return 7;
            seen_input[off]=true;in[Guard+off]=small_fp8(x);
        }
        for(int k=0;k<512;++k) for(int col=0;col<1024;++col) {
            int x=random();b[k*1024+col]=x;
            size_t off=weight_offset(k,col);
            if(off>=WeightBytes || seen_weights[off]) return 7;
            seen_weights[off]=true;weights[Guard+off]=small_fp8(x);
        }
        for(unsigned r=0;r<16*groups;++r) for(int col=0;col<1024;++col) {
            int sum=0;
            for(int k=0;k<512;++k) sum+=a[r*512+k]*b[k*1024+col];
            size_t off=(r/16)*16384+tensor_offset(r%16,col);
            if(off>=OutputBytes || seen_output[off]) return 7;
            seen_output[off]=true;expected[off]=encode_fp8(sum/256.0);
        }
        std::printf("Patterned reference: independent %ux512 by 512x1024 integer GEMM + E4M3 rounding\n",16*groups);
    }
    unsigned char *di=nullptr,*dout=nullptr,*dw=nullptr;
    CHECK(hipMalloc(&di,in.size())); CHECK(hipMalloc(&dout,out.size())); CHECK(hipMalloc(&dw,weights.size()));
    CHECK(hipMemcpy(di,in.data(),in.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dout,out.data(),out.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw,weights.data(),weights.size(),hipMemcpyHostToDevice));
    HeadParams params{di+Guard,dout+Guard,dw+Guard};
    void* args[]={&params};
    auto start=std::chrono::steady_clock::now();
    std::printf("Launching one dispatch: %u workgroup(s) of 256 threads\n",groups);
    CHECK(hipModuleLaunchKernel(fn,groups,1,1,256,1,1,0,nullptr,args,nullptr));
    CHECK(hipDeviceSynchronize());
    std::printf("Synchronization returned after %.3f ms (not itself success evidence)\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    std::vector<unsigned char> readin(in.size()),readweights(weights.size());
    CHECK(hipMemcpy(readin.data(),di,in.size(),hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(out.data(),dout,out.size(),hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(readweights.data(),dw,weights.size(),hipMemcpyDeviceToHost));
    bool guards=true;
    for(size_t i=0;i<Guard;++i) {
        guards &= readin[i]==Canary && readin[Guard+InputBytes+i]==Canary;
        guards &= out[i]==Canary && out[Guard+OutputBytes+i]==Canary;
        guards &= readweights[i]==Canary && readweights[Guard+WeightBytes+i]==Canary;
    }
    bool pass=guards;
    if(abi) {
        unsigned fields[4]{};
        std::memcpy(fields,readin.data()+Guard,sizeof(fields));
        std::printf("Hidden args: blocks=(%u,%u,%u) groupXY=0x%08x\n",fields[0],fields[1],fields[2],fields[3]);
        pass &= fields[0]==1 && fields[1]==1 && fields[2]==1 && fields[3]==0x00010100;
    } else {
        size_t histogram[256]{};size_t mismatches=0;
        for(size_t i=0;i<OutputBytes;++i) { ++histogram[out[Guard+i]]; mismatches += out[Guard+i]!=expected[i]; }
        std::printf("Output bytes=%zu, mismatches against %s reference=%zu\n",OutputBytes,argv[2],mismatches);
        int kinds=0;for(int i=0;i<256;++i) kinds+=histogram[i]!=0;
        std::printf("Distinct output byte values=%d\n",kinds);
        if(mismatches) for(size_t i=0,shown=0;i<OutputBytes && shown<12;++i)
            if(out[Guard+i]!=expected[i]) { std::printf("offset=%zu got=%02x expected=%02x\n",i,out[Guard+i],expected[i]);++shown; }
        pass &= mismatches==0 && readin==in && readweights==weights;
        FILE* f=nullptr;
        const char* output_name=multigroup?"final-head-multigroup-output.bin":
                                (patterned?"final-head-patterned-output.bin":"final-head-output.bin");
        if(fopen_s(&f,output_name,"wb")!=0 || !f) return 6;
        std::fwrite(out.data()+Guard,1,OutputBytes,f);std::fclose(f);
    }
    std::printf("Allocation guards: %s\n",guards?"intact":"FAILED");
    CHECK(hipFree(di)); CHECK(hipFree(dout)); CHECK(hipFree(dw)); CHECK(hipModuleUnload(module));
    std::printf("%s: %s\n",abi?"ABI PROBE":(patterned?"PATTERNED KERNEL TEST":"CONSTANT KERNEL TEST"),pass?"PASS":"FAIL");
    return pass?0:3;
}
