// CPU-staged D3D12-readback -> HIP upload/download test. No neural kernel.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#define CHECK(call) do { auto rc=(call); if(rc!=hipSuccess) { std::printf("FAIL %s: %s\n",#call,hipGetErrorString(rc)); return 1; } } while(0)
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream input(argv[1],std::ios::binary);
    std::string magic; unsigned width=0,height=0,maximum=0;
    input>>magic>>width>>height>>maximum; input.get();
    if(magic!="P6" || width!=96 || height!=80 || maximum!=255) return 3;
    std::vector<unsigned char> pixels(width*height*3),result(pixels.size());
    input.read(reinterpret_cast<char*>(pixels.data()),pixels.size());
    if(input.gcount()!=pixels.size()) return 4;
    for(size_t i=0;i<pixels.size();i+=3) if(pixels[i]!=255 || pixels[i+1] || pixels[i+2]) return 5;
    hipDeviceProp_t props{}; CHECK(hipGetDeviceProperties(&props,0));
    if(std::strcmp(props.gcnArchName,"gfx1030") && std::strncmp(props.gcnArchName,"gfx1030:",8)) return 6;
    std::printf("HIP device=%s arch=%s\n",props.name,props.gcnArchName);
    void* gpu=nullptr; CHECK(hipMalloc(&gpu,pixels.size()));
    hipStream_t stream; CHECK(hipStreamCreate(&stream));
    CHECK(hipMemcpyAsync(gpu,pixels.data(),pixels.size(),hipMemcpyHostToDevice,stream));
    CHECK(hipMemcpyAsync(result.data(),gpu,result.size(),hipMemcpyDeviceToHost,stream));
    CHECK(hipStreamSynchronize(stream));
    bool equal=pixels==result;
    CHECK(hipFree(gpu)); CHECK(hipStreamDestroy(stream));
    std::printf("%s CPU-staged frame HIP roundtrip %zu bytes; no shared-resource interop or neural dispatch\n",equal?"PASS":"FAIL",pixels.size());
    return equal?0:7;
}
