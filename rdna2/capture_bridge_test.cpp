#include <windows.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
struct Dim { unsigned x,y,z; };
template<class F> F get(HMODULE h,const char* name) {
    auto p=GetProcAddress(h,name); assert(p); return reinterpret_cast<F>(p);
}
int main(int argc,char** argv) {
    assert(argc==2);
    auto h=LoadLibraryA(argv[1]); assert(h);
    void* event=nullptr; void* stream=nullptr;
    assert(get<int(*)(void**,unsigned)>(h,"hipEventCreateWithFlags")(&event,0x1234)==17);
    assert(event==reinterpret_cast<void*>(0x12345678));
    assert(get<int(*)(void*)>(h,"hipEventQuery")(event)==600);
    assert(get<int(*)(void**,unsigned)>(h,"hipStreamCreateWithFlags")(&stream,0x5678)==19);
    assert(stream==reinterpret_cast<void*>(0x23456789));
    assert(get<int(*)(void*)>(h,"hipStreamSynchronize")(stream)==23);
    using Launch=int(*)(const void*,Dim,Dim,void**,size_t,void*);
    auto launch=get<Launch>(h,"hipLaunchKernel");
    SetEnvironmentVariableA("DLSSNR_CAPTURE_FILE",nullptr);
    SetEnvironmentVariableA("DLSSNR_GFX1030_ALLOW_KERNEL_LAUNCH","1");
    assert(launch(nullptr,{1,1,1},{256,1,1},nullptr,0,nullptr)==719);
    using Register=void(*)(void**,const void*,char*,const char*,unsigned,void*,void*,void*,void*,int*);
    const void* host=reinterpret_cast<void*>(0x1234);
    get<Register>(h,"__hipRegisterFunction")(nullptr,host,nullptr,
      "_Z10k_swin_varILi32ELb1EEv9VarParams",256,nullptr,nullptr,nullptr,nullptr,nullptr);
    char path[100]; std::snprintf(path,sizeof(path),"bridge-capture-%lu.json",GetCurrentProcessId());
    SetEnvironmentVariableA("DLSSNR_CAPTURE_FILE",path);
    unsigned char bytes[168]{}; bytes[0]=0xab; void* args[]={bytes};
    assert(launch(host,{2,1,1},{256,1,1},args,0,nullptr)==719);
    std::ifstream input(path); std::string record((std::istreambuf_iterator<char>(input)),{});
    assert(record.find("\"args_hex\":\"ab0000")!=std::string::npos);
    assert(record.find("\"launch_forwarded\":false")!=std::string::npos);
    std::puts("PASS: four ABI forwards preserve args/results; forced gate blocks; integrated capture writes exact args");
}
