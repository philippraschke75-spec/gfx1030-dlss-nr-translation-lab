// Host-only controls. Uses ordinary CPU memory; never loads HIP or dispatches.
#include "launch_capture.h"
#include <cassert>
#include <fstream>
#include <string>

int main() {
    unsigned grid[]={2,3,1}, block[]={256,1,1};
    unsigned char params[168];
    for(unsigned i=0;i<168;++i) params[i]=static_cast<unsigned char>(i);
    void* args[]={params};
    const void* host=reinterpret_cast<void*>(0x1234);
    auto module=reinterpret_cast<void**>(0x5678);
    SetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",nullptr);
    assert(!launch_capture::before_launch(host,grid,block,args,0,nullptr));
    wchar_t filename[100];
    swprintf(filename,100,L"capture-test-%lu.json",GetCurrentProcessId());
    SetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",filename);
    assert(launch_capture::before_launch(host,grid,block,args,0,nullptr));
    assert(!launch_capture::attempted); // Unknown symbol never dereferences args.
    launch_capture::registration(module,host,launch_capture::symbols[0]);
    assert(launch_capture::before_launch(host,grid,block,args,17,nullptr));
    std::ifstream file(filename);
    std::string text((std::istreambuf_iterator<char>(file)),{});
    assert(text.find("\"args_read_ok\":true")!=std::string::npos);
    assert(text.find("0001020304050607")!=std::string::npos);
    assert(text.find("\"grid\":[2,3,1]")!=std::string::npos);
    assert(text.find("\"launch_forwarded\":false")!=std::string::npos);
    assert(launch_capture::before_launch(host,grid,block,nullptr,0,nullptr));
    launch_capture::unregister_module(module);
    assert(launch_capture::count==0);
    launch_capture::attempted=false;
    launch_capture::registration(module,host,launch_capture::symbols[0]);
    swprintf(filename,100,L"capture-test-bad-%lu.json",GetCurrentProcessId());
    SetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",filename);
    assert(launch_capture::before_launch(host,grid,block,reinterpret_cast<void**>(1),0,nullptr));
    std::ifstream bad(filename);
    std::string failure((std::istreambuf_iterator<char>(bad)),{});
    assert(failure.find("\"args_read_ok\":false")!=std::string::npos);
    SetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",nullptr);
    std::puts("PASS: disabled, unknown, exact bytes, dimensions, one-shot, unregister, invalid address");
}
