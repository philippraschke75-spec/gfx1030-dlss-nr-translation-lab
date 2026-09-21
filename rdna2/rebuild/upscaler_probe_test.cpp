// Isolated test of the read-only upscaler probe: fake FSR3/FFX DLLs + a real D3D12 texture on the (AMD) GPU. No game involved.
#include "dxgi_probe.h"
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#define CHECK(c,msg) do{ if(!(c)){ std::printf("FAIL %s\n",msg); return 1; } }while(0)
int main(int argc,char** argv) {
    bool amd=argc>2 && !std::strcmp(argv[2],"--amd");
    HMODULE hook=LoadLibraryW(L"DLSSNRGraphicsProbe.dll"); CHECK(hook,"load hook dll");
    auto start=reinterpret_cast<HRESULT(WINAPI*)(const wchar_t*)>(GetProcAddress(hook,"RebuildStart"));
    auto poll=reinterpret_cast<void(WINAPI*)()>(GetProcAddress(hook,"RebuildProbeUpscalers")); CHECK(start&&poll,"exports");
    DeleteFileW(L"upscaler-probe-test.log");
    CHECK(SUCCEEDED(start(L"upscaler-probe-test.log")),"RebuildStart");
    HMODULE fsr=LoadLibraryW(L"ffx_fsr3upscaler_x64.dll"); HMODULE ffx=LoadLibraryW(L"amd_fidelityfx_upscaler_dx12.dll");
    CHECK(fsr&&ffx,"load fake upscaler dlls");
    poll();
    DxgiProbe probe; CHECK(SUCCEEDED(probe.create(amd)),"create device");
    D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width=2560; d.Height=1440; d.DepthOrArraySize=1; d.MipLevels=1;
    d.Format=DXGI_FORMAT_R16G16_FLOAT; d.SampleDesc.Count=1; d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> mv; CHECK(SUCCEEDED(probe.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&mv))),"create texture");
    // fake description: resource pointer at +0x18, a non-resource garbage pointer at +0x20, floats at +0x40
    alignas(16) unsigned char desc[0x300]{}; auto put=[&](size_t off,unsigned long long v){ std::memcpy(desc+off,&v,8); };
    put(0x00,0x10001ull); put(0x08,0ull); put(0x18,reinterpret_cast<unsigned long long>(mv.Get())); put(0x20,0x00007ff612345678ull);
    float jitter[2]={0.25f,-0.375f}; std::memcpy(desc+0x40,jitter,8);
    auto fsr_fn=reinterpret_cast<int(__cdecl*)(void*,const void*)>(GetProcAddress(fsr,"ffxFsr3UpscalerContextDispatch"));
    auto ffx_fn=reinterpret_cast<unsigned(__cdecl*)(void*,const void*)>(GetProcAddress(ffx,"ffxDispatch"));
    CHECK(fsr_fn&&ffx_fn,"resolve exports");
    int r1=fsr_fn(nullptr,desc); unsigned r2=ffx_fn(nullptr,desc);
    CHECK(r1==1234,"fsr return value forwarded"); CHECK(r2==77,"ffx return value forwarded");
    std::ifstream f("upscaler-probe-test.log"); std::stringstream ss; ss<<f.rdbuf(); std::string log=ss.str();
    CHECK(log.find("PROBE hook ffxFsr3UpscalerContextDispatch status=0")!=std::string::npos,"fsr hook installed");
    CHECK(log.find("PROBE hook ffxDispatch status=0")!=std::string::npos,"ffx hook installed");
    CHECK(log.find("D3D12 resource dim=3 2560x1440 mips=1 fmt=34")!=std::string::npos,"resource decoded (R16G16_FLOAT = 34)");
    CHECK(log.find("f32(0.25,-0.375)")!=std::string::npos,"jitter floats visible in dump");
    CHECK(log.find("ffxDispatch chain[0] type=0x10001")!=std::string::npos,"ffx chain type logged");
    size_t garbage=log.find("0x00007ff612345678"); CHECK(garbage!=std::string::npos,"garbage pointer dumped");
    CHECK(log.find("<= ",garbage)==std::string::npos || log.find("<= ",garbage)>log.find('\n',garbage),"garbage pointer NOT treated as resource");
    std::printf("PASS upscaler probe: detours install, forward and preserve returns; D3D12 resource decoded; garbage pointer rejected\n");
    return 0;
}
