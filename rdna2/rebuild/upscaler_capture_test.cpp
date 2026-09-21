// Isolated GPU test of the F9 pixel capture: known-pattern textures (colour/mvec RGBA16F, depth R32G8X24) in the state the FSR3 description
// reports (PIXEL|NON_PIXEL_SHADER_RESOURCE = FFX state 12), a real D3D12 command list standing in for the game's, the detour recording into it, real
// GPU execution, the Present-thread tick, then byte-exact comparison of the written files. No game involved.
#include "dxgi_probe.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#define CHECK(c,msg) do{ if(!(c)){ std::printf("FAIL %s (line %d)\n",msg,__LINE__); return 1; } }while(0)
#define HR(x,msg) do{ HRESULT h_=(x); if(FAILED(h_)){ std::printf("FAIL %s hr=0x%08lx\n",msg,(unsigned long)h_); return 1; } }while(0)
static const UINT W=64, H=48;
static unsigned short pat(UINT x,UINT y,UINT c) { return (unsigned short)((x*3+y*5+c*11)&0xffff); }
int main(int argc,char** argv) {
    bool amd=argc>1 && !std::strcmp(argv[1],"--amd");
    HMODULE hook=LoadLibraryW(L"DLSSNRGraphicsProbe.dll"); CHECK(hook,"load hook");
    auto start=(HRESULT(WINAPI*)(const wchar_t*))GetProcAddress(hook,"RebuildStart");
    auto poll=(void(WINAPI*)())GetProcAddress(hook,"RebuildProbeUpscalers");
    auto arm=(BOOL(WINAPI*)(const wchar_t*))GetProcAddress(hook,"RebuildArmUpscalerCapture");
    auto tick=(void(WINAPI*)(ID3D12CommandQueue*))GetProcAddress(hook,"RebuildUpscalerCaptureTick");
    CHECK(start&&poll&&arm&&tick,"exports");
    DeleteFileW(L"capture-test.log"); HR(start(L"capture-test.log"),"start");
    HMODULE fsr=LoadLibraryW(L"ffx_fsr3upscaler_x64.dll"); CHECK(fsr,"fake fsr"); poll();
    auto dispatch=(int(__cdecl*)(void*,const void*))GetProcAddress(fsr,"ffxFsr3UpscalerContextDispatch"); CHECK(dispatch,"resolve");
    DxgiProbe probe; HR(probe.create(amd),"device"); ID3D12Device* dev=probe.device.Get(); ID3D12CommandQueue* q=probe.queue.Get();
    auto make_tex=[&](DXGI_FORMAT fmt,D3D12_RESOURCE_FLAGS flags,D3D12_RESOURCE_STATES st,ComPtr<ID3D12Resource>& out)->HRESULT {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type=D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width=W; d.Height=H; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=fmt; d.SampleDesc.Count=1; d.Flags=flags;
        return dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,st,nullptr,IID_PPV_ARGS(&out)); };
    ComPtr<ID3D12Resource> color,mvec,depth;
    HR(make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST,color),"color");
    HR(make_tex(DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST,mvec),"mvec");
    HR(make_tex(DXGI_FORMAT_R32G8X24_TYPELESS,D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,D3D12_RESOURCE_STATE_DEPTH_WRITE,depth),"depth");
    // upload buffer with pattern for colour and mvec
    auto rdesc=color->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT rows; UINT64 rowb,total;
    dev->GetCopyableFootprints(&rdesc,0,1,0,&fp,&rows,&rowb,&total);
    ComPtr<ID3D12Resource> up[2];
    for(int k=0;k<2;++k) {
        D3D12_HEAP_PROPERTIES hp{}; hp.Type=D3D12_HEAP_TYPE_UPLOAD; D3D12_RESOURCE_DESC b{}; b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; b.Width=total; b.Height=1; b.DepthOrArraySize=1; b.MipLevels=1; b.SampleDesc.Count=1; b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        HR(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&up[k])),"upload");
        unsigned char* m; up[k]->Map(0,nullptr,(void**)&m);
        for(UINT y=0;y<H;++y) for(UINT x=0;x<W;++x) for(UINT c=0;c<4;++c) { unsigned short v=pat(x,y,c+4*k); std::memcpy(m+fp.Offset+y*fp.Footprint.RowPitch+(x*4+c)*2,&v,2); }
        up[k]->Unmap(0,nullptr);
    }
    D3D12_DESCRIPTOR_HEAP_DESC dh{}; dh.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; dh.NumDescriptors=1; ComPtr<ID3D12DescriptorHeap> dsvh; HR(dev->CreateDescriptorHeap(&dh,IID_PPV_ARGS(&dsvh)),"dsv heap");
    D3D12_DEPTH_STENCIL_VIEW_DESC dv{}; dv.Format=DXGI_FORMAT_D32_FLOAT_S8X24_UINT; dv.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D; dev->CreateDepthStencilView(depth.Get(),&dv,dsvh->GetCPUDescriptorHandleForHeapStart());
    ComPtr<ID3D12CommandAllocator> al; ComPtr<ID3D12GraphicsCommandList> l1,l2; ComPtr<ID3D12Fence> fence; HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr); UINT64 fv=0;
    HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&al)),"alloc");
    HR(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al.Get(),nullptr,IID_PPV_ARGS(&l1)),"list1"); HR(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
    auto wait=[&]() { ++fv; q->Signal(fence.Get(),fv); if(fence->GetCompletedValue()<fv) { fence->SetEventOnCompletion(fv,ev); WaitForSingleObject(ev,10000); } };
    auto bar=[&](ID3D12GraphicsCommandList* l,ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b) { D3D12_RESOURCE_BARRIER x{}; x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource=r; x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; x.Transition.StateBefore=a; x.Transition.StateAfter=b; l->ResourceBarrier(1,&x); };
    const D3D12_RESOURCE_STATES SRV=(D3D12_RESOURCE_STATES)(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    ID3D12Resource* cm[2]={color.Get(),mvec.Get()};
    for(int k=0;k<2;++k) { D3D12_TEXTURE_COPY_LOCATION s{}; s.pResource=up[k].Get(); s.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; s.PlacedFootprint=fp;
        D3D12_TEXTURE_COPY_LOCATION d{}; d.pResource=cm[k]; d.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; l1->CopyTextureRegion(&d,0,0,0,&s,nullptr); bar(l1.Get(),cm[k],D3D12_RESOURCE_STATE_COPY_DEST,SRV); }
    l1->ClearDepthStencilView(dsvh->GetCPUDescriptorHandleForHeapStart(),D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,0.25f,0,0,nullptr); bar(l1.Get(),depth.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE,SRV);
    l1->Close(); ID3D12CommandList* ls1[]={l1.Get()}; q->ExecuteCommandLists(1,ls1); wait();
    // ---- the "game" command list; description as observed in the game
    ComPtr<ID3D12CommandAllocator> al2; HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&al2)),"alloc2");
    HR(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al2.Get(),nullptr,IID_PPV_ARGS(&l2)),"list2");
    alignas(16) unsigned char desc[0x800]{}; auto p64=[&](size_t o,unsigned long long v){ std::memcpy(desc+o,&v,8); }; auto p32=[&](size_t o,unsigned v){ std::memcpy(desc+o,&v,4); };
    p64(0x00,(unsigned long long)l2.Get());
    struct E{ size_t off; ID3D12Resource* r; } es[3]={{0x08,color.Get()},{0xb8,depth.Get()},{0x168,mvec.Get()}};
    for(auto& e:es) { p64(e.off,(unsigned long long)e.r); p32(e.off+0x28,12); }
    float jm[4]={0.25f,-0.375f,(float)W,(float)H}; std::memcpy(desc+0x6e8,jm,16); p32(0x6f8,W); p32(0x6fc,H);
    DeleteFileW(L"cap-out.color_rgba16f.bin"); DeleteFileW(L"cap-out.depth_plane0_r32.bin"); DeleteFileW(L"cap-out.mvec_rgba16f.bin"); DeleteFileW(L"cap-out.upscaler_meta.txt");
    CHECK(arm(L"cap-out.ppm"),"arm");
    CHECK(!arm(L"cap-out.ppm"),"second arm refused while armed");
    int r=dispatch(nullptr,desc); CHECK(r==1234,"original dispatch still called and return preserved");
    l2->Close(); ID3D12CommandList* ls2[]={l2.Get()}; q->ExecuteCommandLists(1,ls2); wait();
    for(int i=0;i<4;++i) tick(q);
    auto slurp=[&](const char* f){ std::ifstream in(f,std::ios::binary); std::stringstream ss; ss<<in.rdbuf(); return ss.str(); };
    std::string cb=slurp("cap-out.color_rgba16f.bin"), db=slurp("cap-out.depth_plane0_r32.bin"), mb=slurp("cap-out.mvec_rgba16f.bin"), meta=slurp("cap-out.upscaler_meta.txt");
    CHECK(cb.size()==size_t(W)*H*8,"colour file size"); CHECK(mb.size()==size_t(W)*H*8,"mvec file size"); CHECK(db.size()==size_t(W)*H*4,"depth file size");
    size_t bad=0;
    for(UINT y=0;y<H;++y) for(UINT x=0;x<W;++x) for(UINT c=0;c<4;++c) { unsigned short a,b; std::memcpy(&a,cb.data()+(y*W+x)*8+c*2,2); std::memcpy(&b,mb.data()+(y*W+x)*8+c*2,2); if(a!=pat(x,y,c)) ++bad; if(b!=pat(x,y,c+4)) ++bad; }
    CHECK(bad==0,"colour and mvec bytes equal the pattern");
    size_t dbad=0; for(size_t i=0;i<size_t(W)*H;++i) { float f; std::memcpy(&f,db.data()+i*4,4); if(f!=0.25f) ++dbad; } CHECK(dbad==0,"depth plane equals 0.25");
    CHECK(meta.find("jitter_offset(+0x6e8)=0.25,-0.375")!=std::string::npos,"jitter in metadata");
    CHECK(meta.find("render_size(+0x6f8)=64,48")!=std::string::npos,"render size in metadata");
    CHECK(meta.find("color: 64x48 dxgi_format=10")!=std::string::npos,"colour dims in metadata");
    std::ifstream lg("capture-test.log"); std::stringstream ls; ls<<lg.rdbuf(); std::string log=ls.str();
    CHECK(log.find("UPSCALER_CAPTURE SAVED")!=std::string::npos,"log says SAVED");
    CHECK(log.find("FAILED")==std::string::npos,"no failure lines");
    // the resources must still be in the states the game believes: transition them out of SRV as the game would
    ComPtr<ID3D12GraphicsCommandList> l3; ComPtr<ID3D12CommandAllocator> al3; dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&al3)); dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,al3.Get(),nullptr,IID_PPV_ARGS(&l3));
    bar(l3.Get(),color.Get(),SRV,D3D12_RESOURCE_STATE_COMMON); bar(l3.Get(),depth.Get(),SRV,D3D12_RESOURCE_STATE_COMMON); bar(l3.Get(),mvec.Get(),SRV,D3D12_RESOURCE_STATE_COMMON);
    l3->Close(); ID3D12CommandList* ls3[]={l3.Get()}; q->ExecuteCommandLists(1,ls3); wait();
    CHECK(SUCCEEDED(dev->GetDeviceRemovedReason())==true,"device not removed");
    std::printf("PASS upscaler pixel capture: byte-exact colour+mvec, depth plane 0 correct, metadata correct, states restored, device healthy\n");
    return 0;
}
