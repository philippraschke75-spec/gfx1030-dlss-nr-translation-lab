// Read-only probe of the game's upscaler dispatch (FSR3 native or FidelityFX API). Logs which entry point is used, how often,
// and - for the first few calls - the raw description bytes plus every pointer slot that is a D3D12 resource (dimensions/format).
// It never reads pixels, never modifies arguments and always forwards to the original function.
#pragma once
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cwchar>
#include "third_party/minhook/include/MinHook.h"

static void record(const char* text);   // defined in graphics_hook.cpp before this header is included

namespace upscaler_probe {
constexpr int kDetailedCalls=3;         // full dump for the first N dispatches of each entry point
constexpr SIZE_T kDumpBytes=0x780;      // bytes of the description dumped

typedef int (__cdecl *FsrDispatchFn)(void*,const void*);
typedef unsigned (__cdecl *FfxDispatchFn)(void*,const void*);
static FsrDispatchFn orig_fsr=nullptr; static FfxDispatchFn orig_ffx=nullptr;
static volatile LONG fsr_calls=0, ffx_calls=0; static bool fsr_hooked=false, ffx_hooked=false;
static LONGLONG last_fsr_t=0;

// --- guarded raw reads
// ReadProcessMemory on our own process fails with an error code on unreadable memory instead of faulting (no SEH dependency).
static bool safe_copy(const void* src,void* dst,SIZE_T n) {
    SIZE_T got=0; return ReadProcessMemory(GetCurrentProcess(),src,dst,n,&got) && got==n;
}
static bool safe_read_u64(const void* p,unsigned long long* out) { return safe_copy(p,out,8); }
static bool module_is_d3d12(const void* addr,char* name,SIZE_T cap) {
    HMODULE m=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(addr),&m) || !m) return false;
    char path[MAX_PATH]; DWORD n=GetModuleFileNameA(m,path,sizeof(path)); if(!n) return false;
    const char* base=path; for(const char* c=path;*c;++c) if(*c=='\\'||*c=='/') base=c+1;
    std::snprintf(name,cap,"%s",base);
    char lower[MAX_PATH]; std::snprintf(lower,sizeof(lower),"%s",base); for(char* c=lower;*c;++c) if(*c>='A'&&*c<='Z') *c=char(*c+32);
    return std::strstr(lower,"d3d12")!=nullptr;
}
// Describe the D3D12 resource at pointer slot `p` if (and only if) its vtable lives in a d3d12 module.
static bool describe_resource(unsigned long long p,char* out,SIZE_T cap) {
    if(p<0x10000ull || (p&7) || p>0x00007fffffffffffull) return false;
    unsigned long long vtbl=0; if(!safe_read_u64(reinterpret_cast<void*>(p),&vtbl)) return false;
    char mod[64]; if(!module_is_d3d12(reinterpret_cast<void*>(vtbl),mod,sizeof(mod))) return false;
    ID3D12Resource* res=nullptr; HRESULT hr=E_FAIL;
    hr=reinterpret_cast<IUnknown*>(p)->QueryInterface(__uuidof(ID3D12Resource),reinterpret_cast<void**>(&res));   // vtable verified inside d3d12 module above
    if(FAILED(hr)||!res) return false;
    D3D12_RESOURCE_DESC d=res->GetDesc(); res->Release();
    std::snprintf(out,cap,"D3D12 resource dim=%d %llux%u mips=%u fmt=%u flags=0x%x",int(d.Dimension),(unsigned long long)d.Width,d.Height,unsigned(d.MipLevels),unsigned(d.Format),unsigned(d.Flags));
    return true;
}
static void dump_description(const char* tag,const void* desc,SIZE_T bytes) {
    unsigned char buf[kDumpBytes]; if(bytes>kDumpBytes) bytes=kDumpBytes;
    if(!safe_copy(desc,buf,bytes)) { record("PROBE dump: description unreadable\r\n"); return; }
    char line[256];
    for(SIZE_T off=0;off<bytes;off+=8) {
        unsigned long long v; std::memcpy(&v,buf+off,8);
        char what[160]; what[0]=0; describe_resource(v,what,sizeof(what));
        float f0,f1; std::memcpy(&f0,buf+off,4); std::memcpy(&f1,buf+off+4,4);
        std::snprintf(line,sizeof(line),"PROBE %s +0x%03zx 0x%016llx  u32(%u,%u) f32(%g,%g)%s%s\r\n",tag,size_t(off),v,unsigned(v&0xffffffffull),unsigned(v>>32),f0,f1,what[0]?"  <= ":"",what);
        record(line);
    }
}
static double qpc_ms_local(LONGLONG a,LONGLONG b) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return double(b-a)*1000.0/double(f.QuadPart); }
// ================= pixel capture of colour / depth / motion vectors (armed by F9; one shot) =================
// FSR3 dispatch description layout observed in-game (see rdna2/diagnostics/2026-09-21/second-fsr3-probe-0x700.log):
// +0x00 commandList; resource entries every 0xB0 bytes starting at +0x08 {ptr, description(32 B), state u32 at +0x28}: colour +0x08, depth +0xb8,
// motion vectors +0x168. Scalars: jitter +0x6e8 (2 floats), motion-vector scale +0x6f0 (2 floats), render size +0x6f8 (2 u32).
enum { CAP_IDLE=0, CAP_ARMED=1, CAP_RECORDING=2, CAP_RECORDED=3, CAP_DONE=4, CAP_FAILED=5 };
static volatile LONG cap_state=CAP_IDLE;
struct CapTarget {
    const char* name=""; UINT64 entry=0; UINT need_format=0;
    ID3D12Resource* rb=nullptr; D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT rows=0; UINT64 row_bytes=0, total=0;
    UINT width=0, height=0, format=0, ffx_state=0;
};
static CapTarget cap_targets[3];
static unsigned char cap_desc[0x780];
static wchar_t cap_base[32768];
static ID3D12Fence* cap_fence=nullptr; static HANDLE cap_event=nullptr; static UINT64 cap_fence_value=0; static int cap_present_count=0, cap_retries=0;
static LONGLONG cap_t_armed=0, cap_t_record=0;

static bool ffx_to_d3d12_state(UINT s,D3D12_RESOURCE_STATES* out) {
    switch(s) { case 1: *out=D3D12_RESOURCE_STATE_COMMON; return true; case 2: *out=D3D12_RESOURCE_STATE_UNORDERED_ACCESS; return true;
        case 4: *out=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; return true; case 8: *out=D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE; return true;
        case 12: *out=D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); return true;
        case 16: *out=D3D12_RESOURCE_STATE_COPY_SOURCE; return true; default: return false; }
}
static void cap_fail(const char* why) { char l[200]; std::snprintf(l,sizeof(l),"UPSCALER_CAPTURE_FAILED %s\r\n",why); record(l); InterlockedExchange(&cap_state,CAP_FAILED); }
extern "C" __declspec(dllexport) BOOL WINAPI RebuildArmUpscalerCapture(const wchar_t* base_path) {
    if(!base_path) return FALSE;
    if(InterlockedCompareExchange(&cap_state,CAP_ARMED,CAP_IDLE)!=CAP_IDLE) return FALSE;
    wcsncpy_s(cap_base,base_path,_TRUNCATE);
    size_t n=wcslen(cap_base); if(n>4 && !_wcsicmp(cap_base+n-4,L".ppm")) cap_base[n-4]=0;
    LARGE_INTEGER t; QueryPerformanceCounter(&t); cap_t_armed=t.QuadPart;
    record("UPSCALER_CAPTURE armed: will record copies at the next FSR3 dispatch\r\n");
    return TRUE;
}
// Runs on the game's render thread inside the detour, BEFORE the original dispatch records its own commands.
static void cap_on_dispatch(void* cmdlist_ptr,const void* desc) {
    if(cap_state!=CAP_ARMED || InterlockedCompareExchange(&cap_state,CAP_RECORDING,CAP_ARMED)!=CAP_ARMED) return;
    LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
    if(!safe_copy(desc,cap_desc,sizeof(cap_desc))) { cap_fail("description unreadable"); return; }
    // the command list must be a real D3D12 graphics command list (vtable in a d3d12 module)
    unsigned long long vt=0; char mod[64];
    if(!safe_read_u64(cmdlist_ptr,&vt) || !module_is_d3d12(reinterpret_cast<void*>(vt),mod,sizeof(mod))) { cap_fail("command list vtable not in d3d12"); return; }
    ID3D12GraphicsCommandList* list=nullptr;
    if(FAILED(reinterpret_cast<IUnknown*>(cmdlist_ptr)->QueryInterface(__uuidof(ID3D12GraphicsCommandList),reinterpret_cast<void**>(&list))) || !list) { cap_fail("not a graphics command list"); return; }
    cap_targets[0].name="color"; cap_targets[0].entry=0x08;  cap_targets[0].need_format=DXGI_FORMAT_R16G16B16A16_FLOAT;
    cap_targets[1].name="depth"; cap_targets[1].entry=0xb8;  cap_targets[1].need_format=DXGI_FORMAT_R32G8X24_TYPELESS;
    cap_targets[2].name="mvec";  cap_targets[2].entry=0x168; cap_targets[2].need_format=DXGI_FORMAT_R16G16B16A16_FLOAT;
    ID3D12Resource* res[3]{}; D3D12_RESOURCE_STATES st[3]{}; ID3D12Device* device=nullptr; bool ok=true;
    for(int i=0;i<3 && ok;++i) {
        CapTarget& T=cap_targets[i]; unsigned long long p=0; UINT ffxs=0;
        std::memcpy(&p,cap_desc+T.entry,8); std::memcpy(&ffxs,cap_desc+T.entry+0x28,4); T.ffx_state=ffxs;
        char what[160]; if(!describe_resource(p,what,sizeof(what))) { cap_fail("resource slot is not a D3D12 resource"); ok=false; break; }
        if(FAILED(reinterpret_cast<IUnknown*>(p)->QueryInterface(__uuidof(ID3D12Resource),reinterpret_cast<void**>(&res[i]))) || !res[i]) { cap_fail("QI resource"); ok=false; break; }
        D3D12_RESOURCE_DESC d=res[i]->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count!=1 || d.DepthOrArraySize!=1 || d.MipLevels!=1 || UINT(d.Format)!=T.need_format) { cap_fail("resource shape/format not the observed one"); ok=false; break; }
        if(!ffx_to_d3d12_state(ffxs,&st[i])) { cap_fail("resource state not mappable"); ok=false; break; }
        T.width=UINT(d.Width); T.height=d.Height; T.format=UINT(d.Format);
        if(!device) res[i]->GetDevice(__uuidof(ID3D12Device),reinterpret_cast<void**>(&device));
        device->GetCopyableFootprints(&d,0,1,0,&T.layout,&T.rows,&T.row_bytes,&T.total);
        if(!T.total || T.total>128ull*1024*1024) { cap_fail("footprint size out of range"); ok=false; break; }
    }
    if(ok) {
        for(int i=0;i<3 && ok;++i) {    // allocate all readback buffers before recording anything
            CapTarget& T=cap_targets[i];
            D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC b{}; b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; b.Width=T.total; b.Height=1; b.DepthOrArraySize=1; b.MipLevels=1; b.SampleDesc.Count=1; b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if(FAILED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,__uuidof(ID3D12Resource),reinterpret_cast<void**>(&T.rb)))) { cap_fail("readback allocation"); ok=false; break; }
            void* m=nullptr; D3D12_RANGE none{0,0};                       // sentinel fill: proves later that the GPU actually wrote the copy
            if(SUCCEEDED(T.rb->Map(0,&none,&m))) { std::memset(m,0xCD,size_t(T.total)); D3D12_RANGE all{0,size_t(T.total)}; T.rb->Unmap(0,&all); }
        }
    }
    if(ok) {
        for(int i=0;i<3;++i) {
            CapTarget& T=cap_targets[i];
            D3D12_RESOURCE_BARRIER bar{}; bar.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bar.Transition.pResource=res[i]; bar.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            bar.Transition.StateBefore=st[i]; bar.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE; list->ResourceBarrier(1,&bar);
            D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource=res[i]; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex=0;
            D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource=T.rb; dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=T.layout;
            list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            bar.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE; bar.Transition.StateAfter=st[i]; list->ResourceBarrier(1,&bar);
        }
        LARGE_INTEGER t1; QueryPerformanceCounter(&t1); cap_t_record=t1.QuadPart;
        char l[200]; std::snprintf(l,sizeof(l),"UPSCALER_CAPTURE recorded copies of colour/depth/mvec into the game's command list (record cpu=%.3f ms)\r\n",qpc_ms_local(t0.QuadPart,t1.QuadPart)); record(l);
        cap_present_count=0; cap_retries=0; InterlockedExchange(&cap_state,CAP_RECORDED);
    }
    for(int i=0;i<3;++i) if(res[i]) res[i]->Release();
    if(device) device->Release();
    list->Release();
}
static bool cap_write_file(const wchar_t* path,const void* data,size_t n) {
    HANDLE f=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr); if(f==INVALID_HANDLE_VALUE) return false;
    DWORD w=0; bool ok=WriteFile(f,data,DWORD(n),&w,nullptr) && w==n; CloseHandle(f); return ok;
}
// Runs on the Present thread. Needs the swapchain's bound DIRECT queue.
extern "C" __declspec(dllexport) void WINAPI RebuildUpscalerCaptureTick(ID3D12CommandQueue* queue) {
    if(cap_state!=CAP_RECORDED || !queue) return;
    if(++cap_present_count<2) return;                       // give the game time to submit the recorded list
    LARGE_INTEGER tw0; QueryPerformanceCounter(&tw0);
    if(!cap_fence) {
        ID3D12Device* dev=nullptr; if(FAILED(queue->GetDevice(__uuidof(ID3D12Device),reinterpret_cast<void**>(&dev)))) { cap_fail("queue device"); return; }
        HRESULT hr=dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),reinterpret_cast<void**>(&cap_fence)); dev->Release();
        cap_event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(FAILED(hr)||!cap_event) { cap_fail("fence creation"); return; }
    }
    UINT64 v=++cap_fence_value;
    if(FAILED(queue->Signal(cap_fence,v))) { cap_fail("queue signal"); return; }
    if(cap_fence->GetCompletedValue()<v) {
        if(FAILED(cap_fence->SetEventOnCompletion(v,cap_event))) { cap_fail("SetEventOnCompletion"); return; }
        if(WaitForSingleObject(cap_event,1500)!=WAIT_OBJECT_0) { cap_fail("fence wait timed out (GPU objects retained)"); return; }
    }
    // sentinel check on the colour buffer: still all 0xCD means the game has not executed the recorded list yet
    CapTarget& C=cap_targets[0]; void* m=nullptr; D3D12_RANGE all{0,size_t(C.total)};
    if(FAILED(C.rb->Map(0,&all,&m))) { cap_fail("map colour"); return; }
    bool untouched=true; const unsigned char* b=static_cast<const unsigned char*>(m);
    for(size_t i=0;i<size_t(C.total) && untouched;i+=64) if(b[i]!=0xCD) untouched=false;
    D3D12_RANGE none{0,0}; C.rb->Unmap(0,&none);
    if(untouched) { if(++cap_retries>8) { cap_fail("recorded list never executed"); return; } cap_present_count=1; return; }
    LARGE_INTEGER tw1; QueryPerformanceCounter(&tw1);
    // write raw planes (row pitch removed) + metadata
    wchar_t path[33000]; size_t bytes_total=0; bool ok=true;
    for(int i=0;i<3 && ok;++i) {
        CapTarget& T=cap_targets[i]; void* mm=nullptr; D3D12_RANGE rd{0,size_t(T.total)};
        if(FAILED(T.rb->Map(0,&rd,&mm))) { ok=false; break; }
        std::vector<unsigned char> plane(size_t(T.row_bytes)*T.rows);
        for(UINT y=0;y<T.rows;++y) std::memcpy(plane.data()+size_t(y)*size_t(T.row_bytes),static_cast<unsigned char*>(mm)+T.layout.Offset+size_t(y)*T.layout.Footprint.RowPitch,size_t(T.row_bytes));
        T.rb->Unmap(0,&none);
        const wchar_t* suffix=i==0?L".color_rgba16f.bin":i==1?L".depth_plane0_r32.bin":L".mvec_rgba16f.bin";
        std::swprintf(path,33000,L"%ls%ls",cap_base,suffix);
        ok=cap_write_file(path,plane.data(),plane.size()); bytes_total+=plane.size();
    }
    // metadata text: dims, formats, states, jitter/mv-scale/render-size (layout observed in game), then the raw description hex
    std::string meta; char line[256];
    for(int i=0;i<3;++i) { CapTarget& T=cap_targets[i]; std::snprintf(line,sizeof(line),"%s: %ux%u dxgi_format=%u row_bytes=%llu ffx_state=%u\n",T.name,T.width,T.height,T.format,(unsigned long long)T.row_bytes,T.ffx_state); meta+=line; }
    float f[6]; unsigned u[2]; std::memcpy(f,cap_desc+0x6e8,16); std::memcpy(u,cap_desc+0x6f8,8);
    std::snprintf(line,sizeof(line),"jitter_offset(+0x6e8)=%g,%g\nmotion_vector_scale(+0x6f0)=%g,%g\nrender_size(+0x6f8)=%u,%u\n",f[0],f[1],f[2],f[3],u[0],u[1]); meta+=line;
    meta+="description_hex(+0x000..+0x77f):\n";
    for(size_t off=0;off<sizeof(cap_desc);off+=16) { std::snprintf(line,sizeof(line),"+%03zx ",off); meta+=line; for(int k=0;k<16;++k) { std::snprintf(line,sizeof(line),"%02x",cap_desc[off+k]); meta+=line; } meta+="\n"; }
    std::swprintf(path,33000,L"%ls.upscaler_meta.txt",cap_base); ok=cap_write_file(path,meta.data(),meta.size()) && ok;
    LARGE_INTEGER tw2; QueryPerformanceCounter(&tw2);
    for(int i=0;i<3;++i) if(cap_targets[i].rb) { cap_targets[i].rb->Release(); cap_targets[i].rb=nullptr; }
    char l[300]; std::snprintf(l,sizeof(l),"UPSCALER_CAPTURE %s: %zu bytes written; TIMING wait_fence+sentinel=%.2f ms, map+write=%.2f ms, since_arm=%.1f ms\r\n",ok?"SAVED":"WRITE_FAILED",bytes_total,qpc_ms_local(tw0.QuadPart,tw1.QuadPart),qpc_ms_local(tw1.QuadPart,tw2.QuadPart),qpc_ms_local(cap_t_armed,tw2.QuadPart));
    record(l); InterlockedExchange(&cap_state,ok?CAP_DONE:CAP_FAILED);
}
static int __cdecl detour_fsr(void* ctx,const void* desc) {
    LONG n=InterlockedIncrement(&fsr_calls);
    if(cap_state==CAP_ARMED) cap_on_dispatch(*reinterpret_cast<void* const*>(desc),desc);   // desc+0x00 = commandList
    if(n<=kDetailedCalls) { char l[96]; std::snprintf(l,sizeof(l),"PROBE ffxFsr3UpscalerContextDispatch call #%ld ctx=%p desc=%p\r\n",n,ctx,desc); record(l); dump_description("fsr3",desc,kDumpBytes); }
    else if(n%600==0) { char l[96]; std::snprintf(l,sizeof(l),"PROBE ffxFsr3UpscalerContextDispatch calls=%ld\r\n",n); record(l); }
    return orig_fsr(ctx,desc);
}
static unsigned __cdecl detour_ffx(void* ctx,const void* header) {
    LONG n=InterlockedIncrement(&ffx_calls);
    if(n<=kDetailedCalls) {
        char l[120]; std::snprintf(l,sizeof(l),"PROBE ffxDispatch call #%ld ctx=%p header=%p\r\n",n,ctx,header); record(l);
        const void* h=header;
        for(int depth=0;h&&depth<8;++depth) {
            unsigned long long type=0,next=0; if(!safe_read_u64(h,&type)||!safe_read_u64(reinterpret_cast<const char*>(h)+8,&next)) break;
            std::snprintf(l,sizeof(l),"PROBE ffxDispatch chain[%d] type=0x%llx\r\n",depth,type); record(l);
            if(depth==0) dump_description("ffx",h,kDumpBytes);
            h=reinterpret_cast<const void*>(next);
        }
    } else if(n%600==0) { char l[96]; std::snprintf(l,sizeof(l),"PROBE ffxDispatch calls=%ld\r\n",n); record(l); }
    return orig_ffx(ctx,header);
}
// Called periodically from the Present hook; installs each detour once, when its DLL is present in the process.
static void maybe_install() {
    if(!fsr_hooked) {
        if(GetModuleHandleW(L"ffx_fsr3upscaler_x64.dll")) {
            fsr_hooked=true;
            MH_STATUS s=MH_CreateHookApi(L"ffx_fsr3upscaler_x64.dll","ffxFsr3UpscalerContextDispatch",reinterpret_cast<void*>(&detour_fsr),reinterpret_cast<void**>(&orig_fsr));
            if(s==MH_OK) s=MH_EnableHook(MH_ALL_HOOKS);
            char l[96]; std::snprintf(l,sizeof(l),"PROBE hook ffxFsr3UpscalerContextDispatch status=%d\r\n",int(s)); record(l);
        }
    }
    if(!ffx_hooked) {
        HMODULE m=GetModuleHandleW(L"amd_fidelityfx_upscaler_dx12.dll"); if(!m) m=GetModuleHandleW(L"amd_fidelityfx_dx12.dll");
        if(m) {
            ffx_hooked=true;
            const wchar_t* dll=GetModuleHandleW(L"amd_fidelityfx_upscaler_dx12.dll")?L"amd_fidelityfx_upscaler_dx12.dll":L"amd_fidelityfx_dx12.dll";
            MH_STATUS s=MH_CreateHookApi(dll,"ffxDispatch",reinterpret_cast<void*>(&detour_ffx),reinterpret_cast<void**>(&orig_ffx));
            if(s==MH_OK) s=MH_EnableHook(MH_ALL_HOOKS);
            char l[96]; std::snprintf(l,sizeof(l),"PROBE hook ffxDispatch status=%d\r\n",int(s)); record(l);
        }
    }
}
}
