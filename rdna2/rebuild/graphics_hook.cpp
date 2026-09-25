// Diagnostic-only replacement. No vendor runtime, HIP, or neural dispatch.
#include "dxgi_probe.h"
#include "third_party/minhook/include/MinHook.h"
#include <cstdio>
#include <cwchar>
#include "frame_metadata.h"
#include "queue_binding.h"
#include "frame_readback.h"
static SRWLOCK capture_lock=SRWLOCK_INIT;
static FrameReadback* frame_copy=nullptr; // Retain pending GPU objects on timeout.
static wchar_t capture_path[32768]{};
static volatile LONG capture_requested=0;
static thread_local UINT creation_depth=0;
struct CreationScope { CreationScope(){++creation_depth;} ~CreationScope(){--creation_depth;} };
using Resize1Fn=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*,UINT,UINT,UINT,DXGI_FORMAT,UINT,const UINT*,IUnknown* const*);
static Resize1Fn original_resize1=nullptr;
using CreateFn=HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*,IUnknown*,DXGI_SWAP_CHAIN_DESC*,IDXGISwapChain**);
using CreateHwndFn=HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*,IUnknown*,HWND,const DXGI_SWAP_CHAIN_DESC1*,const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,IDXGIOutput*,IDXGISwapChain1**);
using CreateCompositionFn=HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*,IUnknown*,const DXGI_SWAP_CHAIN_DESC1*,IDXGIOutput*,IDXGISwapChain1**);
static CreateFn original_create=nullptr;
static CreateHwndFn original_create_hwnd=nullptr;
static CreateCompositionFn original_create_composition=nullptr;
static SRWLOCK metadata_lock=SRWLOCK_INIT;
static FrameMetadata metadata[32]{};
static UINT metadata_size=0;
using PresentFn=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*,UINT,UINT);
using Present1Fn=HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*,UINT,UINT,const DXGI_PRESENT_PARAMETERS*);
static PresentFn original_present=nullptr;
static Present1Fn original_present1=nullptr;
static volatile LONG64 presents=0,presents1=0;
static volatile LONG state=0;
static HANDLE log_file=INVALID_HANDLE_VALUE;
static void record(const char* text) {
    if(log_file==INVALID_HANDLE_VALUE) return;
    DWORD written; WriteFile(log_file,text,DWORD(lstrlenA(text)),&written,nullptr);
}
#include "upscaler_probe.h"
static LONG probe_tick=0;
static void poll_upscaler_probe() { if((InterlockedIncrement(&probe_tick)%120)==1) upscaler_probe::maybe_install(); }
static UINT64 identity(IUnknown* object) {
    ComPtr<IUnknown> canonical;
    if(!object || FAILED(object->QueryInterface(IID_PPV_ARGS(&canonical)))) return 0;
    return reinterpret_cast<UINT64>(canonical.Get());
}
static void observe_frame(IDXGISwapChain* swap,IUnknown* creation_device=nullptr,const char* origin="present") {
    FrameMetadata item{};
    item.swap=identity(swap);
    if(!item.swap) return;
    ComPtr<ID3D12Device> device;
    if(FAILED(swap->GetDevice(IID_PPV_ARGS(&device)))) return;
    item.device=identity(device.Get());
    if(creation_device) {
        bool accepted=bind_queue(swap,creation_device);
        char line[1400],module_path[1024]{}; HMODULE owner=nullptr;
        ComPtr<ID3D12CommandQueue> candidate_queue;
        if(SUCCEEDED(creation_device->QueryInterface(IID_PPV_ARGS(&candidate_queue)))) {
            void* execute=(*reinterpret_cast<void***>(candidate_queue.Get()))[10];
            if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCSTR>(execute),&owner))
                GetModuleFileNameA(owner,module_path,sizeof(module_path));
        }
        std::snprintf(line,sizeof(line),"QUEUE_CANDIDATE swap=0x%llx candidate=0x%llx source=%s depth=%u accepted=%u ambiguous=%u execute_module=%s\r\n",
            item.swap,identity(creation_device),origin,creation_depth,UINT(accepted),UINT(ambiguous_queue(swap)),module_path);
        record(line);
    }
    auto queue=bound_queue(swap);
    if(queue) { item.queue=identity(queue.Get()); item.queue_type=queue->GetDesc().Type; }
    DXGI_SWAP_CHAIN_DESC description{};
    if(SUCCEEDED(swap->GetDesc(&description))) item.buffer_count=description.BufferCount;
    ComPtr<IDXGISwapChain3> swap3;
    if(SUCCEEDED(swap->QueryInterface(IID_PPV_ARGS(&swap3)))) item.current_index=swap3->GetCurrentBackBufferIndex();
    ComPtr<ID3D12Resource> buffer;
    if(SUCCEEDED(swap->GetBuffer(item.current_index,IID_PPV_ARGS(&buffer)))) {
        auto desc=buffer->GetDesc();
        item.width=desc.Width; item.height=desc.Height; item.format=desc.Format; item.valid_buffer=1;
    }
    // Keep scalar identities only: no retained backbuffer references across ResizeBuffers.
    AcquireSRWLockExclusive(&metadata_lock);
    UINT slot=0;
    while(slot<metadata_size && metadata[slot].swap!=item.swap) ++slot;
    if(slot==32) { ReleaseSRWLockExclusive(&metadata_lock); return; }
    if(slot==metadata_size) ++metadata_size;
    const auto old=metadata[slot];
    const bool changed=old.device!=item.device || old.queue!=item.queue || old.width!=item.width ||
        old.height!=item.height || old.format!=item.format || old.buffer_count!=item.buffer_count || old.valid_buffer!=item.valid_buffer;
    metadata[slot]=item;
    ReleaseSRWLockExclusive(&metadata_lock);
    if(changed) {
        char line[512];
        std::snprintf(line,sizeof(line),"FRAME_METADATA swap=0x%llx queue=0x%llx device=0x%llx width=%llu height=%u format=%u buffers=%u queue_type=%u valid_buffer=%u\r\n",
            item.swap,item.queue,item.device,item.width,item.height,item.format,item.buffer_count,item.queue_type,item.valid_buffer);
        record(line);
    }
}
extern "C" __declspec(dllexport) BOOL WINAPI RebuildGetMetadata(IUnknown* swap,FrameMetadata* out) {
    if(!swap || !out) return FALSE;
    UINT64 key=identity(swap); BOOL found=FALSE;
    AcquireSRWLockShared(&metadata_lock);
    for(UINT i=0;i<metadata_size;++i) if(metadata[i].swap==key) { *out=metadata[i]; found=TRUE; break; }
    ReleaseSRWLockShared(&metadata_lock); return found;
}
static HRESULT STDMETHODCALLTYPE create_swap(IDXGIFactory* factory,IUnknown* device,DXGI_SWAP_CHAIN_DESC* desc,IDXGISwapChain** out) {
    CreationScope scope;
    HRESULT hr=original_create(factory,device,desc,out);
    if(SUCCEEDED(hr) && out && *out) observe_frame(*out,device,"CreateSwapChain");
    return hr;
}
static HRESULT STDMETHODCALLTYPE create_hwnd(IDXGIFactory2* factory,IUnknown* device,HWND window,const DXGI_SWAP_CHAIN_DESC1* desc,const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,IDXGIOutput* restrict_output,IDXGISwapChain1** out) {
    CreationScope scope;
    HRESULT hr=original_create_hwnd(factory,device,window,desc,fullscreen,restrict_output,out);
    if(SUCCEEDED(hr) && out && *out) observe_frame(*out,device,"CreateSwapChainForHwnd");
    return hr;
}
static HRESULT STDMETHODCALLTYPE create_composition(IDXGIFactory2* factory,IUnknown* device,const DXGI_SWAP_CHAIN_DESC1* desc,IDXGIOutput* restrict_output,IDXGISwapChain1** out) {
    CreationScope scope;
    HRESULT hr=original_create_composition(factory,device,desc,restrict_output,out);
    if(SUCCEEDED(hr) && out && *out) observe_frame(*out,device,"CreateSwapChainForComposition");
    return hr;
}
static HRESULT STDMETHODCALLTYPE resize1(IDXGISwapChain3* swap,UINT count,UINT width,UINT height,DXGI_FORMAT format,UINT flags,const UINT* masks,IUnknown* const* queues) {
    // A per-buffer queue reassignment requires fresh association tracking.
    invalidate_queue(swap);
    record("QUEUE_INVALIDATED ResizeBuffers1; capture blocked\r\n");
    return original_resize1(swap,count,width,height,format,flags,masks,queues);
}
// ---- capture timing (diagnostic only; QueryPerformanceCounter). Per-swapchain present timestamps feed frame-interval statistics.
static SRWLOCK ts_lock=SRWLOCK_INIT;
struct PresentStamp { const void* swap; LONGLONG t; };
static PresentStamp ts_ring[256]; static unsigned ts_count=0;
static LONGLONG capture_mark=0; static const void* capture_mark_swap=nullptr;
static LONGLONG qpc_now() { LARGE_INTEGER t; QueryPerformanceCounter(&t); return t.QuadPart; }
static double qpc_ms(LONGLONG a,LONGLONG b) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return double(b-a)*1000.0/double(f.QuadPart); }
static void note_present(IDXGISwapChain* swap) {
    LONGLONG t=qpc_now(), mark=0; const void* mswap=nullptr;
    AcquireSRWLockExclusive(&ts_lock);
    ts_ring[ts_count%256]={swap,t}; ++ts_count;
    if(capture_mark && capture_mark_swap==swap) { mark=capture_mark; mswap=capture_mark_swap; capture_mark=0; }
    ReleaseSRWLockExclusive(&ts_lock);
    if(mark) { char line[160]; std::snprintf(line,sizeof(line),"TIMING next_present_after_capture_start=%.2f ms (swap=%p)\r\n",qpc_ms(mark,t),mswap); record(line); }
}
static void log_interval_stats(const void* swap,LONGLONG upto) {
    double sum=0,mx=0,mn=1e9; unsigned n=0; LONGLONG prev=0;
    AcquireSRWLockExclusive(&ts_lock);
    unsigned total=ts_count<256?ts_count:256, first=ts_count-total;
    for(unsigned i=0;i<total;++i) { auto& e=ts_ring[(first+i)%256]; if(e.swap!=swap||e.t>upto) continue;
        if(prev) { double d=qpc_ms(prev,e.t); sum+=d; if(d>mx) mx=d; if(d<mn) mn=d; ++n; } prev=e.t; }
    ReleaseSRWLockExclusive(&ts_lock);
    char line[200];
    if(n) std::snprintf(line,sizeof(line),"TIMING frame_interval_before_capture: n=%u avg=%.2f ms (%.1f fps) min=%.2f max=%.2f\r\n",n,sum/n,1000.0*n/sum,mn,mx);
    else std::snprintf(line,sizeof(line),"TIMING frame_interval_before_capture: not enough presents for this swapchain\r\n");
    record(line);
}
extern "C" __declspec(dllexport) HRESULT WINAPI RebuildCaptureFrame(IDXGISwapChain* swap,const wchar_t* path) {
    if(!swap || !path) return E_INVALIDARG;
    if(!TryAcquireSRWLockExclusive(&capture_lock)) return DXGI_ERROR_WAS_STILL_DRAWING;
    struct Unlock { ~Unlock(){ReleaseSRWLockExclusive(&capture_lock);} } unlock;
    const LONGLONG t0=qpc_now();
    auto queue=bound_queue(swap);
    if(!queue) { record("CAPTURE_BLOCKED queue unknown or ambiguous\r\n"); return E_ACCESSDENIED; }
    if(!frame_copy) frame_copy=new FrameReadback;
    if(frame_copy->pending || frame_copy->poisoned) return E_PENDING;
    ComPtr<IDXGISwapChain3> swap3; ComPtr<ID3D12Resource> buffer;
    HRESULT hr=swap->QueryInterface(IID_PPV_ARGS(&swap3)); if(FAILED(hr)) return hr;
    hr=swap->GetBuffer(swap3->GetCurrentBackBufferIndex(),IID_PPV_ARGS(&buffer)); if(FAILED(hr)) return hr;
    const LONGLONG t_prep=qpc_now();
    hr=frame_copy->submit(queue.Get(),buffer.Get()); if(FAILED(hr)) return hr;
    const LONGLONG t_submit=qpc_now();
    std::vector<unsigned char> pixels;
    hr=frame_copy->finish(pixels);
    const LONGLONG t_finish=qpc_now(); if(FAILED(hr)) { record("CAPTURE_PENDING_OR_FAILED GPU objects retained\r\n"); return hr; }
    auto footprint=frame_copy->layout.Footprint;
    HANDLE file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE) return HRESULT_FROM_WIN32(GetLastError());
    bool ok; LONGLONG t_convert;
    if(footprint.Format==DXGI_FORMAT_R16G16B16A16_FLOAT) {
        // HDR backbuffer (no upscaler dispatch to hook when the upscaler is off): write the raw
        // 8 B/pixel RGBA16F plane, row pitch stripped, same layout as the F9 upscaler-input capture
        // (upscaler_probe.h's .color_rgba16f.bin) and what net_frame_full.py's k_import expects -
        // no PPM conversion, this is not a preview.
        const SIZE_T row_bytes=SIZE_T(footprint.Width)*8;
        std::vector<unsigned char> plane(row_bytes*footprint.Height);
        for(UINT y=0;y<footprint.Height;++y) memcpy(plane.data()+SIZE_T(y)*row_bytes,pixels.data()+SIZE_T(y)*frame_copy->row_bytes,row_bytes);
        t_convert=qpc_now();
        DWORD written=0;
        ok=WriteFile(file,plane.data(),DWORD(plane.size()),&written,nullptr) && written==plane.size();
    } else {
        // Portable RGB preview; R10 packed channels are scaled without tone mapping.
        std::vector<unsigned char> rgb(SIZE_T(footprint.Width)*footprint.Height*3);
        for(SIZE_T i=0;i<SIZE_T(footprint.Width)*footprint.Height;++i) {
            if(footprint.Format==DXGI_FORMAT_R10G10B10A2_UNORM) {
                UINT packed; memcpy(&packed,pixels.data()+4*i,4);
                for(UINT c=0;c<3;++c) rgb[3*i+c]=static_cast<unsigned char>(((packed>>(10*c))&1023)*255/1023);
            } else {
                bool bgra=footprint.Format==DXGI_FORMAT_B8G8R8A8_UNORM;
                rgb[3*i]=pixels[4*i+(bgra?2:0)]; rgb[3*i+1]=pixels[4*i+1]; rgb[3*i+2]=pixels[4*i+(bgra?0:2)];
            }
        }
        t_convert=qpc_now();
        char header[80]; int length=std::snprintf(header,sizeof(header),"P6\n%u %u\n255\n",footprint.Width,footprint.Height);
        DWORD written=0;
        ok=WriteFile(file,header,length,&written,nullptr) && written==UINT(length);
        if(ok) ok=WriteFile(file,rgb.data(),DWORD(rgb.size()),&written,nullptr) && written==rgb.size();
    }
    CloseHandle(file);
    const LONGLONG t_end=qpc_now();
    {
        char line[320]; std::snprintf(line,sizeof(line),"TIMING capture: prep=%.2f submit=%.2f gpu_queue_copy_wait+map=%.2f convert_rgb=%.2f write_file=%.2f TOTAL=%.2f ms (frame %ux%u)\r\n",
            qpc_ms(t0,t_prep),qpc_ms(t_prep,t_submit),qpc_ms(t_submit,t_finish),qpc_ms(t_finish,t_convert),qpc_ms(t_convert,t_end),qpc_ms(t0,t_end),frame_copy->layout.Footprint.Width,frame_copy->layout.Footprint.Height);
        record(line);
        log_interval_stats(swap,t0);
        AcquireSRWLockExclusive(&ts_lock); capture_mark=t0; capture_mark_swap=swap; ReleaseSRWLockExclusive(&ts_lock);
    }
    record(ok?"FRAME_CAPTURE_SAVED fence completed; PRESENT state restored\r\n":"FRAME_CAPTURE_WRITE_FAILED\r\n");
    return ok?S_OK:E_FAIL;
}
static void maybe_capture(IDXGISwapChain* swap,UINT flags) {
    if(!capture_path[0] || (flags&DXGI_PRESENT_TEST)) return;
    if(upscaler_probe::cap_state==upscaler_probe::CAP_RECORDED) { auto q=bound_queue(swap); if(q) upscaler_probe::RebuildUpscalerCaptureTick(q.Get()); }
    if((GetAsyncKeyState(VK_F9)&0x8000) && upscaler_probe::cap_state==upscaler_probe::CAP_IDLE) upscaler_probe::RebuildArmUpscalerCapture(capture_path);   // F9 = upscaler pixel capture (F8 = backbuffer only)
    if((GetAsyncKeyState(VK_F8)&0x8000) && InterlockedCompareExchange(&capture_requested,1,0)==0) {
        HRESULT hr=RebuildCaptureFrame(swap,capture_path);
        char line[96]; std::snprintf(line,sizeof(line),"CAPTURE_RESULT hr=0x%08lx\r\n",static_cast<unsigned long>(hr)); record(line);
    }
}
static HRESULT STDMETHODCALLTYPE present(IDXGISwapChain* swap,UINT sync,UINT flags) {
    observe_frame(swap);
    if(!(flags&DXGI_PRESENT_TEST)) { note_present(swap); poll_upscaler_probe(); }
    maybe_capture(swap,flags);
    if(InterlockedIncrement64(&presents)==1) record("Observed Present; forwarding unchanged\r\n");
    return original_present(swap,sync,flags);
}
static HRESULT STDMETHODCALLTYPE present1(IDXGISwapChain1* swap,UINT sync,UINT flags,const DXGI_PRESENT_PARAMETERS* params) {
    observe_frame(swap);
    if(!(flags&DXGI_PRESENT_TEST)) { note_present(swap); poll_upscaler_probe(); }
    maybe_capture(swap,flags);
    if(InterlockedIncrement64(&presents1)==1) record("Observed Present1; forwarding unchanged\r\n");
    return original_present1(swap,sync,flags,params);
}
extern "C" __declspec(dllexport) void WINAPI RebuildProbeUpscalers() { upscaler_probe::maybe_install(); }   // test hook
extern "C" __declspec(dllexport) LONG64 WINAPI RebuildPresentCount() {
    return InterlockedCompareExchange64(&presents,0,0);
}
extern "C" __declspec(dllexport) LONG64 WINAPI RebuildPresent1Count() {
    return InterlockedCompareExchange64(&presents1,0,0);
}
// Explicit initialization outside DllMain. The module stays loaded for process
// lifetime; live unloading/removing hooks is deliberately unsupported.
extern "C" __declspec(dllexport) HRESULT WINAPI RebuildStart(const wchar_t* log_path) {
    if(!log_path) return E_INVALIDARG;
    if(InterlockedCompareExchange(&state,1,0)!=0) return HRESULT_FROM_WIN32(ERROR_ALREADY_INITIALIZED);
    log_file=CreateFileW(log_path,FILE_APPEND_DATA,FILE_SHARE_READ,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(log_file==INVALID_HANDLE_VALUE) { InterlockedExchange(&state,-1); return HRESULT_FROM_WIN32(GetLastError()); }
    // Pin code before installing callbacks; a caller must not unload live detours.
    HMODULE pinned=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                          reinterpret_cast<LPCWSTR>(&RebuildStart),&pinned)) return E_FAIL;
    DxgiProbe probe;
    HRESULT hr=probe.create();
    if(FAILED(hr)) { record("STOP: WARP probe creation failed\r\n"); return hr; }
    void** table=*reinterpret_cast<void***>(probe.swap.Get());
    void** factory_table=*reinterpret_cast<void***>(probe.factory.Get());
    ComPtr<IDXGISwapChain3> probe3; probe.swap.As(&probe3);
    void** table3=*reinterpret_cast<void***>(probe3.Get());
    MH_STATUS status=MH_Initialize();
    if(status!=MH_OK) { record("STOP: hook initialization failed\r\n"); return E_FAIL; }
    status=MH_CreateHook(table[8],reinterpret_cast<void*>(&present),reinterpret_cast<void**>(&original_present));
    if(status==MH_OK) status=MH_CreateHook(table[22],reinterpret_cast<void*>(&present1),reinterpret_cast<void**>(&original_present1));
    if(status==MH_OK) status=MH_CreateHook(factory_table[10],reinterpret_cast<void*>(&create_swap),reinterpret_cast<void**>(&original_create));
    if(status==MH_OK) status=MH_CreateHook(factory_table[15],reinterpret_cast<void*>(&create_hwnd),reinterpret_cast<void**>(&original_create_hwnd));
    if(status==MH_OK) status=MH_CreateHook(factory_table[24],reinterpret_cast<void*>(&create_composition),reinterpret_cast<void**>(&original_create_composition));
    if(status==MH_OK) status=MH_CreateHook(table3[39],reinterpret_cast<void*>(&resize1),reinterpret_cast<void**>(&original_resize1));
    if(status!=MH_OK) { MH_Uninitialize(); record("STOP: hook preparation failed\r\n"); return E_FAIL; }
    MH_QueueEnableHook(table[8]); MH_QueueEnableHook(table[22]);
    MH_QueueEnableHook(factory_table[10]); MH_QueueEnableHook(factory_table[15]); MH_QueueEnableHook(factory_table[24]);
    MH_QueueEnableHook(table3[39]);
    status=MH_ApplyQueued();
    if(status!=MH_OK) {
        // Preserve callbacks/trampolines if a batch partially succeeded.
        record("STOP: hook activation failed; restart diagnostic process\r\n");
        return E_FAIL;
    }
    InterlockedExchange(&state,2);
    record("DXGI observation hooks active; neural rendering unavailable\r\n");
    return S_OK;
}
static DWORD WINAPI start_game_probe(void*) {
    wchar_t path[32768]{};
    DWORD n=GetEnvironmentVariableW(L"DLSSNR_REBUILD_LOG",path,32768);
    if(!n || n>=32768) return 1;
    if(GetModuleHandleW(L"amdhip64_7.dll")) return 2;
    DWORD frame_length=GetEnvironmentVariableW(L"DLSSNR_REBUILD_FRAME",capture_path,32768);
    if(frame_length>=32768) capture_path[0]=0;
    return FAILED(RebuildStart(path)) ? 3 : 0;
}
BOOL WINAPI DllMain(HINSTANCE instance,DWORD reason,LPVOID) {
    if(reason!=DLL_PROCESS_ATTACH) return TRUE;
    DisableThreadLibraryCalls(instance);
    wchar_t enabled[2]{},path[32768]{};
    if(GetEnvironmentVariableW(L"DLSSNR_REBUILD_PROBE",enabled,2)!=1 || enabled[0]!=L'1') return TRUE;
    DWORD n=GetModuleFileNameW(nullptr,path,32768);
    if(!n || n>=32768) return TRUE;
    const wchar_t* name=wcsrchr(path,L'\\');
    if(_wcsicmp(name ? name+1 : path,L"Cyberpunk2077.exe")) return TRUE;
    HANDLE worker=CreateThread(nullptr,0,start_game_probe,nullptr,0,nullptr);
    if(worker) CloseHandle(worker);
    return TRUE;
}
