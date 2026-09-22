#include "dxgi_probe.h"
#include "frame_metadata.h"
#include "queue_binding.h"
#include <fstream>
#include <cstdio>
#include <atomic>
#include <thread>
#include <vector>
#include <cstdlib>
static void require(bool ok,const char* message) { if(!ok) { std::printf("FAIL %s\n",message); ExitProcess(1); } }
int main(int argc,char** argv) {
    require(argc==2 || (argc==3 && std::string(argv[2])=="--amd"),"DLL argument and optional --amd");
    const bool hardware=argc==3;
    HMODULE module=LoadLibraryA(argv[1]); require(module!=nullptr,"load DLL");
    auto start=reinterpret_cast<HRESULT(WINAPI*)(const wchar_t*)>(GetProcAddress(module,"RebuildStart"));
    auto count=reinterpret_cast<LONG64(WINAPI*)()>(GetProcAddress(module,"RebuildPresentCount"));
    auto count1=reinterpret_cast<LONG64(WINAPI*)()>(GetProcAddress(module,"RebuildPresent1Count"));
    auto metadata=reinterpret_cast<BOOL(WINAPI*)(IUnknown*,FrameMetadata*)>(GetProcAddress(module,"RebuildGetMetadata"));
    auto capture=reinterpret_cast<HRESULT(WINAPI*)(IDXGISwapChain*,const wchar_t*)>(GetProcAddress(module,"RebuildCaptureFrame"));
    require(start&&count&&count1&&metadata&&capture,"exports");
    require(count()==0&&count1()==0,"inert before explicit start");
    DxgiProbe probe; require(SUCCEEDED(probe.create(hardware)),"D3D12 swapchain");
    DXGI_ADAPTER_DESC adapter_desc{}; probe.adapter->GetDesc(&adapter_desc);
    std::printf("Adapter vendor=0x%x device=0x%x hardware=%u\n",adapter_desc.VendorId,adapter_desc.DeviceId,UINT(hardware));
    std::atomic<bool> running{true};
    std::vector<std::thread> workers;
    for(int i=0;i<8;++i) workers.emplace_back([&] {
        while(running.load()) { void* p=HeapAlloc(GetProcessHeap(),0,4096); if(p) HeapFree(GetProcessHeap(),0,p); Sleep(0); }
    });
    HRESULT hr=start(L"rebuild-hook-test.log");
    running=false; for(auto& worker:workers) worker.join();
    require(SUCCEEDED(hr),"initialize while allocator workers active");
    require(FAILED(start(L"rebuild-hook-test.log")),"duplicate initialization rejected");
    hr=probe.swap->Present(0,DXGI_PRESENT_TEST);
    require(SUCCEEDED(hr),"Present forwarding HRESULT");
    DXGI_PRESENT_PARAMETERS params{};
    hr=probe.swap->Present1(0,DXGI_PRESENT_TEST,&params);
    require(SUCCEEDED(hr),"Present1 forwarding HRESULT");
    require(count()>=1&&count1()>=1,"both actual DXGI calls observed");
    FrameMetadata frame{};
    require(metadata(probe.swap.Get(),&frame) && frame.queue==0 && frame.valid_buffer,"preexisting swapchain: queue remains unknown");
    require(capture(probe.swap.Get(),L"must-not-capture.ppm")==E_ACCESSDENIED,"unknown queue blocks GPU submission");
    DxgiProbe created; require(SUCCEEDED(created.create(hardware)),"post-hook swapchain creation");
    ComPtr<IUnknown> queue_identity,device_identity;
    created.queue.As(&queue_identity); created.device.As(&device_identity);
    require(metadata(created.swap.Get(),&frame),"creation metadata");
    require(frame.queue==reinterpret_cast<UINT64>(queue_identity.Get()),"exact queue association");
    require(frame.device==reinterpret_cast<UINT64>(device_identity.Get()),"exact device association");
    require(frame.width==64 && frame.height==64 && frame.buffer_count==2 && frame.format==DXGI_FORMAT_R8G8B8A8_UNORM && frame.valid_buffer,"backbuffer metadata");
    require(SUCCEEDED(created.swap->ResizeBuffers(2,96,80,DXGI_FORMAT_R8G8B8A8_UNORM,0)),"resize without retained resource references");
    require(SUCCEEDED(created.swap->Present(0,DXGI_PRESENT_TEST)),"present after resize");
    require(metadata(created.swap.Get(),&frame) && frame.width==96 && frame.height==80,"refreshed resize metadata");
    require(frame.queue==reinterpret_cast<UINT64>(queue_identity.Get()),"queue association survives resize");
    ComPtr<IDXGISwapChain3> swap3; created.swap.As(&swap3);
    ComPtr<ID3D12Resource> backbuffer;
    require(SUCCEEDED(created.swap->GetBuffer(swap3->GetCurrentBackBufferIndex(),IID_PPV_ARGS(&backbuffer))),"test backbuffer");
    ComPtr<ID3D12DescriptorHeap> rtv;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{}; heap_desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap_desc.NumDescriptors=1;
    require(SUCCEEDED(created.device->CreateDescriptorHeap(&heap_desc,IID_PPV_ARGS(&rtv))),"RTV heap");
    auto rtv_handle=rtv->GetCPUDescriptorHandleForHeapStart();
    created.device->CreateRenderTargetView(backbuffer.Get(),nullptr,rtv_handle);
    ComPtr<ID3D12CommandAllocator> clear_allocator; ComPtr<ID3D12GraphicsCommandList> clear_commands;
    require(SUCCEEDED(created.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&clear_allocator))),"clear allocator");
    require(SUCCEEDED(created.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,clear_allocator.Get(),nullptr,IID_PPV_ARGS(&clear_commands))),"clear list");
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={backbuffer.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_RENDER_TARGET};
    clear_commands->ResourceBarrier(1,&barrier);
    const FLOAT red[]={1,0,0,1}; clear_commands->ClearRenderTargetView(rtv_handle,red,0,nullptr);
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_RENDER_TARGET; barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_PRESENT;
    clear_commands->ResourceBarrier(1,&barrier); require(SUCCEEDED(clear_commands->Close()),"close clear list");
    ID3D12CommandList* clear_lists[]={clear_commands.Get()}; created.queue->ExecuteCommandLists(1,clear_lists);
    DeleteFileW(L"readback-test.ppm");
    require(SUCCEEDED(capture(created.swap.Get(),L"readback-test.ppm")),"fenced frame capture after queued clear");
    std::ifstream captured("readback-test.ppm",std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(captured)),{});
    const std::string header="P6\n96 80\n255\n";
    require(bytes.size()==header.size()+96*80*3 && bytes.substr(0,header.size())==header,"capture dimensions and packed row length");
    for(size_t i=header.size();i<bytes.size();i+=3)
        require(static_cast<unsigned char>(bytes[i])==255 && bytes[i+1]==0 && bytes[i+2]==0,"exact GPU red pixels");
    backbuffer.Reset();
    require(SUCCEEDED(created.swap->ResizeBuffers(2,64,64,DXGI_FORMAT_R8G8B8A8_UNORM,0)),"capture released backbuffer after fence");
    ComPtr<ID3D12CommandQueue> second_queue; D3D12_COMMAND_QUEUE_DESC queue_desc{};
    require(SUCCEEDED(created.device->CreateCommandQueue(&queue_desc,IID_PPV_ARGS(&second_queue))),"second queue");
    require(!bind_queue(created.swap.Get(),second_queue.Get()) && ambiguous_queue(created.swap.Get()),"conflicting queue invalidates binding");
    require(capture(created.swap.Get(),L"must-not-capture.ppm")==E_ACCESSDENIED,"ambiguous queue blocks GPU submission");
    std::puts("PASS fenced GPU pixel readback, pitch removal, resize, unknown/conflicting queue rejection");
    DXGI_SWAP_CHAIN_DESC1 composition_desc{};
    composition_desc.Width=32; composition_desc.Height=48; composition_desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    composition_desc.SampleDesc.Count=1; composition_desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
    composition_desc.BufferCount=2; composition_desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> composition;
    require(SUCCEEDED(created.factory->CreateSwapChainForComposition(created.queue.Get(),&composition_desc,nullptr,&composition)),"composition creation");
    require(metadata(composition.Get(),&frame) && frame.width==32 && frame.height==48 && frame.queue==reinterpret_cast<UINT64>(queue_identity.Get()),"composition metadata");
    HWND legacy_window=CreateWindowExW(0,L"STATIC",L"Legacy test",WS_OVERLAPPEDWINDOW,0,0,64,64,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
    require(legacy_window!=nullptr,"legacy window");
    DXGI_SWAP_CHAIN_DESC legacy_desc{};
    legacy_desc.BufferDesc.Width=40; legacy_desc.BufferDesc.Height=56;
    legacy_desc.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    legacy_desc.SampleDesc.Count=1; legacy_desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
    legacy_desc.BufferCount=2; legacy_desc.OutputWindow=legacy_window;
    legacy_desc.Windowed=TRUE; legacy_desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain> legacy;
    require(SUCCEEDED(created.factory->CreateSwapChain(created.queue.Get(),&legacy_desc,&legacy)),"legacy creation");
    require(metadata(legacy.Get(),&frame) && frame.width==40 && frame.height==56 && frame.queue==reinterpret_cast<UINT64>(queue_identity.Get()),"legacy metadata");
    legacy.Reset(); DestroyWindow(legacy_window);
    std::puts("PASS exact D3D12 queue/device/backbuffer metadata and resize lifecycle");
    std::printf("PASS D3D12 Present/Present1 test calls observed, allocator workers joined: %lld/%lld\n",count(),count1());
}
