#pragma once
#include "dxgi_probe.h"
// Private data follows the swapchain lifetime; no raw-pointer cache ownership.
inline const GUID queue_key={0x3d6b7932,0x371d,0x4a9f,{0x9d,0xa4,0x8a,0x29,0x5b,0x72,0x91,0x32}};
inline const GUID ambiguous_key={0x3d6b7933,0x371d,0x4a9f,{0x9d,0xa4,0x8a,0x29,0x5b,0x72,0x91,0x32}};
inline UINT64 canonical_id(IUnknown* p) {
    ComPtr<IUnknown> canonical;
    return p && SUCCEEDED(p->QueryInterface(IID_PPV_ARGS(&canonical))) ? reinterpret_cast<UINT64>(canonical.Get()) : 0;
}
inline bool ambiguous_queue(IDXGISwapChain* swap) {
    UINT value=0,size=sizeof(value);
    return SUCCEEDED(swap->GetPrivateData(ambiguous_key,&size,&value)) && value!=0;
}
inline void invalidate_queue(IDXGISwapChain* swap) {
    UINT value=1; swap->SetPrivateData(ambiguous_key,sizeof(value),&value);
    swap->SetPrivateDataInterface(queue_key,nullptr);
}
inline ComPtr<ID3D12CommandQueue> bound_queue(IDXGISwapChain* swap) {
    ComPtr<ID3D12CommandQueue> result;
    if(ambiguous_queue(swap)) return result;
    IUnknown* value=nullptr; UINT size=sizeof(value);
    if(SUCCEEDED(swap->GetPrivateData(queue_key,&size,&value)) && value) {
        value->QueryInterface(IID_PPV_ARGS(&result)); value->Release();
    }
    return result;
}
inline bool bind_queue(IDXGISwapChain* swap,IUnknown* candidate) {
    if(ambiguous_queue(swap)) return false;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Device> a,b;
    if(!candidate || FAILED(candidate->QueryInterface(IID_PPV_ARGS(&queue))) ||
       queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT ||
       FAILED(queue->GetDevice(IID_PPV_ARGS(&a))) || FAILED(swap->GetDevice(IID_PPV_ARGS(&b))) ||
       canonical_id(a.Get())!=canonical_id(b.Get())) { invalidate_queue(swap); return false; }
    auto prior=bound_queue(swap);
    if(prior && canonical_id(prior.Get())!=canonical_id(queue.Get())) { invalidate_queue(swap); return false; }
    return SUCCEEDED(swap->SetPrivateDataInterface(queue_key,queue.Get()));
}
