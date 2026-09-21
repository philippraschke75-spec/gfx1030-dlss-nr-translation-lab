#pragma once
#include "dxgi_probe.h"
#include <vector>
// One bounded copy. Caller owns queue selection and guarantees PRESENT state.
// Keep this object alive if wait times out; GPU resources remain retained.
struct FrameReadback {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Resource> source,readback;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    HANDLE event=nullptr;
    bool pending=false, poisoned=false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT rows=0; UINT64 row_bytes=0,total=0;
    ~FrameReadback() {
        if(pending || poisoned) {
            // A timed-out submission must never free objects still used by GPU.
            allocator.Detach(); commands.Detach(); source.Detach(); readback.Detach();
            queue.Detach(); fence.Detach();
        } else if(event) CloseHandle(event);
    }
    HRESULT submit(ID3D12CommandQueue* q,ID3D12Resource* buffer) {
        if(!q || !buffer || pending || poisoned) return E_INVALIDARG;
        if(q->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT) return E_INVALIDARG;
        ComPtr<ID3D12Device> device,other;
        HRESULT hr=q->GetDevice(IID_PPV_ARGS(&device)); if(FAILED(hr)) return hr;
        hr=buffer->GetDevice(IID_PPV_ARGS(&other)); if(FAILED(hr)) return hr;
        ComPtr<IUnknown> a,b; device.As(&a); other.As(&b);
        if(a.Get()!=b.Get()) return E_INVALIDARG;
        auto desc=buffer->GetDesc();
        if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count!=1 || desc.DepthOrArraySize!=1 || desc.MipLevels!=1) return E_INVALIDARG;
        if(desc.Format!=DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM && desc.Format!=DXGI_FORMAT_R10G10B10A2_UNORM) return E_INVALIDARG;
        device->GetCopyableFootprints(&desc,0,1,0,&layout,&rows,&row_bytes,&total);
        if(!total || total>256ull*1024*1024) return E_INVALIDARG;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC staging{}; staging.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        staging.Width=total; staging.Height=1; staging.DepthOrArraySize=1;
        staging.MipLevels=1; staging.SampleDesc.Count=1; staging.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&staging,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)); if(FAILED(hr)) return hr;
        hr=device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)); if(FAILED(hr)) return hr;
        hr=device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&commands)); if(FAILED(hr)) return hr;
        hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)); if(FAILED(hr)) return hr;
        if(event) CloseHandle(event);
        event=CreateEventW(nullptr,FALSE,FALSE,nullptr); if(!event) return HRESULT_FROM_WIN32(GetLastError());
        source=buffer; queue=q;
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={buffer,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_COPY_SOURCE};
        commands->ResourceBarrier(1,&barrier);
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource=buffer; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=layout;
        commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE; barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_PRESENT;
        commands->ResourceBarrier(1,&barrier);
        hr=commands->Close(); if(FAILED(hr)) return hr;
        ID3D12CommandList* lists[]={commands.Get()};
        pending=true; queue->ExecuteCommandLists(1,lists);
        hr=queue->Signal(fence.Get(),1);
        if(FAILED(hr)) { poisoned=true; return hr; }
        return S_OK;
    }
    HRESULT finish(std::vector<unsigned char>& pixels,DWORD timeout=2000) {
        if(!pending || poisoned) return E_UNEXPECTED;
        HRESULT hr;
        if(fence->GetCompletedValue()==UINT64_MAX) { poisoned=true; return DXGI_ERROR_DEVICE_REMOVED; }
        if(fence->GetCompletedValue()<1) {
            hr=fence->SetEventOnCompletion(1,event); if(FAILED(hr)) return hr;
            DWORD waited=WaitForSingleObject(event,timeout);
            if(waited!=WAIT_OBJECT_0) return waited==WAIT_TIMEOUT ? HRESULT_FROM_WIN32(WAIT_TIMEOUT) : HRESULT_FROM_WIN32(GetLastError());
        }
        if(fence->GetCompletedValue()==UINT64_MAX) { poisoned=true; return DXGI_ERROR_DEVICE_REMOVED; }
        pending=false;
        void* mapped=nullptr; D3D12_RANGE range{0,SIZE_T(total)};
        hr=readback->Map(0,&range,&mapped); if(FAILED(hr)) return hr;
        pixels.resize(SIZE_T(row_bytes)*rows);
        for(UINT y=0;y<rows;++y) memcpy(pixels.data()+SIZE_T(y)*row_bytes,
            static_cast<unsigned char*>(mapped)+layout.Offset+SIZE_T(y)*layout.Footprint.RowPitch,SIZE_T(row_bytes));
        D3D12_RANGE no_writes{0,0}; readback->Unmap(0,&no_writes);
        source.Reset(); readback.Reset(); commands.Reset(); allocator.Reset(); queue.Reset(); fence.Reset();
        return S_OK;
    }
};
