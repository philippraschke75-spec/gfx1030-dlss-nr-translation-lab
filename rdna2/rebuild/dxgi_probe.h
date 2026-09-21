#pragma once
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"user32.lib")
using Microsoft::WRL::ComPtr;
// Hidden WARP swapchain used only to discover and test DXGI entry points.
struct DxgiProbe {
    HWND window=nullptr;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain1> swap;
    HRESULT create(bool amd_hardware=false) {
        window=CreateWindowExW(0,L"STATIC",L"DLSSNR hook probe",WS_OVERLAPPEDWINDOW,
                              0,0,64,64,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        if(!window) return HRESULT_FROM_WIN32(GetLastError());
        HRESULT hr=CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if(FAILED(hr)) return hr;
        if(amd_hardware) {
            hr=DXGI_ERROR_NOT_FOUND;
            for(UINT i=0;;++i) {
                ComPtr<IDXGIAdapter1> candidate;
                if(factory->EnumAdapters1(i,&candidate)==DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 info{};
                if(candidate && SUCCEEDED(candidate->GetDesc1(&info)) && info.VendorId==0x1002 && !(info.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    hr=candidate.As(&adapter); break;
                }
            }
        } else hr=factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        if(FAILED(hr)) return hr;
        hr=D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device));
        if(FAILED(hr)) return hr;
        D3D12_COMMAND_QUEUE_DESC q{};
        hr=device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue));
        if(FAILED(hr)) return hr;
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width=64; d.Height=64; d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count=1; d.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount=2; d.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
        return factory->CreateSwapChainForHwnd(queue.Get(),window,&d,nullptr,nullptr,&swap);
    }
    ~DxgiProbe() {
        swap.Reset(); queue.Reset(); device.Reset(); adapter.Reset(); factory.Reset();
        if(window) DestroyWindow(window);
    }
};
