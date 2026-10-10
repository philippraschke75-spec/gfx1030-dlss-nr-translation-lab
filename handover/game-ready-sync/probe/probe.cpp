// Probe: D3D12 shared buffers imported into HIP; mid-list sync via WriteBufferImmediate -> hipStreamWaitValue32 and
// hipStreamWriteValue32 -> bounded spin shader. Every wait is bounded.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <utility>
#define CK(x) do { HRESULT _h = (x); if (FAILED(_h)) { printf("FAIL %s -> 0x%08lx line %d\n", #x, (unsigned long)_h, __LINE__); return 10; } } while (0)
#define HK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { printf("HIPFAIL %s -> %s line %d\n", #x, hipGetErrorString(_e), __LINE__); return 11; } } while (0)
static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* al; static ID3D12GraphicsCommandList2* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE ev;
static bool wait(DWORD ms) { q->Signal(fence, ++fv); if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, ev); return WaitForSingleObject(ev, ms) == WAIT_OBJECT_0; } return true; }
static ID3D12Resource* buf(D3D12_HEAP_TYPE t, UINT64 sz, D3D12_HEAP_FLAGS hf, D3D12_RESOURCE_FLAGS rf, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = t; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = rf; ID3D12Resource* r = nullptr; HRESULT h = dev->CreateCommittedResource(&hp, hf, &d, st, nullptr, IID_PPV_ARGS(&r)); if (FAILED(h)) printf("create buf 0x%08lx\n", (unsigned long)h); return r; }
static const char* kSpin =
"globallycoherent RWByteAddressBuffer fl : register(u0);\n"
"RWByteAddressBuffer st : register(u1);\n"
"cbuffer C : register(b0) { uint want; uint maxIt; };\n"
"[numthreads(1,1,1)] void main() { uint v = 0, i = 0; for (; i < maxIt; i++) { fl.InterlockedOr(0, 0, v); if ((int)(v - want) >= 0) break; } st.Store2(0, uint2(i, v)); }\n";
static bool importBuf(ID3D12Resource* r, void** p, hipExternalMemory_t* em) {
    HANDLE h = nullptr; if (FAILED(dev->CreateSharedHandle(r, nullptr, GENERIC_ALL, nullptr, &h))) { puts("CreateSharedHandle failed"); return false; }
    D3D12_RESOURCE_DESC d = r->GetDesc(); D3D12_RESOURCE_ALLOCATION_INFO ai = dev->GetResourceAllocationInfo(0, 1, &d);
    hipExternalMemoryHandleDesc md = {}; md.type = hipExternalMemoryHandleTypeD3D12Resource; md.handle.win32.handle = h; md.size = ai.SizeInBytes; md.flags = hipExternalMemoryDedicated;
    hipError_t e = hipImportExternalMemory(em, &md); if (e != hipSuccess) { printf("import: %s\n", hipGetErrorString(e)); return false; }
    hipExternalMemoryBufferDesc bd = {}; bd.size = d.Width; e = hipExternalMemoryGetMappedBuffer(p, *em, &bd); if (e != hipSuccess) { printf("map: %s\n", hipGetErrorString(e)); return false; }
    CloseHandle(h); return true;
}
int main() { setvbuf(stdout, nullptr, _IONBF, 0);
    IDXGIFactory4* fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac))); IDXGIAdapter1* ad = nullptr;
    for (UINT i = 0;; i++) { IDXGIAdapter1* a; if (fac->EnumAdapters1(i, &a) != S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d); if (d.VendorId == 0x1002) { ad = a; wprintf(L"adapter %s\n", d.Description); break; } }
    CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {}; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al)));
    CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, IID_PPV_ARGS(&cl))); cl->Close();
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))); ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    int dc = -1; HK(hipSetDevice(0)); hipDeviceProp_t pr; HK(hipGetDeviceProperties(&pr, 0)); HK(hipDeviceGetAttribute(&dc, hipDeviceAttributeCanUseStreamWaitValue, 0)); printf("hip %s, CanUseStreamWaitValue=%d\n", pr.gcnArchName, dc);
    const UINT64 N = 1 << 20;
    auto A = buf(D3D12_HEAP_TYPE_DEFAULT, 2 * N + 65536, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto C = buf(D3D12_HEAP_TYPE_DEFAULT, 65536, D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto S = buf(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto up = buf(D3D12_HEAP_TYPE_UPLOAD, N, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto rb = buf(D3D12_HEAP_TYPE_READBACK, N + 256, D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!A || !C || !S || !up || !rb) return 2;
    void *pA, *pC; hipExternalMemory_t eA, eC; if (!importBuf(A, &pA, &eA) || !importBuf(C, &pC, &eC)) return 3; puts("import ok");
    // --- data path
    uint32_t* u; up->Map(0, nullptr, (void**)&u); for (UINT i = 0; i < N / 4; i++) u[i] = i * 2654435761u; up->Unmap(0, nullptr);
    al->Reset(); cl->Reset(al, nullptr); cl->CopyBufferRegion(A, 0, up, 0, N); cl->Close(); { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000)) { puts("TIMEOUT d3d"); return 4; }
    HK(hipMemcpy((char*)pA + N, pA, N, hipMemcpyDeviceToDevice)); HK(hipDeviceSynchronize());
    al->Reset(); cl->Reset(al, nullptr); cl->CopyBufferRegion(rb, 0, A, N, N); cl->Close(); { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000)) { puts("TIMEOUT d3d"); return 4; }
    { uint32_t* r; rb->Map(0, nullptr, (void**)&r); size_t bad = 0; for (UINT i = 0; i < N / 4; i++) bad += r[i] != i * 2654435761u; rb->Unmap(0, nullptr); printf("data path D3D12->HIP->D3D12: %zu bad words\n", bad); }
    // --- spin shader
    auto comp = (decltype(&D3DCompile))GetProcAddress(LoadLibraryA("d3dcompiler_47.dll"), "D3DCompile"); ID3DBlob *cs = nullptr, *er = nullptr;
    if (FAILED(comp(kSpin, strlen(kSpin), "spin", nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, &er))) { printf("hlsl: %s\n", (char*)er->GetBufferPointer()); return 5; }
    D3D12_ROOT_PARAMETER rp[3] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[0].Descriptor.ShaderRegister = 0; rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[2].Constants.Num32BitValues = 2;
    D3D12_ROOT_SIGNATURE_DESC rsd = {3, rp}; ID3DBlob* sb; ID3D12RootSignature* rs; ID3D12PipelineState* pso;
    CK(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, nullptr)); CK(dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&rs)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = rs; pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()}; CK(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    hipStream_t s; HK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    HK(hipMemset(pA, 0, 4)); HK(hipMemset(pC, 0, 4)); HK(hipDeviceSynchronize());
    // flag A[2N] = HIP->D3D12 (spin target), C[0] = D3D12->HIP (WriteBufferImmediate). Busy HIP work = memset of A[0..N) 50x.
    void* flagOut = (char*)pA + 2 * N;
    int mode = 0; hipEvent_t fl; HK(hipEventCreate(&fl));
    auto frame = [&](UINT k, bool hipSignals, UINT maxIt, double* ms, UINT* it, UINT* got) -> int {
        if (hipSignals) {
            if ((mode == 0 || mode == 4) && hipStreamWaitValue32(s, pC, k, hipStreamWaitValueGte, 0xFFFFFFFF) != hipSuccess) return -1;
            for (int j = 0; j < 50; j++) hipMemsetAsync(pA, j, N, s);
            if (hipStreamWriteValue32(s, flagOut, k, 0) != hipSuccess) return -2;
            if (mode >= 3) { hipStreamQuery(s); hipEventRecord(fl, s); hipEventQuery(fl); }
        }
        al->Reset(); cl->Reset(al, nullptr);
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER wp = {C->GetGPUVirtualAddress(), k}; D3D12_WRITEBUFFERIMMEDIATE_MODE wm = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
        D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = C; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        cl->ResourceBarrier(1, &b); cl->WriteBufferImmediate(1, &wp, &wm); std::swap(b.Transition.StateBefore, b.Transition.StateAfter); cl->ResourceBarrier(1, &b);
        cl->SetComputeRootSignature(rs); cl->SetPipelineState(pso); cl->SetComputeRootUnorderedAccessView(0, A->GetGPUVirtualAddress() + 2 * N);
        cl->SetComputeRootUnorderedAccessView(1, S->GetGPUVirtualAddress()); UINT cb[2] = {k, maxIt}; cl->SetComputeRoot32BitConstants(2, 2, cb, 0); cl->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER u = {}; u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; cl->ResourceBarrier(1, &u);
        cl->CopyBufferRegion(rb, N, S, 0, 8); cl->Close();
        LARGE_INTEGER f0, f1, fq; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&f0);
        ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(10000)) return -3;
        QueryPerformanceCounter(&f1); *ms = (f1.QuadPart - f0.QuadPart) * 1000.0 / fq.QuadPart;
        uint32_t* r; rb->Map(0, nullptr, (void**)&r); *it = r[N / 4]; *got = r[N / 4 + 1]; rb->Unmap(0, nullptr); return 0;
    };
    double ms; UINT it, got; int r; hipEvent_t e; HK(hipEventCreate(&e));
    auto hipIdle = [&](DWORD tmo) { HK(hipEventRecord(e, s)); DWORD t0 = GetTickCount(); while (hipEventQuery(e) == hipErrorNotReady && GetTickCount() - t0 < tmo) Sleep(1); return hipEventQuery(e) == hipSuccess ? 0 : 1; };
    // case A: WBI only (spin bound 1 iteration) -> does HIP see C=k and finish?
    r = frame(1, true, 1, &ms, &it, &got); printf("A: d3d %.2f ms; ", ms); { DWORD t0 = GetTickCount(); int ni = hipIdle(3000); printf("hip after wbi: %s in %lu ms\n", ni ? "STUCK" : "done", GetTickCount() - t0); if (ni) { HK(hipMemset(pC, 0x7f, 4)); return 8; } }
    // case B: HIP signalled, spin bound ~150 ms
    for (UINT k = 2; k <= 6; k++) { r = frame(k, true, 900000, &ms, &it, &got); if (r) { printf("frame r=%d\n", r); return 7; } printf("B frame %u: %.2f ms spin it=%u flag=%u\n", k, ms, it, got); hipIdle(3000); }
    mode = 1; // case C: HIP signals without waiting for D3D12 -> pure concurrency test
    for (UINT k = 10; k <= 13; k++) { r = frame(k, true, 900000, &ms, &it, &got); printf("C frame %u: %.2f ms spin it=%u flag=%u\n", k, ms, it, got); hipIdle(3000); }
    mode = 2; // case E: HIP finished before D3D12 runs -> visibility test
    for (UINT k = 20; k <= 22; k++) { hipStreamWriteValue32(s, flagOut, k, 0); hipIdle(3000); r = frame(k, false, 900000, &ms, &it, &got); printf("E frame %u: %.2f ms spin it=%u flag=%u\n", k, ms, it, got); }
    mode = 3; // case F: like C/B but HIP submission flushed
    for (UINT k = 30; k <= 32; k++) { r = frame(k, true, 900000, &ms, &it, &got); printf("F(no wait) frame %u: %.2f ms spin it=%u flag=%u\n", k, ms, it, got); hipIdle(3000); }
    mode = 4; for (UINT k = 40; k <= 42; k++) { r = frame(k, true, 900000, &ms, &it, &got); printf("F(wait) frame %u: %.2f ms spin it=%u flag=%u\n", k, ms, it, got); hipIdle(3000); }
    puts("PROBE DONE"); return 0;
}
