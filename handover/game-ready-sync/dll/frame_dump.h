// dump_frame=N (nr_port.ini): the N-th evaluated frame of the main feature (pass 0) - the host's colour, depth, motion
// and our output - copied to readback buffers in the host's command list, written ~10 frames later next to the log as
// raw float32 (<name>_<W>x<H>_<C>ch.f32, row-major, C channels) plus a PNG preview. Included by dlssnr_nr.cpp only.
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "r11g10.h"

struct DumpItem { ID3D12Resource* rb = nullptr; uint8_t* p = nullptr; D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {}; DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN; unsigned w = 0, h = 0; std::string name; };

// Copy subresource 0 of r (in `state`, restored after) into a new readback buffer.
static bool dumpQueue(ID3D12Device* dev, ID3D12GraphicsCommandList* cmd, ID3D12Resource* r, D3D12_RESOURCE_STATES state, unsigned w, unsigned h, const char* name, std::vector<DumpItem>& out) {
    if (!r) return false;
    D3D12_RESOURCE_DESC d = r->GetDesc(); DumpItem it; UINT rows = 0; UINT64 rowBytes = 0, total = 0;
    dev->GetCopyableFootprints(&d, 0, 1, 0, &it.fp, &rows, &rowBytes, &total);
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd = {}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&it.rb)))) return false;
    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = r; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = state; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; cmd->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dl = {}, sl = {}; dl.pResource = it.rb; dl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dl.PlacedFootprint = it.fp;
    sl.pResource = r; sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sl.SubresourceIndex = 0;
    cmd->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter); cmd->ResourceBarrier(1, &b);
    it.fmt = d.Format; it.w = (std::min)(w ? w : (unsigned)d.Width, (unsigned)d.Width); it.h = (std::min)(h ? h : d.Height, d.Height); it.name = name;
    out.push_back(it); return true;
}

static float dumpHalf(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 0x3FF; uint32_t f;
    if (e == 0) { if (!m) f = s; else { float v = m / 1024.f / 16384.f; return s ? -v : v; } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13); else f = s | ((e + 112) << 23) | (m << 13);
    float v; memcpy(&v, &f, 4); return v;
}

// Texel -> floats; returns the channel count, 0 for an unknown format.
static int dumpTexel(DXGI_FORMAT f, const uint8_t* p, float* o) {
    uint16_t h[4]; uint32_t u; memcpy(&u, p, 4);
    switch (f) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS: memcpy(h, p, 8); for (int i = 0; i < 4; i++) o[i] = dumpHalf(h[i]); return 4;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_TYPELESS: memcpy(o, p, 16); return 4;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_TYPELESS: for (int i = 0; i < 4; i++) o[i] = p[i] / 255.f; return 4;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_TYPELESS: o[0] = p[2] / 255.f; o[1] = p[1] / 255.f; o[2] = p[0] / 255.f; o[3] = p[3] / 255.f; return 4;
    case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS: for (int i = 0; i < 3; i++) o[i] = ((u >> (10 * i)) & 1023) / 1023.f; o[3] = (u >> 30) / 3.f; return 4;
    case DXGI_FORMAT_R11G11B10_FLOAT: unpack_r11g11b10(u, h); for (int i = 0; i < 3; i++) o[i] = dumpHalf(h[i]); return 3;
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_TYPELESS: memcpy(h, p, 4); o[0] = dumpHalf(h[0]); o[1] = dumpHalf(h[1]); return 2;
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_TYPELESS: memcpy(o, p, 8); return 2;
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: memcpy(o, p, 4); return 1;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: o[0] = (u & 0xFFFFFF) / 16777215.f; return 1;
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: o[0] = (u & 0xFFFF) / 65535.f; return 1;
    case DXGI_FORMAT_R16_FLOAT: o[0] = dumpHalf((uint16_t)u); return 1;
    default: return 0;
    }
}

static unsigned dumpBpp(DXGI_FORMAT f) {
    switch (f) { case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_TYPELESS: return 16;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_TYPELESS: return 8;
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_FLOAT: return 2;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return 8; default: return 4; }
}

// Minimal PNG (8-bit RGB, stored deflate blocks: no zlib needed).
static void dumpPng(const char* path, const std::vector<uint8_t>& rgb, unsigned w, unsigned h) {
    static uint32_t T[256]; if (!T[1]) for (uint32_t n = 0; n < 256; n++) { uint32_t c = n; for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; T[n] = c; }
    std::vector<uint8_t> raw; raw.reserve((size_t)(w * 3 + 1) * h);
    for (unsigned y = 0; y < h; y++) { raw.push_back(0); raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3, rgb.begin() + (size_t)(y + 1) * w * 3); }
    std::vector<uint8_t> z = {0x78, 0x01}; uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw.size(); i += 65535) { const size_t n = (std::min)((size_t)65535, raw.size() - i);
        z.push_back(i + n == raw.size()); z.push_back(n & 255); z.push_back(n >> 8); z.push_back(~n & 255); z.push_back((~n >> 8) & 255); z.insert(z.end(), raw.begin() + i, raw.begin() + i + n); }
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (int s = 24; s >= 0; s -= 8) z.push_back((uint8_t)(((b << 16) | a) >> s));
    FILE* f = nullptr; if (fopen_s(&f, path, "wb") || !f) return;
    auto be = [&](uint32_t v) { uint8_t q[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v}; fwrite(q, 1, 4, f); };
    auto chunk = [&](const char* t, const uint8_t* d, size_t n) { be((uint32_t)n); uint32_t c = 0xFFFFFFFFu; fwrite(t, 1, 4, f); for (int i = 0; i < 4; i++) c = T[(c ^ (uint8_t)t[i]) & 255] ^ (c >> 8);
        if (n) fwrite(d, 1, n, f); for (size_t i = 0; i < n; i++) c = T[(c ^ d[i]) & 255] ^ (c >> 8); be(c ^ 0xFFFFFFFFu); };
    const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10}; fwrite(sig, 1, 8, f);
    uint8_t ih[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w, (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
    chunk("IHDR", ih, 13); chunk("IDAT", z.data(), z.size()); chunk("IEND", nullptr, 0); fclose(f);
}

// Write every queued item (call only once the GPU has run the list that recorded the copies). Returns a log summary.
static std::string dumpWrite(const std::string& dir, std::vector<DumpItem>& items) {
    std::string log;
    for (auto& it : items) {
        void* mp = nullptr; if (FAILED(it.rb->Map(0, nullptr, &mp)) || !mp) { log += " " + it.name + " (map failed)"; continue; } it.p = (uint8_t*)mp;
        float t[4]; const unsigned bpp = dumpBpp(it.fmt); const int C = dumpTexel(it.fmt, it.p + it.fp.Offset, t);
        char base[MAX_PATH]; snprintf(base, sizeof base, "%sdump_%s_%ux%u", dir.c_str(), it.name.c_str(), it.w, it.h);
        if (!C) { std::string p = std::string(base) + "_fmt" + std::to_string((int)it.fmt) + ".bin"; FILE* f = nullptr;   // unknown format: the rows as they are
            if (!fopen_s(&f, p.c_str(), "wb") && f) { for (unsigned y = 0; y < it.h; y++) fwrite(it.p + it.fp.Offset + (size_t)y * it.fp.Footprint.RowPitch, 1, (size_t)it.w * bpp, f); fclose(f); }
            log += " " + it.name + " (raw fmt " + std::to_string((int)it.fmt) + ")"; continue; }
        std::vector<float> v((size_t)it.w * it.h * C); float lo = 1e30f, hi = -1e30f;
        for (unsigned y = 0; y < it.h; y++) for (unsigned x = 0; x < it.w; x++) { dumpTexel(it.fmt, it.p + it.fp.Offset + (size_t)y * it.fp.Footprint.RowPitch + (size_t)x * bpp, t);
            for (int c = 0; c < C; c++) { float q = t[c]; v[((size_t)y * it.w + x) * C + c] = q; if (q == q) { lo = (std::min)(lo, q); hi = (std::max)(hi, q); } } }
        char p[MAX_PATH]; snprintf(p, sizeof p, "%s_%dch.f32", base, C); FILE* f = nullptr;
        if (!fopen_s(&f, p, "wb") && f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        // preview: colour as is (clamped), 1 channel normalised min..max, 2 channels (motion) 0.5 + v / (2 max|v|)
        std::vector<uint8_t> rgb((size_t)it.w * it.h * 3); const float m = (std::max)(fabsf(lo), fabsf(hi)) + 1e-12f;
        for (size_t i = 0; i < (size_t)it.w * it.h; i++) for (int c = 0; c < 3; c++) {
            float q = C >= 3 ? v[i * C + c] : C == 1 ? (v[i] - lo) / (hi - lo + 1e-12f) : c < 2 ? 0.5f + v[i * 2 + c] / (2 * m) : 0.5f;
            rgb[i * 3 + c] = (uint8_t)((q == q ? (std::min)((std::max)(q, 0.f), 1.f) : 0.f) * 255.f + 0.5f); }
        snprintf(p, sizeof p, "%s.png", base); dumpPng(p, rgb, it.w, it.h);
        char s[160]; snprintf(s, sizeof s, " %s (fmt %d, %dch, range %.4g..%.4g)", it.name.c_str(), (int)it.fmt, C, lo, hi); log += s;
    }
    for (auto& it : items) if (it.rb) { if (it.p) it.rb->Unmap(0, nullptr); it.rb->Release(); }
    items.clear(); return log;
}
