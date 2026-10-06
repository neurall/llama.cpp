// dstorage_bench FILE [--sec 0.5]: DirectStorage file reads into CPU memory and straight into a D3D12 GPU buffer (the NVIDIA adapter), random aligned requests.
// Per setting: GB/s, CPU ms per GB. Compare with iobench_win (IOCP / threads, the same file): does DirectStorage beat the drive-limited rate or cost less CPU,
// and does reading into VRAM directly (no pinned staging copy, no RAM traffic for the data) cost speed?
// build (Build Tools): cl /nologo /O2 /EHsc /I N:\ds\pkg\native\include dstorage_bench.cpp /link N:\ds\pkg\native\lib\x64\dstorage.lib d3d12.lib dxgi.lib; dstorage.dll and dstoragecore.dll beside the exe
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;

static double now() { LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double) t.QuadPart / f.QuadPart; }
static double cpu_s() { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u); return (((uint64_t) k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t) u.dwHighDateTime << 32 | u.dwLowDateTime)) * 1e-7; }
static uint64_t rnd(uint64_t & s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
#define CK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { fprintf(stderr, "%s failed: 0x%08lx (line %d)\n", #x, (unsigned long) hr_, __LINE__); exit(1); } } while (0)

static ComPtr<ID3D12Device> g_dev; static ComPtr<IDStorageFactory> g_fac; static ComPtr<IDStorageFile> g_file; static uint64_t g_size; static double g_sec = 0.5;

// one measurement: `qd` requests per wave, two waves in flight (ping-pong), gpu = destination is a D3D12 default-heap buffer, else CPU memory
static void run(size_t bs, int qd, bool gpu) {
    DSTORAGE_QUEUE_DESC qdsc = {}; qdsc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; qdsc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY; qdsc.Priority = DSTORAGE_PRIORITY_NORMAL;
    qdsc.Device = gpu ? g_dev.Get() : nullptr;
    ComPtr<IDStorageQueue> q0; CK(g_fac->CreateQueue(&qdsc, IID_PPV_ARGS(&q0))); ComPtr<IDStorageQueue1> q; CK(q0.As(&q));   // EnqueueSetEvent is on IDStorageQueue1
    size_t slots = (size_t) qd * 2; char * mem = nullptr; ComPtr<ID3D12Resource> res;
    if (gpu) {
        D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT }; D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = bs * slots; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)));
    } else mem = (char *) VirtualAlloc(nullptr, bs * slots, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    HANDLE ev[2] = { CreateEvent(nullptr, FALSE, FALSE, nullptr), CreateEvent(nullptr, FALSE, FALSE, nullptr) };
    uint64_t seed = 88172645463325252ull; size_t nblk = g_size / bs, bytes = 0;
    auto wave = [&](int w) {
        for (int i = 0; i < qd; i++) {
            DSTORAGE_REQUEST r = {}; r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; r.Options.DestinationType = gpu ? DSTORAGE_REQUEST_DESTINATION_BUFFER : DSTORAGE_REQUEST_DESTINATION_MEMORY;
            r.Source.File.Source = g_file.Get(); r.Source.File.Offset = (rnd(seed) % nblk) * bs; r.Source.File.Size = (UINT32) bs; r.UncompressedSize = (UINT32) bs;
            size_t slot = (size_t) w * qd + i;
            if (gpu) { r.Destination.Buffer.Resource = res.Get(); r.Destination.Buffer.Offset = slot * bs; r.Destination.Buffer.Size = (UINT32) bs; }
            else { r.Destination.Memory.Buffer = mem + slot * bs; r.Destination.Memory.Size = (UINT32) bs; }
            q->EnqueueRequest(&r);
        }
        q->EnqueueSetEvent(ev[w]); q->Submit();
    };
    double t0 = now(), c0 = cpu_s(); wave(0); wave(1); int w = 0;
    while (now() - t0 < g_sec) {
        WaitForSingleObject(ev[w], 30000); bytes += bs * qd; wave(w); w ^= 1;
    }
    WaitForSingleObject(ev[w], 30000); WaitForSingleObject(ev[w ^ 1], 30000);
    double el = now() - t0, c1 = cpu_s();
    DSTORAGE_ERROR_RECORD er = {}; q->RetrieveErrorRecord(&er);
    if (FAILED(er.FirstFailure.HResult)) { fprintf(stderr, "request failed: 0x%08lx\n", (unsigned long) er.FirstFailure.HResult); }
    double gbs = bytes / el / 1e9;
    printf("%-5s %6zuk %4d %7.2f %9.1f\n", gpu ? "gpu" : "mem", bs >> 10, qd, gbs, (c1 - c0) * 1e3 / (gbs * el)); fflush(stdout);
    CloseHandle(ev[0]); CloseHandle(ev[1]); if (mem) VirtualFree(mem, 0, MEM_RELEASE);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dstorage_bench FILE [--sec S]\n"); return 1; }
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--sec") && i + 1 < argc) g_sec = atof(argv[++i]);
    ComPtr<IDXGIFactory4> df; CK(CreateDXGIFactory1(IID_PPV_ARGS(&df))); ComPtr<IDXGIAdapter1> ad, pick;
    for (UINT i = 0; df->EnumAdapters1(i, &ad) == S_OK; i++) { DXGI_ADAPTER_DESC1 d; ad->GetDesc1(&d); if (d.VendorId == 0x10DE) { pick = ad; wprintf(L"adapter: %s\n", d.Description); break; } }
    CK(D3D12CreateDevice(pick.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_dev)));
    CK(DStorageGetFactory(IID_PPV_ARGS(&g_fac))); g_fac->SetStagingBufferSize(256u << 20);
    std::wstring w(argv[1], argv[1] + strlen(argv[1])); CK(g_fac->OpenFile(w.c_str(), IID_PPV_ARGS(&g_file)));
    BY_HANDLE_FILE_INFORMATION fi; CK(g_file->GetFileInformation(&fi)); g_size = (uint64_t) fi.nFileSizeHigh << 32 | fi.nFileSizeLow;
    printf("file %.1f GiB, %.2f s per point, two waves of `qd` requests in flight\n%-5s %7s %4s %7s %9s\n", g_size / 1073741824.0, g_sec, "dest", "req", "qd", "GB/s", "cpu ms/GB");
    for (int gpu = 0; gpu < 2; gpu++) for (size_t bs : { 256u << 10, 1u << 20, 4u << 20, 16u << 20 }) for (int qd : { 1, 2, 4, 8 }) run(bs, qd, gpu);
    return 0;
}
