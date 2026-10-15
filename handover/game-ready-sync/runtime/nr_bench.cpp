// nr_bench <dir_with_nr_runtime.dll> <data_dir> <W> <H> [frames=40] [vit_every=1] [hist=1]
// GPU time of the whole network per frame (nr_run2 gpu_ms, e0..e1), zero input; prints median / mean / min over frames 3..N.
// vit_every < -1 = bottleneck reuse, as nr_port.ini reuse=bottleneck. Set NR_PROF=1 for the per-dispatch list of frame 5.
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
typedef void* (*P_create)(const char*, int, int, int, char*, int);
typedef int (*P_run2)(void*, const void*, void*, float*, int);
typedef size_t (*P_bytes)(void*);
typedef void (*P_destroy)(void*);
int main(int argc, char** argv) {
  if (argc < 5) { puts("usage: nr_bench dll_dir data_dir W H [frames] [vit_every] [hist]"); return 1; }
  HMODULE m = LoadLibraryA((std::string(argv[1]) + "\\nr_runtime.dll").c_str()); if (!m) { puts("no nr_runtime.dll"); return 2; }
  auto cr = (P_create)GetProcAddress(m, "nr_create"); auto run = (P_run2)GetProcAddress(m, "nr_run2");
  auto by = (P_bytes)GetProcAddress(m, "nr_bytes"); auto de = (P_destroy)GetProcAddress(m, "nr_destroy");
  int n = argc > 5 ? atoi(argv[5]) : 40, ve = argc > 6 ? atoi(argv[6]) : 1, hv = argc > 7 ? atoi(argv[7]) : 1;
  char err[512] = {}; void* c = cr(argv[2], atoi(argv[3]), atoi(argv[4]), ve, err, sizeof err); if (!c) { printf("nr_create: %s\n", err); return 3; }
  std::vector<unsigned char> in(by(c), 0), out(64 << 20); std::vector<float> t;
  if (const char* p = getenv("NR_BENCH_IN")) { FILE* f = fopen(p, "rb"); if (!f || fread(in.data(), 1, in.size(), f) != in.size()) { puts("bad NR_BENCH_IN"); return 5; } fclose(f); }
  for (int i = 0; i < n; i++) { float ms = 0; if (run(c, in.data(), out.data(), &ms, i ? hv : 0)) { puts("nr_run2 failed"); return 4; } if (i >= 3) t.push_back(ms); }
  double s = 0; for (float x : t) s += x; std::sort(t.begin(), t.end());
  printf("BENCH %sx%s vit_every=%d frames=%zu: median %.3f mean %.3f min %.3f ms\n", argv[3], argv[4], ve, t.size(), t[t.size() / 2], s / t.size(), t[0]);
  if (const char* p = getenv("NR_BENCH_OUT")) { FILE* f = fopen(p, "wb"); if (f) { fwrite(out.data(), 1, by(c), f); fclose(f); } }   // dst_size == src_size
  de(c); return 0;
}
