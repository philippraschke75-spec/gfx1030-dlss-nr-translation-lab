// Bounded single dispatch of a translated kernel whose parameters live in one guarded arena.
// usage: var_gpu_test module.co symbol kernarg.bin arena.bin arena_base_hex gridx gridy threads
// kernarg.bin: the full kernarg block (explicit + hidden). Every 8-byte aligned qword in it whose value
// lies inside [arena_base, arena_base+arena_size) is rebased to the device arena. arena.bin is overwritten
// with the post-run contents. A 64 KiB sentinel guard surrounds the arena and is checked after the run.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::printf("%s: %s\n",#x,hipGetErrorString(e)); return 2;} } while(0)
static std::vector<uint8_t> rd(const char* p){ FILE*f=std::fopen(p,"rb"); if(!f){std::printf("cannot open %s\n",p);std::exit(9);} std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET); std::vector<uint8_t> v(n); if(n&&std::fread(v.data(),1,n,f)!=(size_t)n)std::exit(9); std::fclose(f); return v; }
int main(int argc,char**argv){
 if(argc<9) return 1;
 auto ka=rd(argv[3]); auto arena=rd(argv[4]);
 uint64_t base=std::strtoull(argv[5],nullptr,16); unsigned gx=atoi(argv[6]), gy=atoi(argv[7]), thr=atoi(argv[8]);
 hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p,0)); if(std::strcmp(p.gcnArchName,"gfx1030")) return 3;
 hipModule_t m; hipFunction_t f; CHECK(hipModuleLoad(&m,argv[1])); CHECK(hipModuleGetFunction(&f,m,argv[2]));
 const size_t PAD=1<<16; uint8_t* d; CHECK(hipMalloc(&d,arena.size()+2*PAD));
 std::vector<uint8_t> host(arena.size()+2*PAD,0xA5); std::memcpy(host.data()+PAD,arena.data(),arena.size());
 CHECK(hipMemcpy(d,host.data(),host.size(),hipMemcpyHostToDevice));
 uint64_t dev=(uint64_t)(d+PAD); int rebased=0;
 for(size_t o=0;o+8<=ka.size();o+=8){ uint64_t v; std::memcpy(&v,ka.data()+o,8); if(v>=base && v<base+arena.size()){ v=dev+(v-base); std::memcpy(ka.data()+o,&v,8); rebased++; } }
 size_t sz=ka.size(); void* cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,ka.data(),HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
 hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
 CHECK(hipModuleLaunchKernel(f,gx,gy,1,thr,1,1,0,nullptr,nullptr,cfg)); hipEventRecord(e1);
 double tmo=getenv("SWIN_TIMEOUT")?atof(getenv("SWIN_TIMEOUT")):10; auto t0=std::chrono::steady_clock::now();
 while(hipEventQuery(e1)==hipErrorNotReady){ if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>tmo){std::printf("TIMEOUT after %.0f s; exiting without further GPU calls\n",tmo);std::fflush(stdout);std::_Exit(5);} }
 hipError_t e=hipEventQuery(e1); if(e!=hipSuccess){std::printf("kernel error: %s\n",hipGetErrorString(e));return 6;}
 float ms; hipEventElapsedTime(&ms,e0,e1);
 CHECK(hipMemcpy(host.data(),d,host.size(),hipMemcpyDeviceToHost));
 bool guard=true; for(size_t i=0;i<PAD;i++) guard&=host[i]==0xA5; for(size_t i=PAD+arena.size();i<host.size();i++) guard&=host[i]==0xA5;
 FILE*fo=std::fopen(argv[4],"wb"); std::fwrite(host.data()+PAD,1,arena.size(),fo); std::fclose(fo);
 std::printf("GPU dispatch OK: %.3f ms, %d pointers rebased, arena_dev=0x%llx, guards %s\n",ms,rebased,(unsigned long long)dev,guard?"intact":"CORRUPT"); return guard?0:7;
}
