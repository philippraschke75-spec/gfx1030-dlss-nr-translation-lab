// Bounded single dispatch of a translated SWIN kernel with a fixture written by emu/run_emu.py.
// usage: swin_gpu_test module.co symbol params.bin in.bin blob.bin out.bin gridx gridy [outpad]
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
 auto params=rd(argv[3]), in=rd(argv[4]), blob=rd(argv[5]), out=rd(argv[6]);
 unsigned gx=atoi(argv[7]), gy=atoi(argv[8]);
 hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p,0)); if(std::strcmp(p.gcnArchName,"gfx1030")) return 3;
 hipModule_t m; hipFunction_t f; CHECK(hipModuleLoad(&m,argv[1])); CHECK(hipModuleGetFunction(&f,m,argv[2]));
 uint8_t *dIn,*dOut,*dBlob; const size_t PAD=1<<16;
 CHECK(hipMalloc(&dIn,in.size())); CHECK(hipMalloc(&dOut,out.size()+2*PAD)); CHECK(hipMalloc(&dBlob,blob.size()));
 CHECK(hipMemcpy(dIn,in.data(),in.size(),hipMemcpyHostToDevice)); CHECK(hipMemcpy(dBlob,blob.data(),blob.size(),hipMemcpyHostToDevice));
 std::vector<uint8_t> outh(out.size()+2*PAD,0xA5); std::memcpy(outh.data()+PAD,out.data(),out.size());
 CHECK(hipMemcpy(dOut,outh.data(),outh.size(),hipMemcpyHostToDevice));
 std::vector<uint8_t> ka(296,0); std::memcpy(ka.data(),params.data(),params.size()<80?params.size():80);
 uint64_t a[3]={(uint64_t)dIn,(uint64_t)(dOut+PAD),(uint64_t)dBlob}; std::memcpy(ka.data(),a,24);
 uint32_t cnt[3]={gx,gy,1}; std::memcpy(ka.data()+80,cnt,12); uint16_t gs[3]={256,1,1}; std::memcpy(ka.data()+92,gs,6);
 size_t sz=ka.size(); void* cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,ka.data(),HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
 hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
 CHECK(hipModuleLaunchKernel(f,gx,gy,1,256,1,1,0,nullptr,nullptr,cfg)); hipEventRecord(e1);
 double tmo=getenv("SWIN_TIMEOUT")?atof(getenv("SWIN_TIMEOUT")):20; auto t0=std::chrono::steady_clock::now();
 while(hipEventQuery(e1)==hipErrorNotReady){ if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>tmo){std::printf("TIMEOUT after %.0f s; exiting without further GPU calls\n",tmo);std::fflush(stdout);std::_Exit(5);} }
 hipError_t e=hipEventQuery(e1); if(e!=hipSuccess){std::printf("kernel error: %s\n",hipGetErrorString(e));return 6;}
 float ms; hipEventElapsedTime(&ms,e0,e1);
 CHECK(hipMemcpy(outh.data(),dOut,outh.size(),hipMemcpyDeviceToHost));
 bool guard=true; for(size_t i=0;i<PAD;i++) guard&=outh[i]==0xA5; for(size_t i=PAD+out.size();i<outh.size();i++) guard&=outh[i]==0xA5;
 FILE*fo=std::fopen(argv[6],"wb"); std::fwrite(outh.data()+PAD,1,out.size(),fo); std::fclose(fo);
 std::printf("GPU dispatch OK: %.3f ms, guards %s\n",ms,guard?"intact":"CORRUPT"); return guard?0:7;
}
