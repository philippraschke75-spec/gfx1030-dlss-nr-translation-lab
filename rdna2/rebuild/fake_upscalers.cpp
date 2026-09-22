// Test-only stand-ins with the same export names as the real upscaler DLLs. Each returns a magic value so the test can prove the
// detour forwards to the original and preserves its return value.
#include <windows.h>
#ifdef FAKE_FSR
extern "C" __declspec(dllexport) int ffxFsr3UpscalerContextDispatch(void*,const void*) { return 1234; }
#else
extern "C" __declspec(dllexport) unsigned ffxDispatch(void*,const void*) { return 77; }
#endif
