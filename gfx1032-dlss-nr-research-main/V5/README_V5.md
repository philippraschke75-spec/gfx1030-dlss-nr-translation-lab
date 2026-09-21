# DLSS-NR V5 Diagnostic Companion

This project creates a separate diagnostic DLL named `dlssnr_v5.dll` for research-only inspection of the already-loaded `version_v4.dll`.

Important:
- V5 does not replace or modify `version_v4.dll`.
- V5 does not patch, inject into, overwrite, or rebuild `version.dll`.
- V5 does not alter GTA rendering resources or V4 control flow.
- V5 is intentionally limited to observation, PE/bundle analysis, HIP runtime discovery, and a single guarded `hipModuleLoadData` experiment.

## Safety model

V5 is a companion diagnostic layer. It is designed to answer one narrow question:

> Can the embedded gfx1032 ELF in the already-loaded V4 image be accepted by the installed AMD HIP runtime without a kernel launch or state mutation?

The DLL intentionally stops after a successful `hipModuleLoadData` result and does not call any kernel-launch API.

## Runtime workflow

1. Find the already-loaded `version_v4.dll` by name.
2. Validate the DOS and NT headers and enumerate the loaded sections.
3. Search only the `.hip_fat` / `.hipFatB` section ranges for the CLANG offload bundle marker.
4. Enumerate the available bundle targets and locate the `gfx1032` payload.
5. Validate the ELF64 AMDGPU object before passing it to HIP.
6. Resolve `amdhip64_7.dll` dynamically if present, or load it normally.
7. Query device properties via `hipGetDeviceCount` and `hipGetDevicePropertiesR0600`.
8. Attempt a single `hipModuleLoadData` call using the validated ELF pointer and size.
9. Stop before any kernel launch or dispatch.

## Build status

This workspace does not currently have a functional C/C++ toolchain for Windows DLL builds:
- `cmake` was not found
- `cl` (MSVC compiler) was not found

Because of that, this source was prepared but not compiled here. On a developer machine with Visual Studio 2022 or LLVM/clang-cl installed, the project can be built with the provided `CMakeLists.txt`.

## Files in this directory

- `dlssnr_v5.cpp`: V5 diagnostic implementation
- `CMakeLists.txt`: CMake project for `dlssnr_v5`
- `README_V5.md`: safety and usage notes

## No mutation policy

V5 must never:
- modify `version_v4.dll`
- patch `version.dll`
- overwrite GTA resources
- attach runtime hooks to rendering
- alter V4's state byte
- launch a GPU kernel
- assume an ELF is valid without validation

This is strictly an observation and module-load experiment, not a replacement for V4.
