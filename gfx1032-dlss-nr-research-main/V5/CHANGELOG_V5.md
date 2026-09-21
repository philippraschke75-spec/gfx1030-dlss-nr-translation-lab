# DLSS-NR-on-RDNA2 V5 Diagnostic

## Summary

This V5 adds a diagnostic-only HIP path that is isolated from the known-good V4 runtime flow. The project intentionally avoids removing or replacing the V4 logic and keeps the module-loader experiment behind explicit logging and guarded execution.

## What changed

- Added a dedicated V5 diagnostic DLL entry point and initialization path.
- Added a concise, tag-based log file named `dlssnr_v5.log`.
- Added bundle discovery logic that scans the DLL image for the known `__CLANG_OFFLOAD_BUNDLE__` marker and the `hipv4-amdgcn-amd-amdhsa--gfx1032` target.
- Added ELF validation checks for the embedded AMDGPU payload: magic, class, little-endian, machine type, and recognized object type.
- Added runtime HIP symbol resolution for `hipModuleLoadData` and `hipModuleGetFunction` using the loaded HIP runtime module.
- Added an isolated module-load attempt that logs success/failure but does not execute a kernel.
- Added cleanup/unload handling so the diagnostic path stays safe and does not interfere with rendering output.

## Known constraints

- This code does not assume the embedded ELF is valid for execution merely because it is structurally well-formed.
- The module-load attempt is diagnostic-only and intentionally stopped before any kernel launch.
- The design logs failures and exits cleanly rather than altering the V4 runtime path.
