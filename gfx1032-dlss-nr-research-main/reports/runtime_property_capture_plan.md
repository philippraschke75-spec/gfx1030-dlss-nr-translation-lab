# Runtime property capture plan for the RX 6600 / gfx1032 compatibility gate

## Goal

Capture the actual runtime values used by the compatibility gate on the RX 6600 without patching the production binary.

The required evidence points are immediately after:

- `0x18000AA09` -> `hipGetDevicePropertiesR0600`
- `0x18000AD1A` -> `hipGetDevicePropertiesR0600`

The goal is to record:

- device index
- HIP return code
- GPU name
- gcnArchName / architecture string
- numeric properties actually read by the code
- values copied into locals around `[RBP+0x468]` and `[RBP+0x46C]`
- `R14D` at the final compatibility check
- `RDI` / `R15` values used by the architecture comparisons
- the exact boolean conditions produced by the comparisons

This is not a bypass step. It is purely a data-capture step to prove the original compatibility decision.

---

## 1. What the installed ROCm ABI proves

The authoritative local header is:

`C:\Program Files\AMD\ROCm\7.2\include\hip\hip_runtime_api.h`

The installed AMD HIP runtime defines:

```c
typedef struct hipDeviceProp_t {
  char name[256];
  hipUUID uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  ...
  int hipReserved[32];
  char gcnArchName[256];
  size_t maxSharedMemoryPerMultiProcessor;
  int clockInstructionRate;
  ...
} hipDeviceProp_t;
```

This proves that:

- `name` is the GPU name string field
- `gcnArchName` is the AMD architecture string field
- the code is definitely working with a runtime property blob that contains both the device name and the AMD arch string

It also proves that the earlier mapping of `[RBP+0x468]` and `[RBP+0x46C]` as direct public `hipDeviceProp_t` members is not safe to assume; those addresses are inside a custom local buffer or a locally shaped wrapper, not necessarily a direct public field offset.

This is why runtime proof is required.

---

## 2. Exact runtime points to instrument

The required points, per the disassembly, are:

### A. First device-property call

```asm
18000a9f3: movl    $0x5c0, %r8d
18000a9f9: movq    %r14, %rcx
18000a9fc: xorl    %edx, %edx
18000a9fe: callq   0x180064a00
18000aa03: movq    %r14, %rcx
18000aa06: movl    %r12d, %edx
18000aa09: callq   0x180065a20 ; hipGetDevicePropertiesR0600
18000aa0e: testl   %eax, %eax
```

This call is the first query for the current device. The local buffer is `-0x20(%rbp)` and is pointed to by `R14`.

### B. The final compatibility-validation block

```asm
18000ab91: leaq    -0x20(%rbp), %r12
18000ab95: movl    $0x5c0, %r8d
18000ab9b: movq    %r12, %rcx
18000ab9e: xorl    %edx, %edx
18000aba0: callq   0x180064a00
18000aba5: movq    %r12, %rcx
18000aba8: movl    %ebx, %edx
18000abaa: callq   0x180065a20 ; hipGetDevicePropertiesR0600
18000abaf: movq    %r12, %rcx
18000abb2: callq   0x180064e20
...
18000ac4d: movl    $0x31786667, %eax
18000ac52: xorl    0x468(%rbp), %eax
18000ac58: movzbl  0x46c(%rbp), %ecx
...
18000ac97: testl   %r14d, %r14d
18000aca2: movb    %al, 0x18009adb9
```

And again the diagnostic/failure path:

```asm
18000ad01: leaq    -0x20(%rbp), %rsi
18000ad05: movl    $0x5c0, %r8d
18000ad0b: movq    %rsi, %rcx
18000ad0e: xorl    %edx, %edx
18000ad10: callq   0x180064a00
18000ad15: movq    %rsi, %rcx
18000ad18: xorl    %edx, %edx
18000ad1a: callq   0x180065a20 ; hipGetDevicePropertiesR0600
18000ad22: callq   0x180064e20
...
18000ad4c: movb    $0x0, 0x18009adb9
18000ad53: cmpb    $0x0, 0x18009adb9
```

This is the exact set of locations to inspect.

---

## 3. Safe runtime capture strategy without changing the DLL

### Best approach: debugger breakpoints on the actual call sites

Use a native debug session (WinDbg or VS debugger) and break immediately after the calls.

Recommended breakpoints:

- `bp version.dll+0x18000AA09`
- `bp version.dll+0x18000AD1A`

Then, at each breakpoint, record:

- `@$rax` / `EAX` after return from `hipGetDevicePropertiesR0600`
- `@$rcx` and `@$rdx` for device index and property pointer
- `@$r14` and `@$r15`
- memory at the property pointer (`dq @$r14 L32`, `db @$r14 L256`)
- memory at the frame area around `[RBP+0x468]` / `[RBP+0x46C]`
- any difference between the apparently raw local buffer and the public `hipDeviceProp_t` data

For example:

```text
bp 0x18000AA09
bp 0x18000AD1A
g
r
? @$rax
? @$rcx
? @$rdx
? @$r14
? @$r15
dq @$r14 L32
db @$r14 L256
dq @$rbp+0x468 L8
db @$rbp+0x468 L32
```

This is the lowest-risk capture path because it does not edit the DLL and it captures the exact live values used by the gate.

### Why this is safest

- It does not modify the binary.
- It observes the live structure at the exact call sites.
- It captures the real architecture string, return code, and local variables used by the comparison block.
- It avoids guessing from a static offset map.

---

## 4. What to capture at each breakpoint

### At the first call (`0x18000AA09`)

Capture these live values:

1. device index in `RDX`
2. pointer to the local property buffer in `RCX`
3. `EAX` (HIP return code) immediately after the call
4. the first 256 bytes of the local property buffer at `R14`
5. the string contents at the local arch-name area around `[RBP+0x468]` and `[RBP+0x46C]`
6. `R15` before and after the copy from `[RBP+0x468]`
7. `RDI` immediately before the `strlen` / compare sequence

The relevant code path is:

```asm
18000aa12: movq    %r15, %rdi
18000aa15: movq    0x468(%rbp), %r15
18000aa29: movl    0x468(%rbp), %eax
18000aa36: movzbl  0x46c(%rbp), %ecx
```

This means the exact live bytes at `[RBP+0x468]` and `[RBP+0x46C]` must be captured at that moment.

### At the second call (`0x18000AD1A`)

Capture the same values again for the failure/diagnostic path:

- device index in `RDX`
- property buffer pointer in `RSI`
- `EAX` after `hipGetDevicePropertiesR0600`
- string bytes at `[RBP+0x468]` and `[RBP+0x46C]`
- the value ultimately sent to the helper call at `0x180035980`
- the final state of `0x18009ADB9`

This is the second real property snapshot and is important because the failure path clears the state byte after re-querying the device.

---

## 5. Exact values to record from the code path

The runtime capture should explicitly log or print the following values live:

### A. Device and property information

- `Device index = RDX`
- `hipGetDevicePropertiesR0600 return code = EAX`
- `GPU name = name[256]` from the property blob
- `gcnArchName / architecture string = the AMD arch string from the live property block`
- `major`, `minor` if available
- `multiProcessorCount` / `maxThreadsPerMultiProcessor` / `warpSize` / `clockRate` / `totalConstMem` / `totalGlobalMem`
- `memoryBusWidth` / `memoryClockRate` / `sharedMemPerBlock`

### B. Values around the compatibility gate

- `[RBP+0x468]` as a qword
- `[RBP+0x46C]` as a byte
- `R15` loaded from `[RBP+0x468]`
- `RDI` after masking / pointer reuse
- `R14D` from the helper result at `0x180065AE0`
- `SIL` after `strlen(...) == 7`
- `BL` after `gfx1200` compare
- `AL` before and after the overwrite at `0x18000ACA0`

### C. Exact boolean conditions

The live capture should record:

- `cond_prefix_1 = (dword_from_[RBP+0x468] ^ 0x31786667) | byte_[RBP+0x46C] ^ 0x31` equals zero or nonzero
- `cond_strlen7 = (strlen(...) == 7)`
- `cond_gfx1200 = (masked_qword == 0x30303231786667)`
- `cond_r14_is_zero = (R14D == 0)`
- `cond_final_state = (0x18009ADB9 == 1)`

This is the exact boolean logic that produces the state byte and the final unsupported path.

---

## 6. Existing DLL logging: does it already log enough?

From the disassembly and the earlier static analysis, there is no direct, proven format string for all of the requested fields.

What we can prove:

- There are generic helper calls via `0x180007c00` and other callbacks that are almost certainly used for logging/diagnostic output.
- There is a real unsupported-GPU message string at `0x180080A9C`.
- The code does not appear to have a dedicated, obvious print format for:
  - `HIP device %d`
  - `GPU name`
  - `arch`
  - `WGP count`
  - `CU count`
  - `VRAM`

The exact static evidence we have from the disassembly shows generic callbacks, not a dedicated `printf` block for the data requested above.

Therefore:

- The DLL likely logs some diagnostics, but not clearly all of the values needed for the compatibility gate.
- We cannot prove that the existing DLL already emits a full runtime property dump.

The output likely appears through a generic callback/logging channel, not a straightforward console path. The unsupported-GPU message at `0x180080A9C` is the clearest existing output, but it is not a full property dump.

---

## 7. Instrumentation options (without modifying version.dll)

If extra runtime evidence is needed, the safest options are:

### Option 1: debugger breakpoints only

This is the preferred approach.

- No DLL modification
- No binary dispersion
- Exact values captured live from the running process
- No risk of changing the compatibility decision

### Option 2: instrumentation in a separate experimental copy

Only if the runtime still cannot be observed with breakpoints.

The safe procedure would be:

1. create a separate copy of the DLL in a new directory
2. patch only the desired logging points in the copy
3. leave the original DLL untouched
4. reproduce the same load path with the copy only
5. verify that the copy still loads in the same way and does not alter the compatibility decision

The exact instrumentation point would be the `hipGetDevicePropertiesR0600` call site and the state-byte test/clear site, using breakpoints or output-only calls that do not alter the compatibility logic.

This is still a last resort, because the debugger route is both safer and more reliable.

---

## 8. Minimal instrumentation plan if needed

If a separate copy must be used, the most conservative instrumentation would be:

- place a `nop`-style or `jmp` hook only at a logging helper after the `hipGetDevicePropertiesR0600` return
- log the `RCX`, `RDX`, `RAX`, `R14`, `R15`, and the local string bytes
- do not change the actual control-flow result values for the compatibility check
- avoid patching the `0x18009ADB9` write itself

The instrumentation should be done at the call sites only, not at the state-bit gate or the unsupported message path.

Do not patch:

- the embedded gfx1032 ELF
- the fat-binary registration logic
- the global rejection flag write itself
- generic helper functions
- the unsupported message path

The goal is to collect values, not to bypass any gate.

---

## 9. What we can measure without modifying the DLL

We can measure, without editing the binary:

- the actual live GPU name from `hipGetDevicePropertiesR0600`
- the live architecture string from the property block
- the HIP return code from `EAX`
- the device index from `RDX`
- the pointer to the local property buffer and its contents
- the local string bytes around `[RBP+0x468]` and `[RBP+0x46C]`
- the helper result in `R14D`
- the boolean conditions used by the comparison block
- whether the state byte at `0x18009ADB9` is set or cleared at runtime

---

## 10. What remains unknown without the runtime dump

The remaining unknowns are:

- the exact live values for the RX 6600 in the compatibility block
- whether the branch is driven by architecture string length, a masked qword comparison, or a helper return value
- whether the state byte is set because of a property mismatch or because a later diagnostic path clears and re-tests it
- whether the final compatibility gate is a hardware requirement, a driver requirement, or a software policy check

---

## 11. Safest way to obtain the missing runtime values

The safest way is:

1. use a debugger and break on the two `hipGetDevicePropertiesR0600` call sites
2. capture the memory buffer and the local `RBP`-relative values at the time of the call and immediately after the call
3. confirm the string and integer values used by the compare block
4. dump the final state byte and the branch targets at `0x18000ACA2`, `0x18000AD4C`, and `0x18000AD53`
5. only after that decide whether a single-instruction compatibility patch is even justified

This is the least invasive and most defensible route.

---

## 12. Is a debugger required?

Yes, for the missing ground-truth values a debugger is the safest and most direct tool.

Without a debugger, the static analysis cannot distinguish between:

- a valid `gfx1032` device path,
- a property mismatch path,
- a helper-result path,
- or a later state-byte path that is being cleared and re-tested.

A debugger is therefore required if the goal is to prove which values actually cause the gate to reject the RX 6600.

---

## 13. Can a one-instruction compatibility patch be proven after obtaining those values?

Only after collecting the live values can we prove whether a one-instruction patch is truly safe.

At the moment the answer is: not yet.

We still need the live `hipDeviceProp_t` / local property dump to answer the key question:

- is the rejection caused by the architecture-string comparison,
- by the helper return value in `R14D`,
- or by a different property / state-transition path?

Until that is known, a compatibility patch should not be recommended.

---

## Final recommendation

The correct next step is not to patch the DLL; it is to capture the actual live values at the call sites, then prove which branch in the compatibility check is failing for the RX 6600.

The best evidence is from a debugger, not from further static guessing.
