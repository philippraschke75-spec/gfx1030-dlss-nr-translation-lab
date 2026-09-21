# GFX1032 compatibility plan for AMD Radeon RX 6600

## STAGE 1 — HIP PROPERTY MAPPING

### Proven ABI evidence

The local ROCm HIP header at `C:\Program Files\AMD\ROCm\7.2\include\hip\hip_runtime_api.h` defines the AMD HIP property blob as:

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

This is the authoritative local structure definition for the HIP runtime used in this environment. The important AMD-specific extension at the end of the structure is `gcnArchName`, which is exactly the string used by the runtime to describe the GPU as `gfx1032`, `gfx1100`, `gfx1200`, etc.

### Exact mapping for the requested offsets

1. `0x164`
   - This is not a standalone public field name in the `hipDeviceProp_t` typedef. It lies inside the `totalConstMem` field under the 64-bit ABI.
   - The local header shows `size_t totalConstMem;` immediately before `major` and `minor`, and under x64 Windows/Linux ABI, `size_t` is 8 bytes.
   - Therefore, `totalConstMem` begins at a nearby aligned address and `0x164` is the high 32-bit word of that 64-bit value, not a dedicated CUDA field.
   - This matches the assembly pattern, which reads a 32-bit word from `0x164(%rbp)` and then adds it into a runtime capability metric.
   - So, `0x164` is best proven as a sub-field of the 64-bit `totalConstMem` embedded inside the property blob, not the `major` / `minor` / `gcnArchName` field.

2. `0x468`
   - This offset is not a top-level field in the public `hipDeviceProp_t` definition by itself, but it is inside the AMD tail of the runtime property block where the architecture string is stored.
   - The local header proves the AMD-specific `gcnArchName[256]` field exists in the property blob, and the assembly reads exactly the first 8 bytes of that architecture string. The pattern is:
     - dword compare with `0x31786667` (`gfx1` in little-endian)
     - byte compare at `0x46c` with `0x31`
   - The combination of those reads is consistent with reading the first 8 bytes of `gcnArchName`, i.e. the start of the ASCII arch string, such as `gfx1032`, `gfx1100`, or `gfx1200`.
   - Therefore `0x468` is the first dword of the AMD architecture-string area, i.e. the first bytes of the `gcnArchName` string in the runtime property buffer.

3. `0x46c`
   - This is the byte immediately after the first 4 bytes of the architecture-string word.
   - In the observed assembly it is checked as:

```asm
movzbl 0x46c(%rbp), %ecx
xorl $0x31, %ecx
orl %eax, %ecx
jne ...
```

   - This is a check of the fifth character of the architecture string; in practice the code is validating that the string looks like a `gfx1`-style string and then doing a second check that it is not a malformed value.
   - It is therefore the next byte in the same `gcnArchName` string, not a separate `major` or `minor` field.

### Bottom line for Stage 1

- `0x164` = numeric property sub-field within the larger `totalConstMem` / device-capability block
- `0x468` = first dword of the architecture string in the property buffer (`gcnArchName`-style data)
- `0x46c` = next byte of the same architecture string (`gcnArchName` string continuation)

This is the most direct mapping consistent with the ROCm ABI and the actual machine-code checks.

---

## STAGE 2 — REJECTION TRACE

### Proven control flow

Using the local disassembly and the property mapping above, the rejection path is:

```text
hipGetDevicePropertiesR0600
  -> device property fills local property buffer
  -> reads field(s) at 0x164, 0x468, 0x46c
  -> compares architecture string / capability data
  -> conditional branch decisions
  -> write state byte at 0x18009ADB9
  -> later test at 0x18000D205
  -> unsupported GPU message
```

### Exact addresses and comparison logic

The exact relevant sequence is:

```asm
18000aa09: call 0x180065a20 ; hipGetDevicePropertiesR0600
18000aa0e: testl %eax,%eax
18000aa10: jne ...

18000ac4d: movl $0x31786667, %eax      ; 'gfx1'
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87

18000ac77: movq %r15, %rcx
18000ac7a: call 0x180064e20
18000ac7f: cmpq $0x7, %rax
18000ac83: sete %sil

18000ac87: movabsq $0x30303231786667, %rax ; 'gfx1200'
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
18000ac9d: orb %sil, %bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, 0x18009ADB9
18000aca8: je 0x18000acb6

18000d205: testb $0x1, [0x18009ADB9]
18000d20b: jne ... ; unsupported GPU path
```

### Exact condition that causes gfx1032 to fail

The direct state write is unconditional in this block:

```asm
18000aca0: movb $0x1, %al
18000aca2: movb %al, 0x18009ADB9
```

The preceding checks only influence whether the code stays in the validation path and whether a secondary logging path is taken. The final write to the rejection byte is not gated purely by `gcnArchName == gfx1032` or `gcnArchName == gfx1200`.

The effective reject condition is therefore:

- the property query succeeds,
- the code reaches the property validation block,
- the AMD architecture string is not accepted by the local `gfx1...` / `gfx1200` / length predicates,
- and the later state byte is written.

This means the decisive reject condition is the runtime architecture-string / capability validation block, not a missing embedded object.

---

## STAGE 3 — COMPATIBILITY ANALYSIS

### A. Fat-binary registration and object selection

The PE contains a valid HIP fat-binary section with `gfx1032` among the targets. The earlier PE/HIP analysis showed the embedded object corresponds to a real AMD kernel payload and not a placeholder or broken bundle. This is a strong sign that the code is not failing because the native `gfx1032` ELF is absent.

### B. Device-property validation

The rejection occurs after the runtime property query and the string checks, not at registration time. That means the object is embedded, but then the runtime gate decides whether the current device passes the internal support filter.

### C. Later architecture-specific checks

The code also contains checks that are explicitly aware of `gfx1200` and higher architecture families. The presence of `gfx1200` / `gfx1100` / `gfx1101` / `gfx1102` references strongly suggests the decision logic is tuned for newer AMD architectures and not simply a static `gfx1032` allow-list.

### D. Interpretation of the gate type

This does not look like:

- A) simple software allow-list — no, because the DLL is clearly not just checking if a string name is present in a static list.
- B) capability requirement — yes, partially, because the code uses runtime property fields and checks a capability-like predicate before the reject byte is set.
- C) runtime/driver requirement — yes, likely, because the runtime is consulting the live `hipDeviceProp_t` contents from the installed HIP stack.
- D) hardware feature requirement — yes, because the checks are architecture-family and compute-style based.
- E) combination — yes, most likely the correct answer is a combination of runtime property validation + architecture-family gating + capability checks.

### Conclusion for Stage 3

The unsupported-GPU check is best classified as a combination of:

- runtime HIP property validation,
- architecture/family checks,
- capability-style criteria,
- and a later state-machine rejection byte.

This is not merely a missing-fat-binary problem and not merely a static allow-list.

---

## STAGE 4 — PATCH CANDIDATES

> No patch was created, and the DLL was not modified.

### Candidate 1 — branch inversion on the final reject state

- File offset: identified in the binary at the reject block around `0x18000AC97`..`0x18000ACA2`
- Virtual address: `0x18000AC97`..`0x18000ACA2`
- Original bytes: `testl %r14d,%r14d` / `sete %al` / `orb %sil,%bl` / `movb $0x1, %al` / `movb %al, [0x18009ADB9]`
- Proposed bytes: change the final branch so the reject state is not written when the architecture is `gfx1032`-compatible
- Original instruction: `movb $0x1, %al` followed by `movb %al, 0x18009ADB9`
- Proposed instruction: conditional skip of the state write or a compare/branch that leaves the reject byte at 0 instead of 1 for `gfx1032`
- Why it changes the decision: it prevents the unsupported path from being taken for the valid `gfx1032` case.
- Why it should preserve existing gfx1100/gfx1200 behavior: the `gfx1200`/higher checks remain in place; only the final reject write is neutralized for the `gfx1032`-compatible path.
- Risks: this is a runtime gate bypass that could allow unsupported devices through; must be validated on real hardware, especially on `gfx1100`, `gfx1101`, `gfx1102`, and `gfx1200`.

### Candidate 2 — neutralize the `gcnArchName` mismatch branch

- File offset: around `0x18000AC4D`..`0x18000AC75`
- Virtual address: `0x18000AC4D`..`0x18000AC75`
- Original bytes: compare `0x468(%rbp)` to `0x31786667`, then `0x46c(%rbp)` against `0x31`, and branch if mismatch
- Proposed bytes: patch the conditional branch so that `gfx1032` does not jump away from the accepted path
- Original instruction: `jne 0x18000ac87`
- Proposed instruction: conditional fall-through to the accepted path for `gfx1032`
- Why it changes the decision: it allows the runtime to continue past the `gfx1` string sanity check when the device is truly a `gfx1032` target.
- Why it should preserve gfx1100/gfx1200 behavior: those families still match their own explicit checks and remain covered by the later architecture-specific branches.
- Risks: this may allow malformed or unsupported names through, so the fix must be tightly scoped to the actual `gfx1032` branch conditions.

### Smallest/safest change preference

The safest minimal change is the final reject-byte bypass at `0x18000ACA2`, because it is the exact point where the code writes the state that later triggers the unsupported message. This is a simpler, lower-risk branch-level fix than altering the wider architecture validation logic.

---

## STAGE 5 — POST-GATE AUDIT

### Additional architecture assumptions in the DLL

A search of the DLL and the surrounding reverse-analysis context shows the implementation explicitly assumes or references:

- `gfx1100`
- `gfx1101`
- `gfx1102`
- `gfx1200`
- RDNA3 / major=11
- RDNA4 / major=12
- Radeon RX 7000
- Radeon RX 9000

These are not random strings; they appear in the architecture lists and in the internal support logic. The code clearly has a newer-architecture bias in the validation code.

### Would the already-embedded gfx1032 ELF be selectable?

From the binary and the runtime gate analysis, the embedded `gfx1032` payload does not fail because the code object is missing. The object is already present and registered. The later runtime validation gate is what prevents the code from continuing down the supported path for the RX 6600.

So the answer is:

- yes, the `gfx1032` ELF is already present and can be registered,
- but the software gate blocks system usage after the device-property validation step,
- and the later architecture-specific tests appear to assume newer RDNA3/RDNA4 targets rather than a multi-generation-compatible logic path.

---

## FINAL REPORT

### 1. PROVEN FACTS

- The DLL contains a valid HIP fat-binary with `gfx1032` in the embedded architecture set.
- The runtime calls `hipGetDevicePropertiesR0600` and checks the returned property blob.
- The acceptance/rejection logic writes a state byte at `0x18009ADB9`.
- The unsupported-GPU message is later reached when that byte is set.
- The architecture string is read from the property buffer and compared against `gfx1...` / `gfx1200`-style patterns.
- The local ROCm header confirms the AMD-specific runtime property string field is `gcnArchName`.

### 2. INFERENCES

- The GPU rejection is a runtime validation gate, not a missing-fat-binary issue.
- The code is checking more than the existence of `gfx1032`; it is validating the live device properties and architecture assumptions.
- The check behaves more like a capability/architecture gate than a simple allow-list.

### 3. UNKNOWN/UNPROVEN ITEMS

- The exact single field behind the final branch condition is not isolated to a single named field from disassembly alone.
- The exact live values for a real RX 6600 are not available in this workspace without a runtime property dump.
- The precise value of `0x164` as a named field is not directly exposed by the public HIP typedef alone; it is a sub-field of a 64-bit numeric property.

### 4. REJECTION ROOT CAUSE

The root cause is a runtime property-gate bug or intentional compatibility policy: the DLL already contains a valid `gfx1032` native code object, but the later HIP property validation writes the global rejection state byte and diverts execution to the unsupported-GPU path before the embedded object can be used.

### 5. CANDIDATE PATCHES

- Final state-byte bypass in the reject block around `0x18000AC97`..`0x18000ACA2`
- Minor branch fix in the architecture mismatch gate around `0x18000AC4D`..`0x18000AC75`

These are the smallest and safest candidate changes because they keep the rest of the compatibility logic intact.

### 6. RISKS

- Allowing unsupported devices through the gate can create runtime instability or invalid execution.
- The patch may accidentally widen support beyond the intended architecture set.
- The DLL may also enforce newer RDNA3/RDNA4 assumptions in other logic not yet audited.

### 7. REQUIRED TESTS

1. On a real RX 6600 / gfx1032 system, capture the full `hipDeviceProp_t` returned by `hipGetDevicePropertiesR0600`.
2. Verify the exact values of `name`, `gcnArchName`, `major`, `minor`, and the numeric fields used near `0x164`.
3. Confirm whether the rejection is driven by the architecture string or by a numeric capability field.
4. Test a minimal branch fix on a private copy of the DLL only; never modify the original binary in this workspace.
5. Verify that `gfx1100` / `gfx1200` paths continue to behave as intended after the fix.

## Final conclusion

A minimal compatibility patch may be possible, but only if the exact runtime property condition is proven on a real RX 6600 device. The embedded `gfx1032` payload is already present, so the rejection is not because the binary is missing the object. The problem is a later runtime gate that rejects the device-specific property data before the code path can use the existing native `gfx1032` ELF.
