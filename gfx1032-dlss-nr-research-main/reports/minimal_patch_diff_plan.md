# Minimal patch diff plan for the gfx1032 compatibility gate

## Executive summary

This report re-checks the earlier conclusion that the minimal patch site is `0x18000AC97`..`0x18000ACA2`.

The evidence does not support that conclusion as a safe, minimal, proven fix.

The reason is that the key state byte at `0x18009ADB9` is not only written at `0x18000ACA2`; it is also cleared in a separate failure path beginning at `0x18000AD01` and reaching `0x18000AD4C`, and the later gate at `0x18000AD53` and `0x18000D205` is the actual runtime check. In other words, the final unsupported message depends on a state bit that is set in one validation block and cleared in another, and the earlier write is not necessarily the only or even the decisive decision point.

This report therefore treats the earlier `ACA2` patch idea as a candidate only, not as a proven minimal fix. The evidence is more consistent with a broader runtime validation gate in which:

- one block initializes or sets the state byte,
- another failure block clears it,
- later code tests it,
- and the final unsupported path is controlled by an earlier validation decision that still needs a live property dump to prove.

---

## 1. Proven facts from the installed ROCm 7.2 HIP headers

The local system header used here is:

`C:\Program Files\AMD\ROCm\7.2\include\hip\hip_runtime_api.h`

The installed AMD HIP API defines the structure as:

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

This proves the following:

- `name[256]` exists at the start of the object.
- `gcnArchName[256]` exists later in the structure.
- `gcnArchName` is the AMD-specific architecture string field in the ROCm ABI.

Important consequence:

- the earlier idea that `0x468` / `0x46c` are the actual `gcnArchName` field offsets from the start of `hipDeviceProp_t` is not correct.
- `gcnArchName` is much later in the structure, not at `0x468`.

Therefore, the assembly reads at `[RBP+0x468]` and `[RBP+0x46C]` are not direct top-level `hipDeviceProp_t` fields from the public struct definition. They are reads from a custom local stack buffer or custom property wrapper built by the program, not a direct `hipDeviceProp_t` pointer at those offsets.

This is the key correction to the earlier analysis.

---

## 2. Exact control flow from the clear site and failure path

The exact relevant sequence from the DLL is:

```asm
18000ad01: leaq    -0x20(%rbp), %rsi
18000ad05: movl    $0x5c0, %r8d
18000ad0b: movq    %rsi, %rcx
18000ad0e: xorl    %edx, %edx
18000ad10: callq  0x180064a00
18000ad15: movq    %rsi, %rcx
18000ad18: xorl    %edx, %edx
18000ad1a: callq  0x180065a20 ; hipGetDevicePropertiesR0600
18000ad1f: movq    %rsi, %rcx
18000ad22: callq  0x180064e20
18000ad27: leaq    0x90092(%rip), %rcx ; 0x18009adc0
18000ad2e: movq    %rsi, %rdx
18000ad31: movq    %rax, %r8
18000ad34: callq  0x180035980
18000ad39: leaq    0x468(%rbp), %rdx
18000ad40: leaq    0x725e8(%rip), %rcx ; 0x18007d32f
18000ad47: callq  0x180007c00
18000ad4c: movb    $0x0, 0x90066(%rip) ; 0x18009adb9
18000ad53: cmpb    $0x0, 0x9005f(%rip) ; 0x18009adb9
18000ad5a: je      0x18000ba43
18000ad60: cmpb    $0x0, 0x8f6ba(%rip) ; 0x18009a421
18000ad67: jne    0x18000b8c7
18000ad6d: movb    $0x1, 0x8f6ad(%rip) ; 0x18009a421
...
```

This proves that `0x18000AD4C` is not the normal runtime unsupported gate. It is a failure/diagnostic branch that explicitly clears the state byte after calling `hipGetDevicePropertiesR0600` again and then doing additional logging / formatting work.

The exact flow backwards from `AD4C` is:

- `AD53` checks whether `0x18009ADB9` is zero.
- If zero, it branches away from the unsupported path.
- If nonzero, it continues to later checks.
- There is a separate failure path at `AD01`..`AD4C` that clears the byte before it reaches the later `AD53` gate.

Therefore, the path that writes and clears the byte is not the same path that later triggers the unsupported message.

---

## 3. The actual runtime decision path around ACA2 and the later gate

The critical code region is:

```asm
18000ac4d: movl    $0x31786667, %eax       # imm = 0x31786667
18000ac52: xorl    0x468(%rbp), %eax
18000ac58: movzbl  0x46c(%rbp), %ecx
18000ac5f: xorl    $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi # imm = 0xFFFEFFFFFFFFFFFF
18000ac6c: andq    0x468(%rbp), %rdi
18000ac73: orl     %eax, %ecx
18000ac75: jne     0x18000ac87
18000ac77: movq    %r15, %rcx
18000ac7a: callq  0x180064e20
18000ac7f: cmpq    $0x7, %rax
18000ac83: sete    %sil
18000ac87: movabsq $0x30303231786667, %rax # imm = 0x30303231786667
18000ac91: cmpq    %rax, %rdi
18000ac94: sete    %bl
18000ac97: testl   %r14d, %r14d
18000ac9a: sete    %al
18000ac9d: orb     %sil, %bl
18000aca0: movb    $0x1, %al
18000aca2: movb    %al, 0x90111(%rip) ; 0x18009adb9
18000aca8: je      0x18000acb6
18000acaa: leaq    0x7667f(%rip), %rcx
18000acb1: callq  0x180011460
18000acb6: testb   %sil, %sil
18000acb9: movq    0x990(%rbp), %r12
18000acc0: je      0x18000acd1
18000acc2: leaq    0x72491(%rip), %rcx
18000acc9: movq    %r15, %rdx
18000accc: callq  0x180007c00
18000acd1: cmpl    $0x0, 0x9d0(%rbp)
18000acd8: jns     0x18000acec
18000acda: testb   $0x1, %r13b
18000acde: je      0x18000acec
18000ace0: leaq    0x71e15(%rip), %rcx
18000ace7: callq  0x180007c00
18000acec: testb   %bl, %bl
18000acee: jmp     0x18000ad53
18000acf0: leaq    0x72638(%rip), %rcx
18000acf7: movq    %r15, %rdx
18000acfa: callq  0x180007c00
18000acff: jmp     0x18000ad53
```

This is crucial:

- `BL` is set from `cmpq %rax, %rdi` against the magic value `0x30303231786667`.
- `SIL` is set from `strlen(...) == 7`.
- `AL` is set from `testl %r14d,%r14d` -> `sete %al`, but then immediately overwritten by `movb $0x1,%al`.
- `R14D` is the return value from the helper call at `0x180065AE0`, i.e. a numeric runtime capability/status result.
- `RDI` is the masked qword loaded from local arch-string memory at `[RBP+0x468]` and then compared to the special `gfx1200` constant.
- `R15` is the qword fetched from `[RBP+0x468]` before masking; it is then used as a qword string value / name word for comparison.
- `[RBP+0x468]` and `[RBP+0x46C]` are the first 8 bytes of a local string / arch-name of the current runtime property buffer.
- `[RBP+0x46C]` is not a top-level `hipDeviceProp_t` field; it is the byte immediately following the first 4 bytes of that local string data.

The data flow proves that the code is testing a string buffer, not an `int` object. The `gfx1200` comparison is a qword compare against a packed ASCII word; there is no direct proof that this is the `gcnArchName` field offset from the public HIP struct.

---

## 4. Semantics of the registers and memory involved

### BL

`BL` is set by:

```asm
18000ac87: movabsq $0x30303231786667, %rax
18000ac91: cmpq    %rax, %rdi
18000ac94: sete    %bl
```

This means:

- if the masked 64-bit value from the string buffer equals the little-endian qword for `gfx1200`, then `BL = 1`.
- otherwise `BL = 0`.

This is not a direct generic “architecture accepted” flag; it is a narrow special-case check for one specific architecture pattern.

### SIL

`SIL` is set by:

```asm
18000ac77: movq    %r15, %rcx
18000ac7a: callq  0x180064e20
18000ac7f: cmpq    $0x7, %rax
18000ac83: sete    %sil
```

This means `SIL` = 1 only if `strlen(...) == 7` on the string buffer.

Therefore `SIL` is the string-length result, not a generic success bit.

### AL

`AL` is created by:

```asm
18000ac97: testl   %r14d, %r14d
18000ac9a: sete    %al
18000ac9d: orb     %sil, %bl
18000aca0: movb    $0x1, %al
18000aca2: movb    %al, 0x18009adb9
```

This is the most important observation:

- `AL` is set based on the result of `R14D == 0`.
- then it is overwritten with `1` unconditionally.
- therefore the final write to `0x18009ADB9` is not controlled by the earlier `R14D` test.

### R14D

`R14D` is set by:

```asm
18000abcb: callq  0x180065ae0
18000abd0: movl    %eax, %r14d
18000abd3: testl   %eax, %eax
```

This is a numeric helper return value. It is used in `testl %r14d, %r14d`, but it is then discarded by the unconditional overwrite to `AL` before the state byte write.

Therefore `R14D` is a runtime numeric capability/result value, but it does not directly decide the final `0x18009ADB9` write.

### RDI

`RDI` holds the string pointer / qword value in different phases of the function.

- At `0x18000AA12`, `movq %r15, %rdi` saves the prior pointer.
- At `0x18000AA48`, `movq %rdi, %rcx` and then `strlen` is called on it.
- At `0x18000AC62`, `movabsq $-0x1000000000001, %rdi` and `andq 0x468(%rbp), %rdi` convert the string qword into a masked qword for the later compare.

Thus `RDI` is being repurposed: first it is a pointer to the local string, then it is the masked qword representation of the string bytes.

### R15

At the start of the relevant path:

```asm
18000a8b9: leaq    0x468(%rbp), %r15
...
18000aa15: movq    0x468(%rbp), %r15
```

The code is loading a qword value from `[RBP+0x468]` into `R15` and then later comparing it to the magic `gfx1200` word. This proves that `[RBP+0x468]` is a qword-sized value corresponding to the first 8 bytes of the current string / arch-name data.

### `[RBP+0x468]` and `[RBP+0x46C]`

These are the exact bytes being checked:

- `[RBP+0x468]` contains the first 8 bytes of the local string or name buffer.
- `[RBP+0x46C]` is the byte immediately after the first four bytes of that buffer.

The code is explicitly interpreting the data as ASCII:

```asm
movl $0x31786667, %ecx ; 'gfx1'
xorl %ecx, %eax
movzbl 0x46c(%rbp), %ecx
xorl $0x31, %ecx
```

This is not a numeric capability read. It is an ASCII-string comparison on a local architecture-name buffer.

---

## 5. Why the earlier ACA2 conclusion is not sufficient

The earlier conclusion that `0x18000AC97`..`0x18000ACA2` is the “minimal patch” is not proven safe because:

1. `0x18009ADB9` is set here, but it is also cleared at `0x18000AD4C` in a failure/diagnostic path.
2. There are multiple repeated checks of that byte and multiple code paths around it.
3. The final states `BL`, `SIL`, `AL` are not simple accepted/rejected booleans in the way the earlier report implied.
4. The state byte write is unconditionally overwritten with `1` just before the store, which means the final bit can be valid even when the earlier `R14D` test or `string` check would have suggested a different outcome.
5. The patch site needs to match the actual runtime path for the RX 6600, which requires a live property dump from the device itself; without that dump there is no proof that patching this block will be safe.

Therefore, the earlier `ACA2` conclusion is not sufficiently proven as the minimal safe patch site.

---

## 6. Patch analysis

### Candidate A: patch the final write to `0x18009ADB9`

- File offset: binary offset corresponding to the block starting at `0x18000AC97`
- Virtual address: `0x18000AC97`..`0x18000ACA2`
- Original bytes:
  - `testl %r14d,%r14d`
  - `sete %al`
  - `orb %sil,%bl`
  - `movb $0x1,%al`
  - `movb %al,0x90111(%rip)`
- Original instruction: `movb $0x1, %al ; movb %al, 0x18009adb9`
- Proposed bytes: change the final state write to leave the byte at 0 for the supported `gfx1032` case
- Proposed instruction: conditional skip of the state-byte store or branch to `AD53` without writing `1`
- Exact branch condition being changed: the condition that leads to `movb %al, [0x18009ADB9]`
- Why gfx1032 reaches the old condition: because the `gfx1` / string-length / `gfx1200` logic enters the block and reaches the unconditional state-byte write.
- Why gfx1032 should reach the new path: because the existing embedded `gfx1032` object is valid, and the check is a runtime validation gate rather than a missing-object issue.
- Why existing supported GPUs remain unchanged: if the patch is only applied for the `gfx1032` string case, the higher `gfx1100` / `gfx1101` / `gfx1102` / `gfx1200` logic would remain unchanged.
- Risk: this is not proven safe because we do not yet know whether the block is reached for `gfx1032` in the same way on real hardware; also, the clear path at `AD4C` can still override the bit later.

### Candidate B: patch the earlier mismatch branch around `0x18000AC75`

- File offset: block starting at `0x18000AC4D`
- Virtual address: `0x18000AC4D`..`0x18000AC75`
- Original instruction: `jne 0x18000ac87`
- Proposed instruction: branch to a permitted path for the `gfx1032` case
- Exact branch condition being changed: mismatch of the first 4 bytes of the local string with `gfx1` or the next byte with `0x31`
- Why gfx1032 reaches the old condition: because the first 4 bytes are `gfx1` but the fifth byte is `0x30` (`'0'`), which does not satisfy the simple comparison being performed.
- Why gfx1032 should reach the new path: because `gfx1032` is a valid architecture string and should not be rejected based on a naive byte-value pattern.
- Why existing supported GPUs remain unchanged: the patch can be scoped only to the specific `gfx1032` branch and leave other families untouched.
- Risk: this is a more invasive patch to the string-validation logic and could affect other strings if not constrained carefully.

### Candidate C: patch the `AD53` gate itself

- File offset: `0x18000AD53`..`0x18000AD5A`
- Virtual address: `0x18000AD53` and `0x18000AD5A`
- Original instruction: `cmpb $0x0, [0x18009ADB9] ; je 0x18000BA43`
- Proposed instruction: alter the conditional so that the `gfx1032` state does not trigger the unsupported path
- Exact branch condition being changed: whether the flag byte is zero
- Risk: this is a direct gate bypass. It is the broadest and least safe patch because it changes the actual unsupported-GPU gate rather than the earlier validation condition.

### Conclusion on patch candidates

No single candidate is proven safe with the evidence currently available.

The only candidate that is closest to a “minimal single-branch patch” is the earlier conditional mismatch check or the final state-byte write, but neither is proven sufficient on real RX 6600 hardware because we still do not have the exact live property values for the device and we do not yet know which earlier branch is truly responsible for the `gfx1032` case.

---

## 7. Additional logic that assumes gfx1100 / gfx1101 / gfx1102 / gfx1200 / RDNA3 / RDNA4 / RX7000 / RX9000

The disassembly shows a repeated pattern of newer-architecture assumptions:

- `gfx1200` is compared directly to the masked qword.
- the program explicitly contains `gfx1100`, `gfx1101`, `gfx1102`, and `gfx1200` family assumptions in the architecture list.
- the code has later logic that is tuned to higher RDNA3 / RDNA4 family assumptions.
- the unsupported message/text suggests a planned restriction for certain AMD cards.

These are all evidence that the software is not purely “select the embedded `gfx1032` object if available”; it is using a policy gate based on a family/feature model that is more recent than `gfx1032`.

---

## 8. Final verdict

VERDICT:

- Proven minimal patch candidate: None proven safe.
- Confidence level: Low.
- Remaining unknown: the exact live device-property data for a real RX 6600, the exact branch path that sets the rejection bit in this case, and whether the `ACA2` state write is actually the effective gate or only one of several state transitions.
- Exact runtime test required:
  1. Capture the live `hipDeviceProp_t` for the RX 6600 from `hipGetDevicePropertiesR0600`.
  2. Dump the actual contents of the local `char[]` at `[RBP+0x468]` / `[RBP+0x46C]` and the helper return value in `R14D`.
  3. Single-step or break at `0x18000ACA2`, `0x18000AD4C`, `0x18000AD53`, and `0x18000D205` on the real GPU to confirm which branch is taken for `gfx1032`.
  4. Only then decide whether a minimal conditional branch patch is safe.

This is intentionally conservative. The evidence currently supports the conclusion that the embedded `gfx1032` code object is present and valid, but it does not prove a safe, minimal branch patch for the current rejection gate without a live device-property trace.

---

## Bottom line

The direct verdict is:

- the earlier `ACA2` patch idea is not proven correct as the minimal fix,
- the `AD4C`/`AD53` flow proves there is a broader state-machine gate involved,
- the `0x468` / `0x46c` values are local string bytes, not direct `hipDeviceProp_t` offsets,
- and no safe single-branch patch should be recommended without a real RX 6600 runtime trace.
