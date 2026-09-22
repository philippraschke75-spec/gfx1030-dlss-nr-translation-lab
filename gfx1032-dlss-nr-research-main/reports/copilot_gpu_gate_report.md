# GPU gate analysis for version.dll

## Executive summary

The binary under analysis is the root-level `version.dll` in the workspace. The PE headers and section table are consistent with the user-supplied layout: `.text` begins at VA `0x1000`, `.rdata` at `0x66000`, `.data` at `0x96000`, `.hipFatB` at `0xA2000`, and `.hip_fat` at `0xA3000`. The embedded HIP fat binary really begins at raw file offset `0x9B600`, and the GFX1032 ELF payload begins at `0x9C600` with a real AMDGPU object inside it.

The critical finding is that the DLL does not explicitly reject `gfx1032` in the raw fat-binary / architecture string path. The embedded fat binary contains the expected GFX targets, and the architecture list in `.rdata` includes `gfx1032` alongside `gfx1200` and `gfx1100` family entries. The actual rejection is triggered later by a runtime status byte at `0x18009adb9`, which is set in the block ending at `0x18000aca2` and tested at `0x18000d205`; when set, the code loads the unsupported-GPU format string at `0x180080a9c` and emits:

`Unsupported GPU (%s): this build runs on Radeon RX 9000 (RDNA4) and RX 7000 (RDNA3) only.`

This is a later initialization/capability gate, not a missing or malformed GFX1032 object registration problem.

## Exact DLL being analyzed

- File: `C:\DLSSNRResearch\version.dll`
- Size: ~7,304,459 bytes
- PE format: COFF x86-64, DLL
- Section layout validated with `llvm-readobj -S`:
  - `.text` VA `0x1000` / raw `0x400`
  - `.rdata` VA `0x66000` / raw `0x65400`
  - `.data` VA `0x96000` / raw `0x95200`
  - `.hipFatB` VA `0xA2000` / raw `0x9B400`
  - `.hip_fat` VA `0xA3000` / raw `0x9B600`

## PE and HIP fat-binary verification

Validated via `llvm-readobj` from the ROCm toolchain on the actual binary:

```text
Section {
  Name: .hipFatB
  VirtualAddress: 0xA2000
  PointerToRawData: 0x9B400
}
Section {
  Name: .hip_fat
  VirtualAddress: 0xA3000
  PointerToRawData: 0x9B600
}
```

The fat binary starts at raw offset `0x9B600` and includes architecture strings such as:

- `hipv4-amdgcn-amd-amdhsa--gfx1032`
- `hipv4-amdgcn-amd-amdhsa--gfx11-generic`
- `hipv4-amdgcn-amd-amdhsa--gfx1100`
- `hipv4-amdgcn-amd-amdhsa--gfx1101`
- `hipv4-amdgcn-amd-amdhsa--gfx1102`
- `hipv4-amdgcn-amd-amdhsa--gfx1200`
- `hipv4-amdgcn-amd-amdhsa--gfx1201`
- `hipv4-amdgcn-amd-amdhsa--gfx9-generic`

This confirms the fat binary registration path is real and includes `gfx1032`.

## Relevant addresses and disassembly

### 1) `0x180022550` (initialization entrypoint)

This is the initialization routine that calls the device-probe helper and stores state. The relevant pattern is:

```asm
18000a9f3:  movl    $0x5c0, %r8d
18000a9f9:  movq    %r14, %rcx
18000a9fc:  xorl    %edx, %edx
18000a9fe:  callq   0x180064a00
18000aa03:  movq    %r14, %rcx
18000aa06:  movl    %r12d, %edx
18000aa09:  callq   0x180065a20    ; hipGetDevicePropertiesR0600
18000aa0e:  testl   %eax, %eax
18000aa10:  jne     0x18000a9e1
```

This is the evidence that the device property query is made with:

- `RCX = destination HIP device properties structure`
- `RDX = device index`
- `EAX` from the HIP call is checked immediately for success

This matches the runtime behavior described in the notes.

### 2) `0x180024700`

This function is not the GPU gate itself; it is a large generic allocation/initialization helper called from the setup path. It is not the branch that ultimately rejects `gfx1032`.

### 3) `0x1800658B0`

This address is a jump table / import thunk region and not a logic decision point. It is the 0x1800658B0 family of stubs that jumps through the IAT pointers, including the HIP import table. It is not the actual GPU rejection logic.

```asm
180065a20: jmpq *0x3280a(%rip) ; 0x180098230
```

This is the `hipGetDevicePropertiesR0600` thunk. The import table confirms `hipGetDevicePropertiesR0600` is imported from `amdhip64_7.dll` and the IAT entry is at `0x180098230`.

### 4) `0x18000A800` through `0x18000AD80`

The actual runtime capability loop is here.

```asm
18000ab8f:  xorl    %esi, %esi
18000ab91:  leaq    -0x20(%rbp), %r12
18000ab95:  movl    $0x5c0, %r8d
18000ab9b:  movq    %r12, %rcx
18000ab9e:  xorl    %edx, %edx
18000aba0:  callq   0x180064a00
18000aba5:  movq    %r12, %rcx
18000aba8:  movl    %ebx, %edx
18000abaa:  callq   0x180065a20 ; hipGetDevicePropertiesR0600
18000abaf:  movq    %r12, %rcx
18000abb2:  callq   0x180064e20
18000abb7:  leaq    0x90202(%rip), %rcx ; 0x18009adc0
18000abbe:  movq    %r12, %rdx
18000abc1:  movq    %rax, %r8
18000abc4:  callq   0x180035980
18000abc9:  movl    %ebx, %ecx
18000abcb:  callq   0x180065ae0
```

The key architecture / capability check is later:

```asm
18000ac4d:  movl    $0x31786667, %eax
18000ac52:  xorl    0x468(%rbp), %eax
18000ac58:  movzbl  0x46c(%rbp), %ecx
18000ac5f:  xorl    $0x31, %ecx
18000ac62:  movabsq $-0x1000000000001, %rdi
18000ac6c:  andq    0x468(%rbp), %rdi
18000ac73:  orl     %eax, %ecx
18000ac75:  jne     0x18000ac87
18000ac77:  movq    %r15, %rcx
18000ac7a:  callq   0x180064e20
18000ac7f:  cmpq    $0x7, %rax
18000ac83:  sete    %sil

18000ac87:  movabsq $0x30303231786667, %rax
18000ac91:  cmpq    %rax, %rdi
18000ac94:  sete    %bl
18000ac97:  testl   %r14d, %r14d
18000ac9a:  sete    %al
18000ac9d:  orb     %sil, %bl
18000aca0:  movb    $0x1, %al
18000aca2:  movb    %al, 0x90111(%rip) ; 0x18009adb9
18000aca8:  je      0x18000acb6
```

The constant `0x31786667` decodes to the ASCII bytes `gfx1` in little-endian, and `0x30303231786667` decodes to `gfx1200`.

This proves the code is checking the architecture string in the HIP property block, and the comparison is not a direct `gfx1032` blacklist. It is a generic `gfxNNNN`-style validation plus a special-case check for `gfx1200`.

### 5) `0x18000D0E0` through `0x18000D300`

This is the final unsupported-GPU check and formatter.

```asm
18000d205:  testb   $0x1, 0x8dbad(%rip) ; 0x18009adb9
18000d20c:  jne     0x18000d22f
18000d20e:  cmpq    $0x10, 0x8dbc2(%rip) ; 0x18009add8
18000d216:  jb      0x18000d2d0
18000d21c:  movq    0x8db9d(%rip), %r9 ; 0x18009adc0
18000d223:  leaq    0x73872(%rip), %r8 ; 0x180080a9c
18000d22a:  jmp     0x18000d36b
```

This is the exact path that emits the message:

- `0x18009adb9` is the final gate byte.
- `testb $0x1, ...` and `jne` means “if the flag is set, go to the unsupported message path”.
- `%r8` is loaded with `0x180080a9c`, which is the pointer to the format string.
- that message is the one that says the build only supports RX 9000 / RX 7000.

The format string is exactly in `.rdata` at `VA = 0x180080A9C` and raw offset `0x7FE9C`.

## Exact rejection path

The rejection path is:

1. `0x18000ab8f` begins the device property query loop.
2. `0x18000abaa` calls `hipGetDevicePropertiesR0600`.
3. The code validates the returned architecture string in the block ending at `0x18000aca2`.
4. It stores a state byte at `0x18009adb9` during that validation block.
5. The later block at `0x18000d205` checks that byte.
6. If the byte is set (`testb ... ; jne`), it loads the format string at `0x180080a9c` and emits the unsupported-GPU message.

In other words, the unsupported path is gated by the global flag `0x18009adb9`, not by the raw fat-binary registration or by a direct `gfx1032` compare.

## Architecture comparisons and capability logic

The disassembly proves that the code is doing several kinds of checks:

- prefix check for `gfx1` via `0x31786667` / `gfx1`
- string-length test via `strlen(...) == 7`
- special-case comparison against `gfx1200` via `0x30303231786667`
- additional runtime status / capability checks before the final reject flag is set.

This is not a direct `if (arch == gfx1032) reject` pattern.

The architecture list string in `.rdata` is:

```text
gfx1201,gfx1200,gfx1100,gfx1101,gfx1102,gfx11-generic,gfx1032,gfx9-generic
```

This makes it clear that `gfx1032` is known to the DLL. The reject path is therefore later than the architecture string table and is not caused by a simple allow-list exclusion.

## Global-variable data flow for `0x18009adb9`

The reads/writes of `0x18009adb9` from the disassembly are:

- Write: `0x18000aca2`
  - `movb %al, 0x90111(%rip) ; 0x18009adb9`
- Clear: `0x18000ad4c`
  - `movb $0x0, 0x90066(%rip) ; 0x18009adb9`
- Test: `0x18000ad53`
  - `cmpb $0x0, 0x9005f(%rip) ; 0x18009adb9`
- Final gate: `0x18000d205`
  - `testb $0x1, 0x8dbad(%rip) ; 0x18009adb9`
  - `jne 0x18000d22f`

This makes the byte a true global rejection flag. It is set inside the GPU validation block and later queried before the user-facing unsupported message is emitted.

## Confidence level

High.

The evidence is consistent across:

- the PE section headers,
- the imported HIP APIs,
- the embedded `.hip_fat` content,
- the direct runtime calls to `hipGetDevicePropertiesR0600`,
- the final unsupported-string branch at `0x18000d205`.

The remaining uncertainty is not whether the message is emitted via the flag, but the exact sub-condition within the property validation that causes that flag to be set for this RX 6600 in this particular build. The code does not show a direct `gfx1032` reject branch; instead it shows a later runtime capability/validation gate.

## Unanswered questions

- Which exact sub-field in the HIP device properties structure is causing the failure for the RX 6600 after `gfx1032` has already passed the generic string checks?
- Is the failure due to a vendor-family capability bit, a compute capability/subclass mismatch, or a later setup failure in a different helper?
- Why does the binary carry a hard-coded RX 9000 / RX 7000 message when its own architecture string list includes `gfx1032` and the embedded GFX1032 object is present?

## Recommended next analysis step

The next precise pass should be a targeted disassembly of the exact device-property block around `0x18000ab8f` to `0x18000acb6`, followed by a dump of the fields in the HIP property struct returned by `hipGetDevicePropertiesR0600` for a real RX 6600. The goal is to correlate the exact returned values in the `hipDeviceProp_t` structure against the branch criterion that sets `0x18009adb9`.

This will answer whether the final rejection is based on:

- a direct string allow-list;
- a family/compute-capability bitmask;
- a device property mismatch;
- or a later initialization failure after the architecture is accepted.

## Bottom line

The DLL does not reject `gfx1032` as a missing or unsupported architecture in the actual embedded HIP fat binary. The rejection is a later runtime gate controlled by the global byte `0x18009adb9`, which is set in the validation block ending at `0x18000aca2` and checked at `0x18000d205` before the code loads the unsupported-GPU text at `0x180080a9c`.

## Concise summary for the user

The exact gating branch appears to be the `testb $0x1, 0x18009adb9; jne 0x18000d22f` check at `0x18000d205`, which leads to the `0x180080a9c` unsupported-GPU message. That flag is written earlier in the block ending at `0x18000aca2` after the code validates the returned HIP device properties. The disassembly does not show a direct `gfx1032` blacklist; instead it shows a later runtime capability/validation gate, and the architecture list itself still includes `gfx1032`.
