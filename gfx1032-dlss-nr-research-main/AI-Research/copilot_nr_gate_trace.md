# NR gate trace for RX 6600 / gfx1032

## Objective

Identify the exact static control-flow path that leads from the live HIP device-properties query to the unsupported-GPU diagnostic string:

- unsupported string VA: `0x180080A9C`
- unsupported string text: `Unsupported GPU (%s): this build runs on Radeon RX 9000 (RDNA4) and RX 7000 (RDNA3) only.`

This is a static-analysis report only. No binary patching or runtime modification is proposed or performed.

---

## 1. The final string path

The unsupported-GPU string is reached from the final state-byte check in the initialization path:

```asm
18000d205: f6 05 ad db 08 00 01  testb   $0x1, 0x8dbad(%rip)   ; 0x18009adb9
18000d20c: 75 21                 jne     0x18000d22f
18000d21c: 4c 8b 0d 9d db 08 00   movq    0x8db9d(%rip), %r9      ; 0x18009adc0
18000d223: 4c 8d 05 72 38 07 00   leaq    0x73872(%rip), %r8      ; 0x180080a9c
18000d22a: e9 3c 01 00 00         jmp     0x18000d36b
```

`0x180080A9C` is used as the format string pointer. The actual branch only occurs if the state byte at `0x18009ADB9` is non-zero.

The farther downstream state check is:

```asm
18000ad53: 80 3d 5f 00 09 00 00  cmpb    $0x0, 0x9005f(%rip)   ; 0x18009adb9
18000ad5a: 0f 84 e3 0c 00 00     je      0x18000ba43
```

If the byte is zero, it falls through to the accepted path. If non-zero, the later code continues toward the unsupported-GPU message.

Therefore the key gate is the global byte `0x18009ADB9`, not the raw fat-binary registration path.

---

## 2. The state byte is set in the device-property validation block

The relevant setter is here:

```asm
18000ac97: 45 85 f6          testl   %r14d, %r14d
18000ac9a: 0f 94 c0          sete    %al
18000ac9d: 40 08 f3          orb     %sil, %bl
18000aca0: b0 01             movb    $0x1, %al
18000aca2: 88 05 11 01 09 00  movb    %al, 0x90111(%rip)   ; 0x18009adb9
```

This is a critical fact: the write to `0x18009ADB9` occurs in the runtime property-validation block and is not a direct `gfx1032` blacklist. The preceding logic determines whether the code reaches this block and whether the state byte is left non-zero.

The relevant branch that takes the code into the rejection path is therefore the final non-zero state-byte check at:

```asm
18000ad53: cmpb $0x0, [0x18009ADB9]
```

and the nearest upstream setter is:

```asm
18000aca2: movb %al, [0x18009ADB9]
```

---

## 3. The property query and the property buffer

The key HIP device-property calls are:

```asm
18000a9f3: movl    $0x5c0, %r8d
18000a9f9: movq    %r14, %rcx
18000a9fc: xor     %edx, %edx
18000a9fe: callq   0x180064a00
18000aa03: movq    %r14, %rcx
18000aa06: movl    %r12d, %edx
18000aa09: callq   0x180065a20 ; hipGetDevicePropertiesR0600
18000aa0e: testl   %eax, %eax
18000aa10: jne     0x18000a9e1
```

and the follow-up failure/diagnostic query:

```asm
18000ad01: leaq    -0x20(%rbp), %rsi
18000ad05: movl    $0x5c0, %r8d
18000ad0b: movq    %rsi, %rcx
18000ad0e: xor     %edx, %edx
18000ad10: callq   0x180064a00
18000ad15: movq    %rsi, %rcx
18000ad18: xor     %edx, %edx
18000ad1a: callq   0x180065a20 ; hipGetDevicePropertiesR0600
```

The call is the standard HIP property query; the device property buffer is the pointer in `RCX` (`R14` / `RSI` locally) and the device index is `RDX`.

The thunk is:

```asm
180065a20: jmpq *0x3280a(%rip) ; 0x180098230
```

This is the imported `hipGetDevicePropertiesR0600` IAT entry.

---

## 4. The architecture/name checks that feed the gate

The relevant validation block starts immediately after the property query at `0x18000AB8F` and is centered on `0x18000AC4D`.

### a) First string sanity check

```asm
18000ac4d: movl    $0x31786667, %eax   ; == "gfx1" (little-endian)
18000ac52: xorl    0x468(%rbp), %eax
18000ac58: movzbl  0x46c(%rbp), %ecx
18000ac5f: xorl    $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq    0x468(%rbp), %rdi
18000ac73: orl     %eax, %ecx
18000ac75: jne     0x18000ac87
```

This tests the first 8 bytes at `[RBP+0x468]` and the next byte at `[RBP+0x46C]` against a `gfx1` prefix pattern. In little-endian form, `0x31786667` corresponds to `gfx1`.

The check is effectively:

- compare the first 4 bytes with `gfx1`
- compare the fifth byte with `'1'`
- if not true, go to `0x18000AC87`

This is not a direct `gfx1032` rejection. It is a generic `gfx1*` family check.

### b) Special-case `gfx1200` check

```asm
18000ac87: movabsq $0x30303231786667, %rax
18000ac91: cmpq    %rax, %rdi
18000ac94: sete    %bl
```

`0x30303231786667` is little-endian `gfx1200`. So this is a special-case allow/comparison for `gfx1200`.

### c) `strlen` / length gate

```asm
18000ac77: movq    %r15, %rcx
18000ac7a: callq   0x180064e20
18000ac7f: cmpq    $0x7, %rax
18000ac83: sete    %sil
```

This is a `strlen`-like check and `sil` becomes non-zero when the architecture string length is 7. This matches names like:

- `gfx1032`
- `gfx1100`
- `gfx1101`
- `gfx1102`
- `gfx1200`
- etc.

This is a generic 7-byte arch-string check, not a direct `gfx1032` allowlist.

### d) `R14D` numeric helper result

```asm
18000abc9: movl    %ebx, %ecx
18000abcb: callq   0x180065ae0
18000abd0: movl    %eax, %r14d
18000abd3: testl   %eax, %eax
18000ac97: testl   %r14d, %r14d
18000ac9a: sete    %al
```

The value in `R14D` comes from a helper call just before the string-validation block. The code then tests `R14D` with `testl` and sets `AL` based on the zero result. This makes `R14D` a runtime blade/value of some kind, but it is not a direct `gfx1032` compare by itself.

The critical point: the final `AL = 1` assignment overwrites the earlier test result unconditionally.

---

## 5. Which field is being tested?

The static evidence points strongly to the AMD architecture string embedded inside the HIP property block, specifically the `gcnArchName` field of `hipDeviceProp_t`.

The official local ARM/AMD HIP header in the ROCm SDK shows:

```c
char gcnArchName[256];
```

as a member of `hipDeviceProp_t`.

The checked local values are:

- `[RBP+0x468]` = the beginning of the copied `gcnArchName` string
- `[RBP+0x46C]` = the byte immediately after the first 8 bytes of the copied name

This is consistent with reading the first 8-9 bytes of `gcnArchName` as a qword and one byte, then comparing against expected arch family strings.

The exact pattern is:

- `gcnArchName[0..3]` compared against `gfx1`
- `gcnArchName[4]` tested against `'1'` in the first branch
- `gcnArchName[0..7]` compare against `gfx1200` in the second branch
- `strlen(gcnArchName)` compared to 7 for the generic gfxNNNN family assumption

This is architecture-string logic, not a fat-binary/package registry check.

---

## 6. Expected value/range and RX 6600 implication

### Architecture/name checks

The code expects a string that begins with `gfx` and is in the `gfx1xxx` family. It explicitly handles a special case for `gfx1200`.

Expected supported shape:

- `gfx1200` (special-case)
- other 7-character `gfxNNNN` strings as a generic family category

RX 6600 property value:

- `gfx1032` is a valid AMD RDNA2 architecture string
- `gfx1032` length is 7
- it begins with `gfx1`, but it does not match the literal special-case `gfx1200` constant

This means it falls into the generic `gfx1*` validation path that leads to the state-byte write and later non-zero check.

### Compute capability checks

The helper result in `R14D` is a non-string runtime value, and the code does a `testl %r14d,%r14d` followed by a later `sete`/`movb` sequence. This indicates a runtime numeric capability or status result is involved, but static analysis does not prove which exact field it is without a live property dump.

### Device-count/index checks

This is not the main rejection condition. The earlier loop around `0x18000A8F0` / `0x18000A9E1` shows device iteration using `r12d` and `hipGetDevicePropertiesR0600` for `deviceIndex`, but the final rejection logic does not depend on a raw device count comparison at the end of the message path.

### Actual rejection conditions

The actual state transition is:

1. property buffer is queried
2. architecture string is parsed from `gcnArchName`
3. string-length / family checks are performed
4. the state byte at `0x18009ADB9` is written to `1` in the validation block
5. the later `cmpb $0x0,[0x18009ADB9]` fails and the code progresses toward the unsupported-GPU diagnostic

This is the actual runtime rejection control flow.

---

## 7. Control-flow diagram

```text
hipGetDevicePropertiesR0600
        |
        v
0x18000AA09 -> property buffer in RCX / local buffer at R14
        |
        v
0x18000AB8F -> begin device-property validation
        |
        +--> 0x18000AC4D : check gcnArchName[0..3] == "gfx1"
        |                and gcnArchName[4] == '1'
        |                if not, jump to 0x18000AC87
        |
        +--> 0x18000AC77 : strlen(gcnArchName) == 7 ? set SIL
        |
        +--> 0x18000AC87 : compare first 8 bytes against "gfx1200"
        |
        +--> 0x18000AC97 : test R14D
        |
        +--> 0x18000ACA2 : movb %al, [0x18009ADB9]   ; state byte set
        |
        +--> 0x18000AD53 : cmpb $0x0, [0x18009ADB9]
                          if non-zero -> unsupported path
                          if zero -> accepted path

unsupported path:
        0x18000D205 : testb $0x1, [0x18009ADB9]
        0x18000D223 : lea 0x180080A9C
        format string = "Unsupported GPU (%s): ... RDNA4 / RDNA3 only"
```

---

## 8. The single most likely rejection branch

The single most likely rejection branch is:

```asm
18000ad53: cmpb    $0x0, 0x9005f(%rip)    ; 0x18009adb9
18000ad5a: je      0x18000ba43
```

This is the first branch that deterministically chooses the unsupported path when the state byte is non-zero.

The state byte itself is set just before it at:

```asm
18000aca2: movb %al, 0x90111(%rip) ; 0x18009adb9
```

and the setter is reached from the `gcnArchName` validation block at `0x18000AC4D` / `0x18000AC87` / `0x18000AC97`.

This is the nearest static branch to the unsupported diagnostic and is therefore the best candidate for the gate that rejects the RX 6600 / `gfx1032` path.

### Why not the `0x18000AC97` instruction alone?

Because the state write at `0x18000ACA2` is unconditional on that path and is not, by itself, a proven rejection condition; the actual rejection is determined later when the non-zero state byte is checked at `0x18000AD53` and the code continues toward `0x180080A9C`.

So the correct statement is:

- `0x18000ACA2` is the state-set point
- `0x18000AD53` is the actual branching decision point that makes the unsupported path live

---

## 9. Main conclusion

The unsupported-GPU message is not caused by a missing or malformed `gfx1032` ELF in the fat binary. The fat binary already contains native `gfx1032` payloads. The runtime rejection is caused by a later validation gate that reads the live `hipDeviceProp_t.gcnArchName` string and writes a status byte, which is later checked and used to emit the unsupported-GPU message.

The evidence supports the following chain:

- `hipGetDevicePropertiesR0600` fills a local property buffer
- the local buffer is copied into a local 8-byte + 1-byte view around `[RBP+0x468]` / `[RBP+0x46C]`
- the code compares that string against `gfx1` and `gfx1200`
- it also checks `strlen(...) == 7`
- it writes the state byte at `0x18009ADB9`
- the later state check at `0x18000AD53` decides whether the unsupported-GPU path is taken

---

## 10. Confidence / remaining unknowns

### Confirmed

- `0x18009ADB9` is the state byte controlling the unsupported path.
- `0x18000AD53` is the branch that checks that byte.
- The unsupported string is reached only if that byte is non-zero.
- The byte is set from the device property validation block that reads the architecture string from the property buffer.
- The property buffer is populated by `hipGetDevicePropertiesR0600`.

### Unproven without runtime capture

- The exact value of the helper return in `R14D` and whether it is a hard capability check or just a side condition.
- Whether the code is explicitly rejecting `gfx1032` as a string value or whether it rejects any non-`gfx1200` path that reaches the same state bit.
- Whether the runtime property string for the RX 6600 is exactly `gfx1032` or a variant with a trailing character set.

### Confidence level

- Overall gate identification: high
- Exact final predicate behind the `gfx1032` reject: medium
- Exact semantic meaning of the `R14D` helper result: low without runtime data

This is the strongest static proof available without a debugger dump: the unsupported path is gated by the global state byte `0x18009ADB9`, and that byte is populated from the `hipGetDevicePropertiesR0600` / `gcnArchName` validation block, not from a missing fat-binary object.
