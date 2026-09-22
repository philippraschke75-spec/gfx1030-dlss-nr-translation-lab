# R14D helper origin and semantics trace

## Scope

This report traces the immediate origin of `R14D` used here:

```asm
18000ac97: 45 85 f6        testl   %r14d, %r14d
18000ac9a: 0f 94 c0        sete    %al
18000aca0: b0 01           movb    $0x1, %al
18000aca2: 88 05 11 01 09 00 movb    %al, 0x90111(%rip) ; 0x18009adb9
```

The goal is to determine exactly what last defined `R14D`, where it came from, and whether that helper is a GPU-compatibility predicate or a generic utility used elsewhere.

This is static analysis only. No DLL modification is performed.

---

## 1. Immediate last definition of `R14D`

The exact last definition in the local region is:

```asm
18000abcb: e8 10 af 05 00  callq   0x180065ae0
18000abd0: 41 89 c6        movl    %eax, %r14d
```

So the immediate source of `R14D` is the return value from the indirect call at `0x180065AE0`, copied into `R14D` by:

```asm
movl %eax, %r14d
```

This is the last definition of the value used by:

```asm
18000ac97: testl %r14d, %r14d
```

Therefore, the function/method behind `0x180065AE0` is the final unresolved origin of the value that controls the temporary `AL` computation before the unconditional state write.

---

## 2. Exact helper target and why it remains indirect

The target is an import thunk:

```asm
180065ae0: jmpq *0x327aa(%rip) ; 0x180098290
```

This is not a local helper body. It is a thunk that jumps through a function pointer stored at `0x180098290`.

The information we can prove statically is:

- the call is indirect
- the target is a function pointer in data/IAT space
- the binary does not locally contain the helper body itself
- therefore the actual helper implementation is not in this DLL and cannot be reconstructed from this binary alone without a runtime import dump or debugger memory dump

This is important: we must not assume a name from its address alone. The local code does not provide a function body.

---

## 3. What the local code proves about the helper return value

Immediately after the helper call, the code does:

```asm
18000abd0: movl %eax, %r14d
18000abd3: testl %eax, %eax
18000abd5: leaq 0x75179(%rip), %rax
18000abdc: leaq 0x779ce(%rip), %rcx
18000abe3: cmoveq %rcx, %rax
```

This means the helper return result is treated as a numeric value and then used in a zero/nonzero branch decision. The code is not comparing a string; it is checking whether the returned integer is zero and then selecting a branch target pointer for subsequent logic.

This is critical: the helper is used as a generic numeric status/result value, not as an architecture-string compare or a direct `gfx1032` reject predicate.

---

## 4. The helper is not locally defined as a string or architecture predicate

The block immediately surrounding the use is:

```asm
18000ac4d: movl $0x31786667, %eax ; "gfx1"
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87
18000ac77: movq %r15, %rcx
18000ac7a: callq 0x180064e20
18000ac7f: cmpq $0x7, %rax
18000ac83: sete %sil
18000ac87: movabsq $0x30303231786667, %rax ; "gfx1200"
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
18000ac9d: orb %sil, %bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, [0x18009ADB9]
```

This tells us the string checks are local and explicit, but the helper result is just a numeric value used in a local zero-test. It is not a `gfx1032` compare and not a string parser. The actual state-byte write then overrides it with `1`.

Therefore, the helper return is not the definitive architecture compatibility check driving the final `0x18009ADB9` write. The final state write is forced to `1` by:

```asm
18000aca0: b0 01 movb $0x1, %al
```

---

## 5. What is the helper return used for elsewhere?

The same call target `0x180065AE0` appears in multiple places in the binary. The whole-binary search shows:

- `0x18000ABCB`
- `0x18000B9FE`
- `0x18000D02C`
- `0x1800135E9`
- `0x1800139B6`
- `0x180019DA0`
- `0x18001C82F`

This is a strong sign the helper is a generic utility used at several unrelated points, not a `gfx1032`-specific test.

Examples from the disassembly:

### 0x18000B9FE

```asm
18000b9fe: callq 0x180065ae0
18000ba03: movl 0x8f3ff(%rip), %eax
18000ba09: movl %eax, (%rsi)
```

The helper return is immediately stored to memory, not compared to a GPU architecture string.

### 0x1800135E9

```asm
1800135e9: callq 0x180065ae0
1800135ee: movl 0x87814(%rip), %eax
1800135f4: movl %eax, (%rdi)
```

Again, the return value is stored, not used as an architecture discriminator.

### 0x18000D02C

```asm
18000d02c: callq 0x180065ae0
18000d031: callq 0x180065a40
18000d036: movl %eax, %esi
```

The helper return is consumed in the same broader runtime/control-flow pattern, not as a GPU string test.

This strongly supports the conclusion that the helper is a generic runtime utility returning a numeric status/result, and the local `R14D` use is only one consumer among many.

---

## 6. What does the helper return for conceptual GPU strings?

This cannot be proved from the local disassembly because:

- the helper is indirect and not implemented in this DLL
- the local code never compares the helper result to specific architecture strings
- the helper is used in many unrelated contexts

Therefore, there is no statically provable mapping such as:

- `gfx1200` => 0
- `gfx1100` => 1
- `gfx1102` => 1
- `gfx1032` => 0
- `gfx900` => 0

The local code is not performing a switch on `gcnArchName` and then calling the helper. The actual `gfx1` string tests are local and explicit; the helper return is a separate numeric check.

So the correct static statement is:

- no direct per-architecture semantic is proven for the helper return
- no direct `gfx1032` outcome is proven from this helper alone
- the helper result is not enough to establish a GPU compatibility decision

---

## 7. Constants and strings in the relevant local block

The explicit architecture and string constants used locally are:

- `gfx1`: encoded literal `0x31786667`
- `gfx1200`: encoded literal `0x30303231786667`
- `strlen(... ) == 7` check on the string
- no direct local `gfx1032` compare in the block
- no local `gfx11` or `gfx9` compare in this specific block

These are the constants visible in the immediate path to `0x18000ACA2`.

The actual block checks the architecture string, but does not reject `gfx1032` by string literal. The code is broad enough to include `gfx1032`, `gfx1100`, `gfx1102`, etc. (all 7-character `gfxNNNN` names) as valid inputs to the generic string path.

---

## 8. Does the helper depend on `hipDeviceProp_t` or `gcnArchName`?

The local code proves that `gcnArchName` is being read and validated immediately before the `R14D` check, but the helper itself is not defined in this DLL. There is no local code that shows the helper takes a pointer to `hipDeviceProp_t` or reads `gcnArchName` directly.

The local data flow is:

- `hipGetDevicePropertiesR0600` populates a property buffer
- the code copies/reads string bytes from the buffer into local scratch variables
- separate checks are performed locally against the string bytes
- a helper call is made, returning a numeric value into `EAX` and then `R14D`

This helper return is not proven to be a `hipDeviceProp_t` field access or a GPU-architecture boolean. It is a separate runtime helper result, likely a generic integer status, and it is not the direct `gfx1032` determinant.

---

## 9. Why the helper is not the GPU compatibility gate

The strongest static evidence is:

1. The final state byte is explicitly overridden to `1` by `movb $0x1, %al`.
2. The helper return in `R14D` is only used as a temporary boolean before that unconditional overwrite.
3. The same helper is called from many unrelated places in the binary with the return value stored or passed along, not only in the architecture validation block.
4. The local architecture checks are explicit string tests for the `gfx1` family; there is no direct `gfx1032` compare.

Therefore the helper is not proven to be a GPU compatibility test or a `gfx1032` decision point. It is more consistent with a generic numeric helper or runtime status function used as a boolean flag in several unrelated code paths.

---

## R14D ORIGIN

- instruction: `movl %eax, %r14d`
- address: `0x18000ABD0`
- helper: indirect import thunk at `0x180065AE0` -> `jmpq *0x327aa(%rip)` -> `0x180098290`
- inputs: not proven locally; the helper is called from the same property-query flow, but the binary does not expose a local implementation or the exact argument list
- return semantics: a generic numeric return, used only as a zero/nonzero test here (`testl %r14d,%r14d`), not a direct `gfx1032` decision; the function is used at multiple unrelated call sites, which argues against a single architecture-specific purpose
- relevant constants: `gfx1` (`0x31786667`), `gfx1200` (`0x30303231786667`), one 7-character string-length test; no direct `gfx1032` compare in this block
- relationship to gfx1032: no direct relationship is proven; the helper is not the direct `gfx1032` compatibility gate
- confidence: medium on the call origin and usage pattern; low-to-medium on the exact imported helper semantic, because the implementation is external and indirect
