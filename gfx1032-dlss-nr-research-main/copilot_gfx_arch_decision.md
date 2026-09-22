# gfx arch decision trace for the RX 6600 / gfx1032 gate

## Scope

This report is limited to the static control-flow from:

- `0x18000AC20`
- to `0x18000ACA2`

Goal: determine exactly what condition makes `AL` become the value later written to:

- `0x18009ADB9`

No patching is proposed, and no binary modification is performed.

---

## 1. Exact disassembly of the decision block

Disassembly from the actual binary:

```asm
18000ac20: 09 48 89                     orl     %ecx, -0x77(%rax)
18000ac23: 44 24 40                     andb    $0x40, %al
18000ac26: 89 54 24 30                  movl    %edx, 0x30(%rsp)
18000ac2a: 89 4c 24 28                  movl    %ecx, 0x28(%rsp)
18000ac2e: 4c 89 7c 24 20               movq    %r15, 0x20(%rsp)
18000ac33: f2 0f 11 4c 24 38            movsd   %xmm1, 0x38(%rsp)
18000ac39: 48 8d 0d 82 fc 06 00         leaq    0x6fc82(%rip), %rcx     # 0x18007a8c2
18000ac40: 89 da                        movl    %ebx, %edx
18000ac42: 49 89 f8                     movq    %rdi, %r8
18000ac45: 4d 89 e1                     movq    %r12, %r9
18000ac48: e8 b3 cf ff ff               callq   0x180007c00

18000ac4d: b8 67 66 78 31               movl    $0x31786667, %eax       # imm = 0x31786667
18000ac52: 33 85 68 04 00 00            xorl    0x468(%rbp), %eax
18000ac58: 0f b6 8d 6c 04 00 00         movzbl  0x46c(%rbp), %ecx
18000ac5f: 83 f1 31                     xorl    $0x31, %ecx
18000ac62: 48 bf ff ff ff ff ff ff fe ff        movabsq $-0x1000000000001, %rdi
18000ac6c: 48 23 bd 68 04 00 00         andq    0x468(%rbp), %rdi
18000ac73: 09 c1                        orl     %eax, %ecx
18000ac75: 75 10                        jne     0x18000ac87

18000ac77: 4c 89 f9                     movq    %r15, %rcx
18000ac7a: e8 a1 a1 05 00               callq   0x180064e20
18000ac7f: 48 83 f8 07                  cmpq    $0x7, %rax
18000ac83: 40 0f 94 c6                  sete    %sil

18000ac87: 48 b8 67 66 78 31 32 30 30 00        movabsq $0x30303231786667, %rax   ; "gfx1200"
18000ac91: 48 39 c7                     cmpq    %rax, %rdi
18000ac94: 0f 94 c3                     sete    %bl

18000ac97: 45 85 f6                     testl   %r14d, %r14d
18000ac9a: 0f 94 c0                     sete    %al
18000ac9d: 40 08 f3                     orb     %sil, %bl
18000aca0: b0 01                        movb    $0x1, %al
18000aca2: 88 05 11 01 09 00            movb    %al, 0x90111(%rip)      ; 0x18009adb9
18000aca8: 74 0c                        je      0x18000acb6
```

---

## 2. Control-flow reconstruction of the block

The block is a decision tree with three main stages.

### Stage A: detect whether the architecture string starts with `gfx1`

```asm
18000ac4d: movl $0x31786667, %eax
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87
```

This is effectively checking:

- the qword at `[RBP+0x468]` equals a value whose bytes are `gfx1` in little-endian form
- and the byte at `[RBP+0x46C]` is `'1'`

The constant `0x31786667` is the little-endian encoding of the ASCII string bytes:

- `67 66 78 31` = `gfx1`

So this is validating the prefix `gfx1` at the beginning of the architecture string.

If the value is not a `gfx1` prefix, the block jumps to `0x18000AC87` without the strlen check.

### Stage B: generic length check for gfxNNNN strings

```asm
18000ac77: movq %r15, %rcx
18000ac7a: callq 0x180064e20
18000ac7f: cmpq $0x7, %rax
18000ac83: sete %sil
```

`0x180064E20` is a `strlen`-like helper.

`SIL` becomes 1 only if the string length is exactly 7.

This matches architecture strings like:

- `gfx1032`
- `gfx1100`
- `gfx1101`
- `gfx1102`
- `gfx1200`
- `gfx900` (length 7)

This is a generic family check, not a `gfx1032`-specific check.

### Stage C: special-case `gfx1200` and the final state-byte write

```asm
18000ac87: movabsq $0x30303231786667, %rax   ; bytes = 67 66 78 31 32 30 30 = "gfx1200"
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl

18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
18000ac9d: orb %sil, %bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, [0x18009ADB9]
```

This is the key point: `AL` is forced to `1` immediately before the store.

So the final write is:

```asm
movb $0x1, %al
movb %al, [0x18009ADB9]
```

That means the state byte is set to `1` whenever the block reaches this point.

The earlier `testl %r14d,%r14d` / `sete %al` only computes a temporary value for `AL`, but it is then overwritten by the unconditional `movb $0x1, %al`.

Therefore:

- the final state byte is not determined by the prior `R14D` test
- the prior `R14D` test only affects a temporary result that is discarded
- the actual store to `[0x18009ADB9]` is unconditional once execution reaches `0x18000ACA2`

---

## 3. Every instruction affecting AL / BL / SIL / R14D / R14 / flags

### AL

Contributors to `AL` in this block:

```asm
18000ac9a: 0f 94 c0     sete    %al
18000aca0: b0 01       movb    $0x1, %al
18000aca2: 88 05 11 01 09 00  movb    %al, [0x18009ADB9]
```

The `sete %al` is transient and overwritten by `movb $0x1, %al`.

This means `AL` is forced to `1` at `0x18000ACA0` regardless of the earlier `R14D` result.

### BL

```asm
18000ac94: 0f 94 c3    sete %bl
18000ac9d: 40 08 f3    orb %sil, %bl
```

`BL` is set to 1 only if the string equals `gfx1200`.

Then `BL` is ORed with `SIL`, so after `0x18000AC9D`:

- `BL = 1` if either `strcmp-like-`equal-to-`gfx1200` OR `strlen(...) == 7`

This is a generic flag used later, not the final state byte itself.

### SIL

```asm
18000ac83: 40 0f 94 c6  sete %sil
```

`SIL` is set to 1 if the string length is exactly 7.

This is the generic `gfxNNNN` family indicator.

### R14D

The block at `0x18000AC97` uses `R14D` directly:

```asm
18000ac97: 45 85 f6    testl %r14d, %r14d
18000ac9a: 0f 94 c0    sete %al
```

This is an integer status test. It does not inspect the architecture string directly.

### R14

`R14` is loaded just before this block when `hipGetDevicePropertiesR0600` is called:

```asm
18000ab91: leaq -0x20(%rbp), %r12
18000ab95: movl $0x5c0, %r8d
18000ab9b: movq %r12, %rcx
18000ab9e: xor %edx, %edx
18000aba0: call 0x180064a00
18000aba5: movq %r12, %rcx
18000aba8: movl %ebx, %edx
18000abaa: call 0x180065a20  ; hipGetDevicePropertiesR0600
```

The local property buffer is on the stack at `-0x20(%rbp)`, and `R14` was previously set to that buffer pointer. The code then uses `R15` to point to the `gcnArchName` region and `R14` as a scratch/general register in the same function.

The exact `R14D` origin is: the return value from the immediately previous helper call at:

```asm
18000abcb: callq 0x180065ae0
18000abd0: movl %eax, %r14d
```

This is the only direct way `R14D` is populated in this region.

### Flags consumed by conditional branches

Relevant flag-producing instructions:

```asm
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87

18000ac7f: cmpq $0x7, %rax
18000ac83: sete %sil

18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl

18000ac97: testl %r14d, %r14d
18000ac9a: sete %al

18000aca8: je 0x18000acb6
```

The `JNE` at `0x18000AC75` and the `JE` at `0x18000ACA8` are the only branch conditions in this block. The final `JE` is based on the `ZF` from the `testl %r14d,%r14d` result, but it is reached after `AL` has already been overwritten to `1`.

---

## 4. Exactly where `R14D` comes from

The immediate source is:

```asm
18000abcb: e8 10 af 05 00  callq 0x180065ae0
18000abd0: 41 89 c6        movl %eax, %r14d
```

So the flow is:

```text
device index -> call 0x180065a20 (hipGetDevicePropertiesR0600)
    -> function returns status/metadata
    -> call 0x180065ae0
    -> EAX returned from helper
    -> MOV EAX, R14D
```

This is the only assignment to `R14D` in the local path.

### Helper target at `0x180065AE0`

The target is an indirect thunk:

```asm
180065ae0: ff 25 aa 27 03 00  jmpq *0x327aa(%rip)  ; 0x180098290
```

That means `0x180065AE0` is not a local implementation. It is a jump-indirect import thunk to the IAT slot pointed at by `0x180098290`.

This means, from static disassembly alone:

- we can prove the return value is not computed inside this function body
- we can prove `R14D` is an imported helper result
- we cannot prove the exact semantic of that helper without a runtime export or debugger dump

The call semantics are therefore limited to: `R14D = EAX` after that helper returns.

This is important: the helper is not a `gfx1032`-detector; it is a generic helper returning some integer status/result used in the local branch chain.

---

## 5. The meaning of 0x18000AC97 / 0x18000AC9D / 0x18000ACA0 / 0x18000ACA2

### 5.1 `0x18000AC97: test r14d,r14d`

```asm
18000ac97: 45 85 f6     testl %r14d, %r14d
```

This sets `ZF = 1` if `R14D == 0`.

Then:

```asm
18000ac9a: 0f 94 c0     sete %al
```

So:

- if `R14D == 0`, then `AL = 1`
- otherwise `AL = 0`

This is only a temporary value.

### 5.2 `0x18000AC9D: or bl,sil`

```asm
18000ac9d: 40 08 f3    orb %sil, %bl
```

This combines the two architecture-related flags:

- `BL` = 1 if the string equals `gfx1200`
- `SIL` = 1 if `strlen(...) == 7`

After this instruction:

- `BL = BL OR SIL`

This is not a direct `gfx1032` compare. It is simply a merged boolean for the later state logic.

### 5.3 `0x18000ACA0: mov al,1`

```asm
18000aca0: b0 01  movb $0x1, %al
```

This unconditionally overwrites the earlier `sete %al` result.

This is the most important fact in the whole block:

- the previous `R14D` test does not determine the value written to the state byte
- the previous `AL` value is discarded

### 5.4 `0x18000ACA2: mov [0x18009ADB9],al`

```asm
18000aca2: 88 05 11 01 09 00  movb %al, [0x18009ADB9]
```

This is just the store of the final `AL` value to the global state byte.

Because of the previous instruction, the stored value is always `1` whenever execution reaches this point.

This means `0x18000ACA2` is not, by itself, a rejection condition. The earlier control-flow decides whether the block is reached, and then this store happens once the block is reached.

---

## 6. What condition actually makes `AL` become the value written?

The real answer is:

- the condition is not a `gfx1032` check
- the value is forced to `1` by the unconditional instruction:

```asm
18000aca0: b0 01  movb $0x1, %al
```

The block reaches this point only if the earlier architecture-family check succeeds or falls through as expected.

The earlier conditions are:

```asm
18000ac75: jne 0x18000ac87
18000ac83: sete %sil
18000ac94: sete %bl
18000ac97: testl %r14d, %r14d
```

but they do not decide the final stored value. They only decide whether execution continues down the branch and whether the temporary `AL` or merged flags are used in later logic.

The decisive fact is that the final state byte store is unconditional once the block is entered.

---

## 7. What is actually being checked in this block?

The block is not performing a direct `if (arch == gfx1032)` compare.

The exact checks are:

1. Prefix check against `gfx1`
2. `strlen-like` check for length 7
3. special-case compare against `gfx1200`
4. integer status check of an imported helper return in `R14D`

This means the block is doing a generic architecture-family validation plus a helper return check, not a hardcoded reject for `gfx1032`.

---

## 8. Does gfx1032 get rejected here?

### Proven static answer

There is no explicit `gfx1032` compare in this block.

There is:

- `gfx1` prefix check at `0x18000AC4D`
- `strlen == 7` at `0x18000AC77`
- `gfx1200` compare at `0x18000AC87`

There is no `gfx1032` constant compare and no `gfx1100` / `gfx1102` / `gfx900` direct compare.

Therefore the static evidence strongly indicates:

- `gfx1032` is not directly rejected by an explicit architecture compare
- `gfx1032` is accepted by the architecture test in the same way as other 7-character `gfxNNNN` strings
- the later helper result in `R14D` and the state-byte gate at `0x18000AD53` are the real controlling path

This is most consistent with:

- Option B: accepted by the architecture test but rejected by a later capability/status test

but static analysis does not prove the exact helper semantic, so the true final cause remains unproven without runtime values.

It is not proven to be direct A (`gfx1032` is deliberately rejected by a specific condition), because no such compare exists in this block.

---

## 9. Hypothetical architecture strings through this block

### 9.1 gfx1200

- first check: prefix `gfx1` passes
- `strlen == 7`: true
- `cmpq %rax, %rdi` where `%rax == "gfx1200"` => `BL = 1`
- `R14D` test may set `AL` temporarily
- then `movb $0x1, %al` forces final store to `0x18009ADB9 = 1`
- final branch `je 0x18000acb6` depends on `ZF` from `testl %r14d,%r14d`

Conclusion: `gfx1200` definitely enters this block and the state byte is set to 1 once the block is reached.

### 9.2 gfx1100

- prefix `gfx1` passes
- `strlen == 7`: true
- `BL = 0` because it is not `gfx1200`
- `R14D` test is still executed
- then `AL` overwritten to 1
- final store to `0x18009ADB9 = 1`

Conclusion: the block reaches the same final state store. The special-case `gfx1200` is not unique; the final state write is unconditional.

### 9.3 gfx1102

Same as `gfx1100`:

- prefix passes
- length 7 passes
- not `gfx1200`
- final state store still becomes 1

### 9.4 gfx1032

- prefix passes
- length 7 passes
- not `gfx1200`
- final state store still becomes 1

### 9.5 gfx900

- prefix `gfx9` does not pass the `gfx1` prefix check, so it jumps to `0x18000AC87` without the `strlen == 7` path
- later compare against `gfx1200` fails
- `AL` may still be set in a different later logic path, but this block is specifically for `gfx1` namespace strings

Conclusion: `gfx900` is not a valid entry to the `gfx1` path; it is outside the exact check domain.

---

## 10. Truth table

| Architecture string | Prefix `gfx1`? | `strlen==7`? | `== "gfx1200"`? | Final state byte at `0x18009ADB9` | Notes |
|---|---:|---:|---:|---:|---|
| `gfx1200` | yes | yes | yes | 1 | special-case path, but still final store is 1 once block reached |
| `gfx1100` | yes | yes | no | 1 | generic `gfx1` family |
| `gfx1102` | yes | yes | no | 1 | generic `gfx1` family |
| `gfx1032` | yes | yes | no | 1 | generic `gfx1` family |
| `gfx900` | no | n/a | no | not proven here | outside this block’s `gfx1` test |

This table is static-analysis based; it is not a runtime proof of the actual RX 6600 values.

---

## 11. Earliest conditional instruction that specifically distinguishes gfx1032 from a supported RDNA3/RDNA4 architecture

There is no such instruction in this block.

The earliest relevant conditions are:

```asm
18000ac75: jne 0x18000ac87
18000ac83: sete %sil
18000ac94: sete %bl
```

and the special-case `gfx1200` compare:

```asm
18000ac87: cmpq %rax, %rdi
18000ac94: sete %bl
```

This is the earliest code that distinguishes a `gfx1200` string from other `gfx1` family strings, not `gfx1032` from RDNA3/RDNA4.

There is no explicit compare against `gfx1032`, `gfx1100`, `gfx1101`, `gfx1102`, or any supported RDNA3/RDNA4 naming pattern in this block.

Therefore the earliest proven condition that actually decides the later unsupported path is not a `gfx1032` check; it is the state-byte check later at:

```asm
18000ad53: cmpb $0x0, [0x18009ADB9]
```

and the state byte is written earlier at:

```asm
18000aca2: movb %al, [0x18009ADB9]
```

---

## 12. Exact instruction bytes of the relevant critical instructions

### `0x18000AC97`

```asm
45 85 f6
```

Instruction:

```asm
testl %r14d, %r14d
```

### `0x18000AC9D`

```asm
40 08 f3
```

Instruction:

```asm
orb %sil, %bl
```

### `0x18000ACA0`

```asm
b0 01
```

Instruction:

```asm
movb $0x1, %al
```

### `0x18000ACA2`

```asm
88 05 11 01 09 00
```

Instruction:

```asm
movb %al, [0x18009ADB9]
```

---

## 13. Final determination

### Earliest proven rejection condition

The earliest branch that actually determines whether the unsupported path is taken is:

```asm
0x18000AD53: 80 3d 5f 00 09 00 00  cmpb $0x0, 0x9005f(%rip) ; [0x18009ADB9]
0x18000AD5A: 0f 84 e3 0c 00 00     je 0x18000BA43
```

This is the earliest conditional in the actual rejection chain that decides accepted-vs-unsupported after the state byte is set.

### Address

- `0x18000AD53`

### Original bytes

```text
80 3d 5f 00 09 00 00
```

### Operands / fields involved

- `cmpb $0x0, [0x18009ADB9]`
- the state byte `0x18009ADB9`
- not a direct `gfx1032` compare; it is a state-byte gate that is reached after the earlier architecture validation block

### gfx1032 result

- static proof: `gfx1032` is not directly rejected by a dedicated compare in this block
- likely path: enters `gfx1` validation, length 7 path, final state byte set to 1
- therefore it reaches the same later rejection gate as the generic `gfx1` path

### gfx1100 result

- same as `gfx1032`: enters path and reaches final store to `0x18009ADB9 = 1`

### gfx1200 result

- special-case compares true, but final state byte still becomes 1 once the block is reached

### Confidence

- high: there is no direct `gfx1032` reject compare in this block
- high: the later state-byte gate at `0x18000AD53` is the real accepted-vs-unsupported decision point
- medium: the exact helper return in `R14D` and its runtime semantics remain unproven without live property data

### Remaining unknowns

- what imported helper returns into `R14D`
- whether the helper result is a real capability check or just a side-effect/status bit
- whether the final unsupported path is triggered for all `gfx1` values or only for a specific runtime combination of `gcnArchName` and the helper result
- whether the code intends to reject a different architecture family later in the function, not in this exact block

---

## Bottom line

The answer to the original question is:

- `AL` is forced to `1` at `0x18000ACA0`
- therefore the value written at `0x18009ADB9` is `1` whenever execution reaches `0x18000ACA2`
- the actual rejection is not caused by `gfx1032` being directly compared against a blacklist in this block
- the earlier conditional logic only determines whether the code reaches this state write
- the decisive runtime rejection is the later state-byte test at `0x18000AD53`

This is the most conservative static conclusion supported by the actual binary disassembly.
