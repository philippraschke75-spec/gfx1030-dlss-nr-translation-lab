# Status-byte trace for `0x18009ADB9`

## Scope

This report traces every write to the global byte at:

- `0x18009ADB9`

and the precise control flow that leads to it. The goal is to determine what the code is actually testing and whether the later unsupported-GPU path is driven by a real `gfx1032` rejection or by a generic runtime state flag.

This is static analysis only. No DLL modification or patch is proposed.

---

## 1. Executive summary

The key fact is not that `0x18009ADB9` is itself a `gfx1032` decision. The decisive fact is that the code reaches the write at `0x18000ACA2` only after a generic `gfx1`-family validation block, and once it gets there, the state byte is assigned `1` by the unconditional sequence:

```asm
18000aca0: b0 01         movb    $0x1, %al
18000aca2: 88 05 11 01 09 00  movb    %al, 0x90111(%rip) ; 0x18009adb9
```

The state byte is later consumed at:

```asm
18000ad53: 80 3d 5f 00 09 00 00  cmpb $0x0, 0x9005f(%rip) ; 0x18009adb9
18000ad5a: 0f 84 e3 0c 00 00     je  0x18000ba43
```

and again later at:

```asm
18000d205: f6 05 ad db 08 00 01  testb $0x1, 0x8dbad(%rip) ; 0x18009adb9
18000d20c: 75 23                 jne  0x18000d22f
```

The state flag therefore acts as a runtime gate that is set on one validation path and cleared on another. It is not a direct `gfx1032` compare by itself.

---

## 2. Every write to `0x18009ADB9`

The complete disassembly cross-reference inventory shows exactly two writes to the same global byte in the relevant code path.

### Write 1: set nonzero state byte

- Address: `0x18000ACA2`
- Original instruction bytes: `88 05 11 01 09 00`
- Disassembly:

```asm
18000aca2: 88 05 11 01 09 00  movb    %al, 0x90111(%rip) ; 0x18009adb9
```

- Incoming value in `AL`:
  - `AL` was previously set from `sete %al` via:

```asm
18000ac97: 45 85 f6  testl %r14d, %r14d
18000ac9a: 0f 94 c0  sete  %al
```

  - but immediately after that it is overwritten by:

```asm
18000aca0: b0 01  movb $0x1, %al
```

  - therefore the actual value stored to the byte is always `1` whenever execution reaches this instruction.

- Immediate predecessors:
  - `0x18000AC9D`:

```asm
18000ac9d: 40 08 f3  orb %sil, %bl
```

  - `0x18000ACA0`:

```asm
18000aca0: b0 01    movb $0x1, %al
```

- Conditional branches that can reach the write:
  - The write is reached after the block at `0x18000AC4D` falls through to `0x18000ACA2`.
  - The relevant incoming branch conditions are:

```asm
18000ac75: 75 10   jne 0x18000ac87
18000ac87: ...
18000ac97: 45 85 f6  testl %r14d, %r14d
18000ac9a: 0f 94 c0  sete %al
18000aca8: 74 0c   je 0x18000acb6
```

  - The branch at `0x18000AC75` is taken if the initial `gfx1` prefix check fails. Otherwise the code continues into the `strlen`/generic-family logic.
  - The branch at `0x18000ACA8` checks the zero-result from `testl %r14d,%r14d`, but it is reached after AL has already been overwritten to `1`; the write itself is unaffected by that branch.

### Write 2: clear the state byte

- Address: `0x18000AD4C`
- Original instruction bytes: `c6 05 66 00 90 00 00`
- Disassembly:

```asm
18000ad4c: c6 05 66 00 90 00 00  movb    $0x0, 0x90066(%rip) ; 0x18009adb9
```

- Incoming value:
  - `0` (explicit zero write)

- Immediate predecessor instructions:
  - `0x18000AD47`:

```asm
18000ad47: e8 2a 00 00 00  callq 0x180007c00
```

  - the preceding block at `0x18000AD01`..`0x18000AD47` is a secondary runtime property / status path that eventually emits the explicit clear.

- Conditional branches that can reach the write:
  - The `js` at `0x18000AB89` appears to route into the `0x18000AD01` path when a negative device/index/status condition is detected:

```asm
18000ab87: 45 85 db  testl %ebx, %ebx
18000ab89: 78 76     js    0x18000ad01
```

  - This is the branch that leads to the explicit zero-write path. The state byte is reset before the later `cmpb $0x0,[0x18009ADB9]` test.

---

## 3. Full write inventory summary

| Write site | Value written | Instruction | Meaning |
|---|---:|---|---|
| `0x18000ACA2` | `1` | `movb %al, [0x18009ADB9]` | set nonzero state bit on validation path |
| `0x18000AD4C` | `0` | `movb $0x0, [0x18009ADB9]` | clear state bit on alternate/failure path |

This is the complete write set for the relevant global byte in the reachable code under the static disassembly inspected here.

---

## 4. All cross-references to `0x18009ADB9`

The binary shows these references:

1. Write:
   - `0x18000ACA2` → `movb %al, 0x90111(%rip) ; 0x18009adb9`
2. Write:
   - `0x18000AD4C` → `movb $0x0, 0x90066(%rip) ; 0x18009adb9`
3. Test:
   - `0x18000AD53` → `cmpb $0x0, 0x9005f(%rip) ; 0x18009adb9`
4. Late decision check:
   - `0x18000D205` → `testb $0x1, 0x8dbad(%rip) ; 0x18009adb9`

There are no other direct writes to that address in the inspected disassembly.

---

## 5. Mini control-flow graph around every write

### A. Set path (`0x18000ACA2`)

```text
0x18000AA09  call hipGetDevicePropertiesR0600
  |
  v
0x18000AB91  prepare property buffer / call helper
  |
  v
0x18000ABCB  call 0x180065ae0
  |
  v
0x18000ABD0  movl %eax, %r14d
  |
  v
0x18000AC4D  validate gcnArchName prefix "gfx1"
  |
  +--> if prefix fails: jump to 0x18000AC87 (still in same block)
  |
  v
0x18000AC77  strlen-like check (==7)
  |
  +--> sets SIL if length == 7
  |
  v
0x18000AC87  special case compare against "gfx1200"
  |
  +--> sets BL if equal
  |
  v
0x18000AC97  testl %r14d, %r14d
  |
  +--> sete %al
  |
  v
0x18000AC9D  orb %sil, %bl
  |
  v
0x18000ACA0  movb $0x1, %al
  |
  v
0x18000ACA2  movb %al, [0x18009ADB9]   ; write 1
  |
  v
0x18000ACA8  je 0x18000ACB6
```

Key observation: the write at `0x18000ACA2` occurs after the string checks and helper-result check, but the final value is forced to `1` regardless of the earlier boolean results.

### B. Clear path (`0x18000AD4C`)

```text
0x18000AB87  testl %ebx, %ebx
  |
  +--> js 0x18000AD01
  |
  v
0x18000AD01  secondary property query / status path
  |
  v
0x18000AD47  call 0x180007c00
  |
  v
0x18000AD4C  movb $0x0, [0x18009ADB9]
  |
  v
0x18000AD53  cmpb $0x0, [0x18009ADB9]
  |
  +--> if zero: jump to accepted path 0x18000BA43
  |
  +--> if nonzero: continue to later rejection path
```

This is the explicit clear path; it is the inverse of the set path.

---

## 6. The path `0x18000AC4D → ... → 0x18000ACA0 → 0x18000ACA2`

The important block is:

```asm
18000ac4d: movl $0x31786667, %eax
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
18000ac87: movabsq $0x30303231786667, %rax
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
18000ac9d: orb %sil, %bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, 0x90111(%rip) ; 0x18009adb9
```

### What these checks actually do

The checks are not a direct `gfx1032` reject. They are generic architecture-string controls:

1. `gfx1` prefix test:
   - validates the leading bytes of the architecture string
2. `strlen == 7` test:
   - valid for short architecture names such as `gfx1032`, `gfx1100`, `gfx1200`, etc.
3. `gfx1200` special case:
   - detects one special value in the family
4. `R14D` helper result test:
   - uses an imported helper result in a temporary boolean

The final store is not driven by a direct compare to `gfx1032`; instead the code forces the final `AL` to `1` before writing the byte.

Therefore these checks determine only whether execution reaches the write site and then the later byte gate. They do not directly encode “AMD GPU unsupported“ in this block.

---

## 7. Exact conditions that can reach `0x18000ACA0`

The earliest branch conditions affecting reachability are:

### Condition 1: initial `gfx1` family detection

```asm
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87
```

- This is a test for whether the architecture string begins with `gfx1`.
- If it is not `gfx1`, the block jumps away.
- If it is, execution continues.

### Condition 2: length check

```asm
18000ac77: movq %r15, %rcx
18000ac7a: callq 0x180064e20
18000ac7f: cmpq $0x7, %rax
18000ac83: sete %sil
```

- `SIL = 1` if `strlen(gcnArchName) == 7`.
- This is a generic family rule, not a rejection rule.

### Condition 3: `gfx1200` special-case compare

```asm
18000ac87: movabsq $0x30303231786667, %rax ; "gfx1200"
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl
```

- `BL = 1` only if the string is exactly `gfx1200`.

### Condition 4: helper return check

```asm
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
```

- `AL = 1` when `R14D == 0`.
- This is a temporary result only.

### Important: this does not decide the byte value

```asm
18000aca0: movb $0x1, %al
```

This unconditional overwrite is the decisive fact. Once the block is reached, the byte is written as `1` regardless of the conditions above.

---

## 8. Trace the operands backward to origin

### 8.1 `gcnArchName`

The architecture name is a `hipDeviceProp_t` member:

```c
char gcnArchName[256];
```

This is confirmed in the ROCm HIP header:

- `C:\Program Files\AMD\ROCm\7.2\include\hip\hip_runtime_api.h`
- field: `char gcnArchName[256]; // AMD GCN Arch Name. HIP Only.`

The code inspects it here:

```asm
18000ac4d: movl $0x31786667, %eax ; "gfx1"
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax, %ecx
18000ac75: jne 0x18000ac87
```

This is effectively checking the leading bytes of the architecture string in the `hipDeviceProp_t` structure.

### 8.2 `hipDeviceProp_t` / property buffer

The property buffer is populated by a HIP call:

```asm
18000ab91: leaq -0x20(%rbp), %r12
18000ab95: movl $0x5c0, %r8d
18000ab9b: movq %r12, %rcx
18000ab9e: xorl %edx, %edx
18000aba0: callq 0x180064a00
18000aba5: movq %r12, %rcx
18000aba8: movl %ebx, %edx
18000abaa: callq 0x180065a20
```

This is the property-query path. The pointer to the `hipDeviceProp_t` buffer is in `R12` and is then used to pull the string data for `gcnArchName`.

### 8.3 helper result in `R14D`

The value that feeds `R14D` is assigned here:

```asm
18000abcb: callq 0x180065ae0
18000abd0: movl %eax, %r14d
```

Then used here:

```asm
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
```

This shows `R14D` is not a direct `gfx1032` compare; it is a helper return value, likely a status/capability bit or a numeric predicate from an imported helper. It is only a temporary temporary boolean used in this block.

### 8.4 device count / index precedence

The device selection is driven by `EBX` / `EDX` in the surrounding code:

```asm
18000ab8f: xorl %esi,%esi
18000ab91: leaq -0x20(%rbp), %r12
...
18000aba8: movl %ebx, %edx
18000abaa: callq 0x180065a20
```

This means the call is per device index. The device index is an external runtime input, not a fixed `gfx1032` check.

---

## 9. State machine

```text
INITIAL VALUE
   (unknown / zeroed by default or before this function)
        ↓
set/clear logic around 0x18009ADB9
   + write 1: 0x18000ACA2 -> movb %al, [0x18009ADB9]  ; sets to 1
   + write 2: 0x18000AD4C -> movb $0x0, [0x18009ADB9] ; clears to 0
        ↓
0x18000AD53: cmpb $0x0, [0x18009ADB9]
        ↓
if zero -> go to accepted path 0x18000BA43
if nonzero -> continue to later unsupported / diagnostic path
        ↓
0x18000D205: testb $0x1, [0x18009ADB9]
        ↓
if bit set -> unsupported message path
```

This is a classic state-flag machine, not a direct `gfx1032` blacklist.

---

## 10. Earliest conditional branch that causes the state to become nonzero

The earliest branch that can permit the nonzero set is the path reached from the `gfx1` validation block after the property string is accepted as a valid `gfx1` architecture name.

### Exact earliest branch in the relevant chain

```asm
18000ac75: jne 0x18000ac87
```

This branch determines whether the initial `gfx1` prefix test fails or passes. If it passes, execution continues through the length check and then to the final state write.

### But the actual “nonzero state” is forced by:

```asm
18000aca0: movb $0x1, %al
18000aca2: movb %al, [0x18009ADB9]
```

So the earliest branch that leads to a nonzero state is the continuation of the `gfx1`-family validation path, not a direct `gfx1032` comparison.

---

## 11. What should an AMD GPU at this point be expected to look like?

The code expects a valid `hipDeviceProp_t` object and a valid `gcnArchName` string. In AMD HIP, that field is a textual architecture string such as:

- `gfx1032`
- `gfx1100`
- `gfx1200`
- etc.

The code then tries to validate that the returned string is one of the expected `gfx1...` strings and continues through a generic state-machine path. It does not explicitly say “if `gfx1032` then reject”; it says “if the runtime device metadata passes the local validation path, set the state byte and continue to later checks.”

The static evidence therefore supports the interpretation that the byte is a generic runtime gate, not a direct whitelist/blacklist for `gfx1032`.

---

## 12. Final conclusion

The state byte at `0x18009ADB9` is not the raw `gfx1032` decision itself.

Instead:

- one branch path sets it to `1` at `0x18000ACA2`
- a different branch path clears it to `0` at `0x18000AD4C`
- the final accepted/rejected behavior is chosen at `0x18000AD53` and consumed again at `0x18000D205`

This means the functionality is a generic runtime state flag whose value is driven by architecture-string validation + helper-status logic, not by a direct compare to `gfx1032`.

No patch recommendation is made here.

---

## EARLIEST FAILURE CONDITION

- address: `0x18000ACA2`
- original bytes: `88 05 11 01 09 00`
- branch: `0x18000AC75: jne 0x18000AC87` followed by the later `0x18000ACA8: je 0x18000ACB6`
- condition: the code has reached the generic `gfx1`-name validation path, and the final state byte write is performed after `AL` is forced to `1` by `movb $0x1, %al`
- source value: `AL = 1` from `movb $0x1, %al` immediately before `movb %al, [0x18009ADB9]`
- what makes it true: the block reaches `0x18000ACA0` after a valid `gfx1` string passes the prefix/length checks and then falls through the helper-result check
- what makes it false: if the early `gfx1` prefix test fails and the block jumps away before the final set operation, or if the alternate zero-clear path at `0x18000AD4C` is reached
- relation to gfx1032: there is no direct `gfx1032` compare here; `gfx1032` would satisfy the generic `gfx1` + 7-character family pattern and therefore still reach the same unconditional set
- confidence: high for the write semantics, medium for final runtime intent because the exact helper return value in `R14D` is only indirectly known from static disassembly and requires a live device dump to confirm the actual AMD GPU property values
