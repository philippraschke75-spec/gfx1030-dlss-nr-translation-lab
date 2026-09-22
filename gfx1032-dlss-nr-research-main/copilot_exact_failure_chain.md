# Exact branch chain to `0x18000ACA0` / `0x18000ACA2`

## Scope

This report traces the complete conditional chain that decides whether execution reaches:

```asm
18000aca0: b0 01          movb    $0x1, %al
18000aca2: 88 05 11 01 09 00 movb    %al, 0x90111(%rip) ; 0x18009adb9
```

The goal is to determine the exact branch logic without assuming the answer. This is a static analysis of `version.dll`, with no DLL modification.

---

## 1. Immediate target and local state

The write target is the global byte:

- `0x18009ADB9`

The immediate preceding instructions are:

```asm
18000ac97: 45 85 f6        testl   %r14d, %r14d
18000ac9a: 0f 94 c0        sete    %al
18000ac9d: 40 08 f3        orb     %sil, %bl
18000aca0: b0 01           movb    $0x1, %al
18000aca2: 88 05 11 01 09 00 movb    %al, 0x90111(%rip) ; 0x18009adb9
```

The decisive fact is this:

```asm
18000aca0: movb $0x1, %al
```

This overwrites any prior result from the `testl %r14d,%r14d` / `sete %al` sequence. The final write to the byte is therefore forced to `1` once execution reaches `0x18000ACA2`.

This means the real question is not “what value does AL take at ACA2?” but rather “what branch chain reaches ACA0/ACA2 in the first place?”

---

## 2. Full conditional branch chain that can reach `0x18000ACA0`

### Branch 1: device-index / initialization sanity gate

Address: `0x18000AB87`

Original bytes:

```asm
45 85 db
```

Disassembly:

```asm
18000ab87: 45 85 db  testl   %ebx, %ebx
18000ab89: 78 76     js      0x18000ad01
```

Branch target: `0x18000AD01`
Fall-through target: `0x18000AB8F`
Condition being tested:
- `EBX < 0`
Registers / memory operands involved:
- `EBX`
Where `EBX` was last defined:
- earlier in the same function, as the currently selected device index / active device handle state; this is not a `gcnArchName` value
- this is a device-selection / runtime index variable

This branch does not set the state byte to `1`; it routes to a different block that eventually clears it to zero at `0x18000AD4C`.

If this branch is not taken, execution continues to the property-query and architecture-string validation path.

### Branch 2: `gfx1` prefix gate within the architecture-string check

Address: `0x18000AC73`

Original bytes:

```asm
09 c1
```

Disassembly:

```asm
18000ac73: 09 c1       orl     %eax, %ecx
18000ac75: 75 10       jne     0x18000ac87
```

Branch target: `0x18000AC87`
Fall-through target: `0x18000AC77`
Condition being tested:
- `((EAX ^ ???) OR ECX) != 0` effectively checks whether the initial bytes of the string are not the `gfx1` prefix
- more precisely, it tests the first four bytes and the fifth byte of the architecture string in the `hipDeviceProp_t` buffer

Registers / memory operands involved:
- `EAX`
- `ECX`
- `[RBP+0x468]`
- `[RBP+0x46C]`

Where those operands were last defined:
- `EAX`:

```asm
18000ac4d: b8 67 66 78 31  movl    $0x31786667, %eax ; 'gfx1'
18000ac52: 33 85 68 04 00 00 xorl 0x468(%rbp), %eax
```

- `ECX`:

```asm
18000ac58: 0f b6 8d 6c 04 00 00 movzbl 0x46c(%rbp), %ecx
18000ac5f: 83 f1 31 xorl $0x31, %ecx
```

- `[RBP+0x468]` and `[RBP+0x46C]` are within the local property object / `hipDeviceProp_t` buffer and belong to the architecture string region, strongly consistent with `gcnArchName` or the bytes immediately adjacent to it

This branch is specifically architecture-string based, but it is not a direct `gfx1032` rejection. It only checks whether the string begins with the `gfx1` prefix.

Important: the branch does not prevent the later status-byte write. It jumps to `0x18000AC87`, which is still within the same validation block. The code still continues to the end of the block and reaches `0x18000ACA2`.

### Branch 3: string-length check

Address: `0x18000AC7F`

Original bytes:

```asm
48 83 f8 07
```

Disassembly:

```asm
18000ac77: 4c 89 f9         movq    %r15, %rcx
18000ac7a: e8 a1 a1 05 00   callq   0x180064e20
18000ac7f: 48 83 f8 07      cmpq    $0x7, %rax
18000ac83: 40 0f 94 c6      sete    %sil
```

There is no branch instruction here; this is a compare + set, not a branch. However, the result is used as a boolean:

- `SIL = 1` iff `strlen(gcnArchName-like string) == 7`

The operands originate from:
- `R15` as a pointer into the property string region
- `callq 0x180064e20` which is a `strlen`-like helper

This is a generic string-length test, not a `gfx1032` rejection.

### Branch 4: `gfx1200` special-case compare

Address: `0x18000AC91`

Original bytes:

```asm
48 39 c7
```

Disassembly:

```asm
18000ac87: 48 b8 67 66 78 31 32 30 30 00  movabsq $0x30303231786667, %rax ; "gfx1200"
18000ac91: 48 39 c7                    cmpq    %rax, %rdi
18000ac94: 0f 94 c3                    sete    %bl
```

This is a direct compare against one specific string literal. It sets `BL = 1` only when the string equals `gfx1200`.

There is no direct compare to `gfx1032` here.

### Branch 5: helper-return / temporary AL test

Address: `0x18000AC97`

Original bytes:

```asm
45 85 f6
```

Disassembly:

```asm
18000ac97: 45 85 f6      testl   %r14d, %r14d
18000ac9a: 0f 94 c0      sete    %al
```

This sets `AL = 1` if `R14D == 0`.

Operands involved:
- `R14D`

Where `R14D` was last defined:

```asm
18000abcb: e8 10 af 05 00   callq   0x180065ae0
18000abd0: 41 89 c6         movl    %eax, %r14d
```

This is a helper-return value, not a direct architecture-name comparison. The helper is external to the local function; the exact semantic is not statically proven from the local code alone. It is likely a capability or status predicate, but the actual meaning cannot be concluded from static analysis alone.

### Branch 6: post-write branch after the store

Address: `0x18000ACA8`

Original bytes:

```asm
74 0c
```

Disassembly:

```asm
18000aca8: 74 0c  je      0x18000acb6
```

Branch target: `0x18000ACB6`
Fall-through target: `0x18000ACAA`
Condition being tested:
- zero flag from the earlier `testl %r14d,%r14d`

This branch is reached after the store to `[0x18009ADB9]` has already happened. It does not control whether the state byte is written; it only decides whether a later code path is taken after the state-set.

This is the crucial difference: the branch at `0x18000ACA8` is after the state write, not before it.

---

## 3. The key fact: the state write is not gated by a branch

The condition chain to reach `0x18000ACA0/ACA2` is:

- selected device index is not negative;
- then the function calls `hipGetDevicePropertiesR0600` and inspects the architecture string
- it checks the `gfx1` prefix and the length check
- it may set `BL` for `gfx1200`
- it checks the helper return in `R14D`
- then the code executes:

```asm
movb $0x1, %al
movb %al, [0x18009ADB9]
```

There is no branch immediately before this store that prevents the write. The state byte is written on the path that reaches it, regardless of `R14D`, `SIL`, or `BL`, because the store is preceded by an unconditional `movb $0x1,%al`.

Therefore, the architecture checks do not determine the final write value. They only determine whether the code is in that validation block and then the write happens.

---

## 4. Pseudocode reconstruction derived from actual branches

This is the actual branch structure exposed by the code, without guessing beyond the disassembly:

```c
if (device_index < 0) {
    // alternate path to zero-clear block
    state_byte = 0;
} else {
    props = hipGetDevicePropertiesR0600(device_index);

    // local property buffer contains architecture string data
    // the code reads bytes from props and compares against gfx1 prefix
    if (props.gcnArchName does not start with "gfx1") {
        // branch to 0x18000AC87, still in same block
        // no early exit; control continues to the rest of the block
    }

    len = strlen(props.gcnArchName);
    if (len == 7) {
        sil = 1;
    } else {
        sil = 0;
    }

    if (props.gcnArchName == "gfx1200") {
        bl = 1;
    } else {
        bl = 0;
    }

    helper_ret = helper_function_return(); // in R14D
    if (helper_ret == 0) {
        al = 1;
    } else {
        al = 0;
    }

    bl = bl | sil;

    // critical override:
    al = 1;
    state_byte = al;  // writes 1 to 0x18009ADB9

    if (ZF_from_testl_r14d == 0) {
        goto 0x18000ACB6;
    }
}
```

This is faithful to the instructions present. The important part is that the unconditional overwrite at `0x18000ACA0` nullifies any previous value in `AL`.

---

## 5. Alternate path to `0x18000AD4C`

The alternate zero-write path begins with the negative-index condition:

```asm
18000ab87: 45 85 db  testl   %ebx, %ebx
18000ab89: 78 76     js      0x18000ad01
```

If `EBX < 0`, execution jumps to `0x18000AD01`.

The code then goes through a secondary property/status block and eventually reaches:

```asm
18000ad47: e8 2a 00 00 00 callq 0x180007c00
18000ad4c: c6 05 66 00 90 00 00 movb $0x0, 0x90066(%rip) ; 0x18009adb9
```

This is the explicit zero path.

So the state machine is:

- negative-device-index / alternate-status path => `0x18009ADB9 = 0`
- architecture-validation path => `0x18009ADB9 = 1`

The later test at `0x18000AD53` then decides whether to take the accepted path or continue toward the unsupported-device flow.

---

## 6. What is being tested, really?

### Architecture string

Yes, the code reads the architecture string and performs:

- prefix test for `gfx1`
- length check
- special case against `gfx1200`

But this does not constitute a `gfx1032` reject condition.

### Helper return value

The code also reads a helper return into `R14D`:

```asm
18000abcb: callq 0x180065ae0
18000abd0: movl %eax, %r14d
```

and then tests it here:

```asm
18000ac97: testl %r14d, %r14d
```

This is a numeric status/capability predicate. Its exact meaning is not established statically, but it is not a literal `gfx1032` comparison.

### Capability query

The code clearly queries a device property structure through the HIP runtime and then reads string and helper-return values. The final behavior is runtime, not compile-time.

### Conclusion

The architecture string is not the discriminator in the code path that sets `0x18009ADB9` to `1`.

The relevant discriminator is not a direct `gfx1032` compare; it is a runtime validation/state machine around:

- `hipGetDevicePropertiesR0600`
- `hipDeviceProp_t`
- `gcnArchName`
- a helper return into `R14D`
- the device index / status state

and the final state is then consumed by the later `cmpb $0x0,[0x18009ADB9]` gate.

---

## 7. Earliest static divergence between supported GPU and RX 6600

The static code does not disclose a direct “if architecture == gfx1032 then reject” condition.

The earliest branch that can change control flow is:

```asm
18000ab87: 45 85 db
18000ab89: 78 76 js 0x18000ad01
```

This is a negative index / alternate status gate, not an RX 6600-specific check.

The next branch is:

```asm
18000ac73: 09 c1
18000ac75: 75 10 jne 0x18000ac87
```

This checks the `gfx1` prefix, not `gfx1032` specifically.

There is no static branch that specifically distinguishes an RX 6600 (`gfx1032`) from a supported GPU in this block. The actual state-bit behavior is generic and the final nonzero value is forced by `movb $0x1,%al`.

Therefore, the earliest static divergence between a supported GPU and the RX 6600 cannot be established from the disassembly alone. The code is not statically proving a unique `gfx1032` rejection point.

---

## EARLIEST DIVERGENCE

- address: `0x18000AB87`
- bytes: `45 85 db 78 76`
- instruction: `testl %ebx, %ebx ; js 0x18000ad01`
- condition: `EBX < 0`
- true path: `0x18000AD01` → zero-clear path at `0x18000AD4C`
- false path: `0x18000AB8F` → continue property-query and `gcnArchName` validation
- source of condition: device index / runtime selection state in `EBX`
- RX 6600 implication: this does not specifically identify RX 6600; it is a generic runtime-index or status gate and does not prove an AMD GPU rejection by architecture name alone
- confidence: medium for branch existence, low for a unique RX 6600-specific divergence, because the actual architecture-string and helper-return logic does not directly discriminate `gfx1032` in the static disassembly
