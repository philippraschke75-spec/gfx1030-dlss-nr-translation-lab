# Broad static reverse-engineering investigation of version.dll

## 1. Executive conclusion

This broad static investigation did not find a proven direct `gfx1032` discriminator in the main GPU-init path. The strongest evidence is:

- the unsupported-GPU diagnostic string is loaded from `0x180080A9C`
- the final unsupported path is reached only after the state byte at `0x18009ADB9` is tested
- the state byte is set in the validation block ending at `0x18000ACA2` by an unconditional `movb $0x1,%al`
- the earlier architecture checks are generic `gfx1`-family tests (`gfx1` prefix, 7-character length, `gfx1200` special case), not a direct `gfx1032` blacklist
- the helper return into `R14D` from `0x180065AE0` is indirect and not locally defined; it is not proven to be the compatibility gate

The result is:

- no direct, statically proven discriminator for `gfx1032` was found in the local function body
- the relevant property path is still GPU-driven, but the final decision is not conclusively attributed to a direct architecture-string comparison in static disassembly alone
- therefore, the static path is exhausted and the smallest missing fact is a live runtime snapshot of the GPU property data and the helper return value at the moment the path reaches `0x18000AC97` / `0x18000ACA2`

This meets the required stop condition: `STATIC ANALYSIS EXHAUSTED — RUNTIME CAPTURE REQUIRED`.

---

## 2. Unsupported-message call tree

### 2.1 String base and direct consumer

The unsupported string is at:

- `0x180080A9C`

The string text is:

- `Unsupported GPU (%s): this build runs on Radeon RX 9000 (RDNA4) and RX 7000 (RDNA3) only.`

The relevant direct reference found in disassembly is:

```asm
18000d223: 48 8d 05 4e 7a 00 00  leaq 0x73872(%rip), %r8   ; 0x180080a9c
18000d22a: eb 41                jmp 0x18000d36b
```

This confirms the string pointer is loaded into `%r8` and used in a later formatting / error path.

### 2.2 The later gate that decides whether to take the unsupported path

The decision gate is later in the same artifact:

```asm
18000d205: f6 05 ad db 08 00 01  testb $0x1, 0x8dbad(%rip) ; 0x18009adb9
18000d20c: 75 21                jne   0x18000d22f
```

This means:

- if the state byte bit 0 is set (`1`), the code jumps to the unsupported-message path that eventually loads `0x180080A9C`
- if the bit is clear (`0`), it continues into the non-error flow

This is the undisputed final gate for the unsupported message.

### 2.3 Backward chain to the state byte

Before the late gate, the state byte is tested at:

```asm
18000ad53: 80 3d 5f 00 09 00 00  cmpb $0x0, 0x9005f(%rip) ; 0x18009adb9
18000ad5a: 0f 84 e3 0c 00 00     je   0x18000ba43
```

This is the immediate read of the state byte before the accepted-versus-unsupported split. The path is:

- if zero: accepted path to `0x18000BA43`
- if nonzero: continue to the unsupported path / later gate

### 2.4 Full unsupported path summary

```text
0x18000ACA2 -> set status byte = 1
    |
    v
0x18000AD53 -> cmpb $0x0, [0x18009ADB9]
    |
    +-- if zero -> accepted path 0x18000BA43
    |
    +-- if nonzero -> continue deeper runtime state
    |
    v
0x18000D205 -> testb $0x1, [0x18009ADB9]
    |
    +-- if bit set -> jump to 0x18000D22F
    |
    v
0x18000D22F -> unsupported formatting / diagnostic path
    |
    v
0x180080A9C -> unsupported GPU format string
```

This is the proven unsupported-message backtrace established statically.

---

## 3. Status-byte state machine

### 3.1 The global byte and all writes

The relevant address is:

- `0x18009ADB9`

It has two observed relevant writes in the inspected path:

#### Write 1: set nonzero

```asm
18000aca2: 88 05 11 01 09 00  movb %al, 0x90111(%rip) ; 0x18009adb9
```

Value written: `1`

This write is preceded by:

```asm
18000ac97: 45 85 f6  testl %r14d, %r14d
18000ac9a: 0f 94 c0  sete %al
18000ac9d: 40 08 f3  orb %sil, %bl
18000aca0: b0 01     movb $0x1, %al
```

The decisive fact is that `0x18000ACA0` overwrites `AL` unconditionally, forcing the value to `1` before the store at `0x18000ACA2`.

#### Write 2: clear to zero

```asm
18000ad4c: c6 05 66 00 90 00 00  movb $0x0, 0x90066(%rip) ; 0x18009adb9
```

Value written: `0`

This is the explicit zero path.

### 3.2 The state machine

```text
INITIAL / default state
        |
        v
PATH A: validation / property block
        |
        v
0x18000ACA2: store 1 to [0x18009ADB9]
        |
        v
0x18000AD53: cmpb $0x0, [0x18009ADB9]
        |
        +--> if zero: accepted path (0x18000BA43)
        |
        +--> if nonzero: continue runtime status / unsupported flow
        |
        v
0x18000D205: testb $0x1, [0x18009ADB9]
        |
        +--> if bit 0 set: unsupported message path to 0x180080A9C

ALTERNATE PATH:
        |
        v
0x18000AB87: testl %ebx, %ebx
0x18000AB89: js 0x18000ad01
        |
        v
0x18000AD4C: store 0 to [0x18009ADB9]
        |
        v
0x18000AD53: cmpb $0x0, [0x18009ADB9]
        |
        +--> if zero: accepted path 0x18000BA43
```

### 3.3 What does the byte represent?

The static evidence supports the following:

- it is not a raw `gfx1032` blacklist bit
- it is not a direct unsupported-GPU flag set by a single architecture compare
- it is a generic runtime state byte used in a later accepted-versus-rejected path
- it is best interpreted as a runtime initialization / validation state bit, not a direct GPU-compatibility bit

The exact semantic is not fully proven statically, but explicitly it is not only a helper-return value; it is a global state bit consumed later at two distinct gates.

---

## 4. HIP property access map

Relevant device-property call sites:

- `0x18000AA09`
- `0x18000ABAA`
- `0x18000AD1A`

The local property buffer is prepared here:

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

The key observation:

- `%r12` points at a local `hipDeviceProp_t`-like property buffer on the stack
- `hipGetDevicePropertiesR0600` is called using the device index in `%ebx`
- the property buffer is then examined for string and capability data

The property retrieval call is the imported HIP API:

- `hipGetDevicePropertiesR0600`
- IAT address: `0x180098230`

This matches the known import table.

### 4.1 Property fields that are actually read in the local block

The local code reads from the property buffer and uses it in architecture checks:

```asm
18000ac4d: movl $0x31786667, %eax
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax, %ecx
```

This is strongly consistent with reading the leading bytes of the architecture string in a `hipDeviceProp_t` object, specifically the `gcnArchName` field or bytes immediately adjacent to it.

The relevant HIP definition from ROCm confirms that the AMD architecture string field is:

```c
char gcnArchName[256];
```

in `hipDeviceProp_t`.

### 4.2 What else is read from the property buffer

The property block also uses the generic helper call and numeric operations around the buffer, but the direct architecture logic is local and string-oriented. The local function is not directly checking a large set of numeric property fields in this path; it is parsing the architecture string and then setting a runtime state bit.

### 4.3 Map of direct comparisons

Direct comparisons in the local validation block:

- prefix check against `gfx1`
- length check using `strlen` around `0x180064E20`
- special-case compare against `gfx1200`
- helper return in `R14D` test

There is no direct compare to `gfx1032` or to a full list of supported RDNA3 / RDNA4 architecture names in this block.

---

## 5. Relevant IAT / import map

The relevant imported HIP symbols found in the PE import table are:

- `__hipRegisterFatBinary` at `0x180065D4C`
- `__hipRegisterFunction` at `0x180065D58`
- `__hipRegisterVar` at `0x180065D64`
- `__hipUnregisterFatBinary` at `0x180065D70`
- `hipGetDeviceCount` at `0x180065E00`
- `hipGetDevicePropertiesR0600` at `0x180065E0C`
- other runtime APIs follow, but the relevant ones are the property-query and registration calls

The known IAT slot for the property query is:

- `0x180098230`

This is the actual target for the import thunk resolving `hipGetDevicePropertiesR0600`.

### 5.1 The indirect helper thunk

The other unresolved helper is:

```asm
180065ae0: jmpq *0x327aa(%rip) ; 0x180098290
```

This is a jump through a function pointer at `0x180098290`.

The exact function behind that pointer is not locally defined in `version.dll`; therefore its behavior is not statically provable from this binary alone.

### 5.2 What those imports indicate

The binary is definitely using the HIP runtime to:

- register a fat binary
- set device context / count / properties
- query device properties
- later gate GPU initialization using runtime state

But the static code does not show a direct `gfx1032`-specific imported function that rejects the card.

---

## 6. Architecture-string decision tree

The relevant string logic around `0x18000AC4D` to `0x18000ACA2` is:

```asm
18000ac4d: movl $0x31786667, %eax       ; 'gfx1'
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
18000ac87: movabsq $0x30303231786667, %rax ; 'gfx1200'
18000ac91: cmpq %rax, %rdi
18000ac94: sete %bl
18000ac97: testl %r14d, %r14d
18000ac9a: sete %al
18000ac9d: orb %sil, %bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, [0x18009ADB9]
```

This reconstructs the actual logical pattern:

```c
if (arch_string starts with "gfx1") {
    if (strlen(arch_string) == 7) {
        sil = 1;
    }
    if (arch_string == "gfx1200") {
        bl = 1;
    }
    helper_result = helper_return();
    if (helper_result == 0) {
        al = 1;
    } else {
        al = 0;
    }
    bl = bl | sil;
    al = 1;
    state_byte = 1;
}
```

### 6.1 What this means for specific strings

The static code clearly accepts a broad family of `gfx1`-style architecture names. It does not directly reject `gfx1032`.

Conceptually:

- `gfx1200`: special-case compare sees it and may set `BL` but the final state byte is still forced to `1`
- `gfx1100`, `gfx1101`, `gfx1102`: generic `gfx1`/7-character path, no direct reject
- `gfx1032`: generic `gfx1`/7-character path, no direct reject
- `gfx900`: not in the `gfx1` prefix family; goes outside the relevant block

The actual logical predicate is therefore broad and generic, not a direct `gfx1032` filter.

---

## 7. Capability gate search

The broad static search did not identify a direct capability gate that distinguishes `gfx1032` from supported RDNA3 / RDNA4 GPUs. The code in the relevant path mainly does:

- property retrieval via `hipGetDevicePropertiesR0600`
- local string inspection of `gcnArchName`
- helper return numeric check
- final state-byte set to `1`

This is not a static proof of a true capability gate. The helper is indirect and the numeric property proof is not available locally.

The search did not yield a proven compare against any of the following in the local path:

- `gfx1032`
- `gfx1100`
- `gfx1101`
- `gfx1102`
- `gfx1200`
- wavefront or warp-size thresholds
- explicit matrix / WMMA checks
- numeric `maxThreadsPerMultiProcessor` gate against a known supported list
- explicit `RDNA3` / `RDNA4` device feature bitmask check

The evidence points away from those direct checks and toward a generic state-byte gate built around runtime property validation.

---

## 8. First proven divergence

### Static status

The phrase “first proven divergence” is not satisfied by the available static analysis.

The code does not contain a direct and proven branch that says:

- `if arch == gfx1032 then reject`
- `if device is Rdna3 and not Rdna4 then reject`
- `if maxThreadsPerMultiProcessor < X then reject`
- `if gcnArchName == gfx1032 then unsupported`

Instead, what is actually proven is:

- the later state byte at `0x18009ADB9` is the deciding runtime gate
- it is set to `1` on the relevant validation path
- the earlier block is generic `gfx1` validation and not a `gfx1032` blacklist

Therefore, no direct discriminator for `gfx1032` was proven statically.

The earliest relevant divergence that is statically visible is not a `gfx1032` divergence but a generic runtime-state divergence:

```asm
18000ab87: 45 85 db
18000ab89: 78 76 js 0x18000ad01
```

This is a generic negative-index / alternate status branch. It does not identify gfx1032 specifically.

The next relevant branch is:

```asm
18000ac73: 09 c1
18000ac75: 75 10 jne 0x18000ac87
```

This is a generic `gfx1` prefix test, not a `gfx1032` discriminator.

Thus, no direct supported-vs-`gfx1032` divergence could be proven at the static level.

---

## 9. Exact instruction address and data flow to the missing discriminator

The most important unresolved static fact is still the helper result behind `R14D`:

```asm
18000abcb: e8 10 af 05 00 callq 0x180065ae0
18000abd0: 41 89 c6       movl %eax, %r14d
18000ac97: 45 85 f6       testl %r14d, %r14d
```

The return value is used as an integer status check, but its actual semantics remain unproven because the call target is indirect and not local to the binary.

This is the single remaining missing fact that determines whether the code is genuinely testing a GPU capability or just a generic runtime status bit.

---

## 10. Smallest required runtime capture

The investigation has reached the required stop condition:

`STATIC ANALYSIS EXHAUSTED — RUNTIME CAPTURE REQUIRED`

The smallest useful runtime capture is:

1. the actual `hipDeviceProp_t.gcnArchName` string returned by `hipGetDevicePropertiesR0600` for the RX 6600
2. the helper return value in `R14D` immediately after `0x18000ABCB` and at the moment of `0x18000AC97`

This is the minimum required to resolve the remaining ambiguity. One snapshot of the property buffer and the value in `R14D` is enough; there is no need to dump the whole binary or the full memory image.

---

## 11. Final findings summary

- The unsupported message is definitely reached via the global state byte at `0x18009ADB9`.
- That byte is set to `1` in the validation block ending at `0x18000ACA2`.
- This byte is later checked at `0x18000AD53` and `0x18000D205`.
- The earlier block checks generic `gfx1`-family string characteristics, not a direct `gfx1032` rejection.
- The helper return via `R14D` is not proven to be a compatibility discriminator.
- The exact supported-vs-`gfx1032` divergence was not found statically.

Therefore, the final result is:

- static analysis exhausted
- runtime capture required
- no proof of a direct GPU-discriminator in the binary alone

---

## FIRST PROVEN DIVERGENCE

This broad investigation did not locate a single proven static divergence that distinguishes a supported GPU from `gfx1032`.

The strongest static evidence is generic runtime validation logic, not a direct `gfx1032` check.

Therefore the conclusion is:

- no direct architecture discriminator was proven
- no exact `gfx1032` rejection instruction was found
- runtime capture is required to resolve the remaining missing fact
