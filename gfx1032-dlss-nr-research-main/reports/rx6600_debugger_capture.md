# RX 6600 debugger capture procedure for the live HIP compatibility gate

## Goal

Capture one clean runtime snapshot of the live RX 6600 GPU properties at the first `hipGetDevicePropertiesR0600` query, without modifying `version.dll`.

This procedure is intentionally limited to data collection. The aim is to record the actual values used by the compatibility gate at these addresses:

- `0x18000AA09`
- `0x18000AD1A`
- `0x18000AC4D`
- `0x18000AC75`
- `0x18000AC87`
- `0x18000AC97`
- `0x18000ACA2`
- `0x18000AD4C`
- `0x18000AD53`

The DLL is assumed to load at its normal image base:

- `0x180000000`

This is the address basis for the static offsets below. If ASLR changes the runtime base, the debugger must be used to determine the actual loaded base and then convert the address accordingly.

---

## 1. How to handle ASLR and runtime base differences

The static addresses are valid only if `version.dll` is loaded at `0x180000000`.

### If the module is rebased by ASLR

Use the debugger to read the actual base of `version.dll` and compute the delta:

### WinDbg

```text
.lm
lm m version
```

Look for the `Base` value for `version.dll`.

If the output shows a different base, for example:

```text
Base       Size      Module name
00007ff7`7d100000  0x0000000000010000 version
```

then the runtime module base is `0x00007ff77d100000`.

The general formula is:

```text
runtime_va = runtime_base + (static_va - 0x180000000)
module_relative = runtime_va - runtime_base
```

Example:

```text
runtime_base = 0x00007ff77d100000
static_va = 0x18000AA09
delta = 0x18000AA09 - 0x180000000
runtime_va = runtime_base + delta
```

In WinDbg:

```text
? 0x18000AA09 - 0x180000000
? 0x00007ff77d100000 + (0x18000AA09 - 0x180000000)
```

### x64dbg

Use the module list or the command bar to read the base address of `version.dll`.

Then compute the real runtime address as:

```text
real_target = module_base + (0x18000AA09 - 0x180000000)
```

If the module is not at the standard base, do not break on the hardcoded absolute addresses; break on the computed runtime address or on the module-relative symbol address once the DLL is loaded.

---

## 2. Preferred runtime procedure: break immediately after the first `hipGetDevicePropertiesR0600` call

The single best capture point is the instruction immediately after the first call at `0x18000AA09`.

### WinDbg, if loaded at the normal base

```text
bp 0x18000AA09
g
```

Then capture the registers and memory before continuing:

```text
r
r rax rbx rcx rdx rsi rdi r14 r15 rbp rsp rip
```

To break after the call returns, use the next instruction address:

```text
bp 0x18000AA0E
g
```

The first call often returns a HIP status in `EAX` and the property struct pointer in `RCX`/`R14`/`RSI` depending on the caller.

### x64dbg

```text
bp 0x18000AA09
```

Then at the breakpoint, capture:

```text
r
```

or, more explicitly:

```text
r eax
r ebx
r ecx
r edx
r esi
r edi
r r14
r r15
r rbp
```

If you want the CPU state after the call returns, set a second breakpoint at:

```text
bp 0x18000AA0E
```

This is the cleanest place to read `EAX` as the HIP return value.

---

## 3. What to record at each breakpoint

For every target breakpoint below, record the following minimum set:

- `RAX`, `RBX`, `RCX`, `RDX`, `RSI`, `RDI`, `R14`, `R15`, `RBP`, `RSP`, `RIP`
- `EAX`, `EDX`, `R14D`, `R15D`, `EDI`, `ESI`
- `ZF`, `CF`, `SF`, `OF` flags after each `test` or `cmp`
- the property buffer pointer passed to `hipGetDevicePropertiesR0600`
- the first 256 bytes of the device property struct
- the `name[256]` string
- the `gcnArchName` string if visible in the same property block
- `[RBP+0x468]` and `[RBP+0x46C]`
- the state byte at `0x18009ADB9`

---

## 4. Breakpoint 1: first `hipGetDevicePropertiesR0600` call at `0x18000AA09`

### Why this address matters

This is the first time the DLL requests live device properties for the active GPU. We need the exact buffer and result before the compatibility comparisons begin.

### Registers to record

```text
r
r rax rbx rcx rdx rsi rdi r14 r15 rbp
r rax r14d r15d edi esi
```

### Memory to inspect

#### 1) The property buffer pointer

Look at the first argument to `hipGetDevicePropertiesR0600`:

- `RCX` is the property buffer pointer in a typical call sequence
- `R14` or `RSI` may also be the local property buffer pointer depending on caller state

#### 2) The full struct

WinDbg:

```text
dq @$rcx L0x20
db @$rcx L0x100
du @$rcx L0x40
```

x64dbg:

```text
dq rcx L0x20
db rcx L0x100
du rcx L0x40
```

This dump should contain the GPU name string at the beginning of the structure.

### How to identify the HIP property buffer

The property buffer is the pointer passed as the first argument to `hipGetDevicePropertiesR0600`.

The cleanest evidence is:

- the `RCX` value at the call site,
- the fact that the dump begins with a printable GPU name (`AMD Radeon RX 6600` or similar),
- the presence of `major`, `minor`, and the AMD architecture string later in the same block.

### How to inspect the GPU name

Because `hipDeviceProp_t.name[256]` is the first major field in the struct, dump the start of the block:

```text
du @$rcx L0x40
```

or

```text
db @$rcx L0x100
```

The output should show the GPU name as ASCII.

### How to inspect the architecture string

Search the property block for the `gfx` string rather than assuming a fixed offset.

WinDbg:

```text
s -a @$rcx L0x500 gfx
```

or more generally:

```text
s -a @$rcx L0x500 gfx
s -a @$rcx L0x500 AMD
```

x64dbg:

```text
s rcx L0x500 "gfx"
```

If the property block is large enough, also check nearby memory after the `gfx` hit:

```text
du @$rcx+0x200 L0x20
```

or

```text
du <address_found_by_s> L0x20
```

The architecture string normally appears as a printable ASCII string such as `gfx1032`.

### How to capture the HIP return value

At the call site, the return value is in `EAX` after the call completes. The cleanest capture method is to break on the next instruction after the call.

WinDbg:

```text
bp 0x18000AA0E
g
r eax
```

x64dbg:

```text
bp 0x18000AA0E
```

On break:

```text
r eax
```

`EAX == 0` means success for HIP.

### How to capture `R14D` / `RDI` / `R15`

At the call site or immediately afterwards:

```text
r r14d r15d edi
```

In WinDbg:

```text
r r14d r15d rdi
```

This is important because the subsequent compatibility checks compare the local values and the architecture-derived state.

### How to capture `[RBP+0x468]` and `[RBP+0x46C]`

At the same point in the function, dump the local stack area:

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
```

or

```text
db @$rbp+0x46c L0x10
```

x64dbg:

```text
dq rbp+0x468 L0x8
db rbp+0x468 L0x20
db rbp+0x46c L0x10
```

This is exactly the local data that needs to be compared and later written to the global flag.

### Conditional flags that matter immediately after each comparison

At the compare/test instructions, record the flags after the instruction executes:

```text
r efl
```

or in x64dbg:

```text
r flags
```

The critical flags at the relevant block are:

- `ZF` after `test` or `cmp`
- `SF` if signed comparison is involved
- `CF` when checking carry or unsigned relationships
- `OF` for signed arithmetic overflow context

The main comparison block is around `0x18000AC4D` and `0x18000AC97`, so capture `EFLAGS` right after those instructions.

---

## 5. Breakpoint 2: second query at `0x18000AD1A`

### Why this matters

This is the second live HIP property query in the failure/diagnostic path. It is the second runtime snapshot of the same device and is crucial for proving whether the rejection logic is tied to the same device data.

### Registers to record

```text
r
r rax rbx rcx rdx rsi rdi r14 r15 rbp
r eax r14d r15d edi esi
```

### Memory to inspect

This is the same property-buffer pattern as the first query, but the pointer now comes from `RSI` or another local buffer. Confirm the pointer by dumping the buffer and searching for the same device name and `gfx` architecture string:

```text
r rsi
r rcx
```

Then:

```text
dq @$rsi L0x20
du @$rsi L0x40
s -a @$rsi L0x500 gfx
```

x64dbg:

```text
r rsi
r rcx

dq rsi L0x20
du rsi L0x40
s rsi L0x500 "gfx"
```

### How to identify the property buffer here

The property buffer is the pointer passed to the second `hipGetDevicePropertiesR0600` call. The buffer should again contain a real device-name string and the AMD architecture string.

### Capture `R14D` / `RDI` / `R15`

At the relevant points after the call, record:

```text
r r14d r15d edi rdi r15
```

### Capture `[RBP+0x468]` and `[RBP+0x46C]`

Same as before:

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
db @$rbp+0x46c L0x10
```

---

## 6. Breakpoint 3: compare / mask site at `0x18000AC4D`

### Why this matters

This site is one of the first explicit compatibility checks. It is a key point for determining whether the decision is driven by a string or numeric mask value.

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
```

### What matters immediately after the comparison

The result of the `xor` / `test` sequence determines the next branch. We need `ZF` and `CF` after the instruction at `0x18000AC4D` and nearby branches.

```text
r efl
```

`ZF == 1` is especially important because it indicates the value being compared is zero or equal after XOR/test.

---

## 7. Breakpoint 4: branch site at `0x18000AC75`

### Why this matters

This is near the second architectural compare and helps determine whether a string-length or masked-string check is happening.

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
db @$rbp+0x46c L0x10
```

### Conditional flags to capture

Right after the compare/test at this location, capture:

```text
r efl
```

The critical values are:

- `ZF` for equality or zero test
- `SF` for signed comparison usage
- `CF` if an unsigned comparison has happened

---

## 8. Breakpoint 5: branch site at `0x18000AC87`

### Why this matters

This is the second branch point in the same compatibility chain. It helps determine if the code is rejecting due to a string comparison or due to a numeric helper result.

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
```

### Conditional flags to capture

Immediately after the compare/test:

```text
r efl
```

Record `ZF`, `CF`, `SF`, and `OF`.

---

## 9. Breakpoint 6: compare / test site at `0x18000AC97`

### Why this matters

This is the critical `testl %r14d, %r14d` site that feeds the branch that ultimately writes the state byte.

### Registers to record

```text
r r14d r15d eax ecx edi esi
r efl
```

### Memory to inspect

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
db @$rbp+0x46c L0x10
```

### Conditional flags to capture

This instruction is the key gate:

```asm
testl %r14d, %r14d
```

The flags matter here because:

- `ZF = 1` means `R14D == 0`
- `ZF = 0` means the value is nonzero
- `SF` and `OF` are useful context, but `ZF` is the deciding flag for the branch

```text
r efl
```

This is the cleanest place to confirm whether the compatibility decision is driven by the helper-output register `R14D`.

---

## 10. Breakpoint 7: state-byte write at `0x18000ACA2`

### Why this matters

This is the state byte write that drives the later rejection path:

```asm
movb %al, 0x18009adb9
```

This writes to the global compatibility flag.

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

#### 1) The memory flag

```text
? poi(0x18009ADB9)
```

or, in WinDbg, if the memory is readable as a byte:

```text
db 0x18009ADB9 L0x8
```

x64dbg:

```text
db 0x18009ADB9 L0x8
```

#### 2) The local values feeding the flag

```text
dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
db @$rbp+0x46c L0x10
```

### What to capture here

- `AL` before the store
- the resulting byte at `0x18009ADB9`
- the state of `ZF` / comparisons already executed

This is the point where the compatibility path decides whether the device is accepted or rejected.

---

## 11. Breakpoint 8: state-byte clear at `0x18000AD4C`

### Why this matters

The failure path clears the global state byte before the final zero-check.

```asm
movb $0x0, 0x18009adb9
```

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

```text
db 0x18009ADB9 L0x8
```

### What to capture

Immediately after the write, check the byte:

```text
db 0x18009ADB9 L0x8
```

The global state should be zero after this path executes.

---

## 12. Breakpoint 9: final check at `0x18000AD53`

### Why this matters

This is the final state check:

```asm
cmpb $0x0, 0x18009adb9
```

This determines whether the unsupported path is reached.

### Registers to record

```text
r eax ecx r14d r15d edi esi
r efl
```

### Memory to inspect

```text
db 0x18009ADB9 L0x8
```

### Conditional flags to capture

After the compare, record:

```text
r efl
```

The crucial flag is:

- `ZF = 1` means the byte is equal to zero
- `ZF = 0` means the byte is nonzero and the branch path is different

This tells you whether the final unsupported path is being taken based on the state byte.

---

## 13. Combined WinDbg capture script

This is a copy-paste-friendly WinDbg capture sequence for the first real query and the state flag:

```text
.lm
lm m version

bp 0x18000AA09
bp 0x18000AA0E
bp 0x18000AC4D
bp 0x18000AC75
bp 0x18000AC87
bp 0x18000AC97
bp 0x18000ACA2
bp 0x18000AD4C
bp 0x18000AD53

g

r
r rax rbx rcx rdx rsi rdi r14 r15 rbp
r rax r14d r15d edi esi

r rcx
r rsi
r r14
r r15

? @$rcx
? @$rsi
? @$r14
? @$r15

// property buffer dump
rq @$rcx L0x20
// if dq is preferred

dq @$rcx L0x20
db @$rcx L0x100
du @$rcx L0x40
s -a @$rcx L0x500 gfx
s -a @$rcx L0x500 AMD

// local stack values

dq @$rbp+0x468 L0x8
db @$rbp+0x468 L0x20
db @$rbp+0x46c L0x10

// flags
r efl

// state byte
db 0x18009ADB9 L0x8
```

Use `g` repeatedly between breakpoints and keep the output for each stop. The first `0x18000AA09` stop is the primary clean snapshot.

---

## 14. Combined x64dbg capture sequence

This is the x64dbg equivalent for the same points:

```text
bp 0x18000AA09
bp 0x18000AA0E
bp 0x18000AC4D
bp 0x18000AC75
bp 0x18000AC87
bp 0x18000AC97
bp 0x18000ACA2
bp 0x18000AD4C
bp 0x18000AD53
```

Then, at each breakpoint:

```text
r
r eax ebx ecx edx esi edi r14 r15 rbp

r eax
r ecx
r edx
r r14d
r r15d
r edi
r esi

// dump property buffer
dq rcx L0x20
db rcx L0x100
du rcx L0x40

// search for architecture string
s rcx L0x500 "gfx"

// inspect local stack values
dq rbp+0x468 L0x8
db rbp+0x468 L0x20
db rbp+0x46c L0x10

// flags
r flags

// global state byte
db 0x18009ADB9 L0x8
```

The first stop at `0x18000AA09` is the clean snapshot to preserve.

---

## 15. How to decide whether the first property buffer is valid

A valid buffer should show all of the following in the raw memory dump:

- a printable name string at the start of the struct
- `major` / `minor` fields with a meaningful GPU version
- a later AMD architecture string such as `gfx1032`
- no obvious zeroed or garbage values in the device name area

If the `name[256]` string is empty or garbage, the query may be failing or the pointer may be wrong.

If the property block contains a valid AMD name and `gfx1032`, then the rejection is very likely in the compatibility logic after the query, not in the query itself.

---

## 16. Expected final evidence from this procedure

The desired final evidence is one clean snapshot showing:

- the exact device name reported by the RX 6600
- the exact AMD architecture string from the property block
- the exact HIP return code from `hipGetDevicePropertiesR0600`
- the exact value of `R14D`/`RDI`/`R15`
- the exact bytes at `[RBP+0x468]` and `[RBP+0x46C]`
- the exact `ZF`, `CF`, `SF`, and `OF` states after the compare/test instructions
- the final value of the global state byte at `0x18009ADB9`

This is the evidence needed to determine whether the unsupported path is driven by:

- an architecture string mismatch,
- a helper result in `R14D`,
- or a later state-byte toggling path.

---

## 17. Important constraints

- Do not modify `version.dll`.
- Do not patch the binary.
- Do not propose a compatibility patch in this procedure.
- This is only a runtime capture and proof step.
- The immediate goal is one clean snapshot of the real RX 6600 at the first `hipGetDevicePropertiesR0600` call.

This is the minimum evidence needed before any patch idea can even be meaningfully evaluated.
