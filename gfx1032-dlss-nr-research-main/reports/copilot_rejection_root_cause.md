# Rejection root-cause trace for gfx1032 / RX 6600

## Executive summary

This analysis confirms that `0x18009ADB9` is the key state flag that drives the later unsupported-GPU message. The flag is set in the final device-property validation block at `0x18000ACA2` and is checked at `0x18000D205` before the code loads the unsupported-GPU string at `0x180080A9C`.

The direct evidence makes the following points clear:

- The DLL does register and contain a valid `gfx1032` HIP fat binary.
- `gfx1032` is present in the embedded architecture list and is not rejected by the raw fat-binary registration path.
- The rejection is a later runtime check in the device-property validation logic.
- The code reads the HIP device property structure returned by `hipGetDevicePropertiesR0600`, compares the architecture strings, and then sets the global state flag.
- The exact underlying HIP field that causes the RX 6600/GFX1032 object to take the reject branch is not yet isolated to a single struct member from disassembly alone, but the control flow making the decision is now identified.

## Relevant addresses and the exact rejection path

- `0x18000A800` through `0x18000AD80`: main device-property validation / capability loop
- `0x18000AA09`: call to `hipGetDevicePropertiesR0600`
- `0x18000AA0E`: `test eax, eax` / check success of HIP call
- `0x18000AA12`..`0x18000AA57`: architecture-name checks
- `0x18000AB8F`..`0x18000ACB6`: second validation pass using a local `hipDeviceProp_t`-like buffer
- `0x18000ACA2`: `mov byte ptr [0x18009ADB9], 1`
- `0x18000AD4C`: `mov byte ptr [0x18009ADB9], 0`
- `0x18000AD53`: `cmpb $0x0, [0x18009ADB9]`
- `0x18000D205`: `testb $0x1, [0x18009ADB9]` followed by `jne` to the unsupported message path
- `0x180080A9C`: unsupported-GPU format string

## Full control-flow region: `0x18000A800` to `0x18000AD80`

The following block-level flow is the relevant path:

### Block A: initial loop setup

```asm
18000a8ac: xorl %r13d,%r13d
18000a8af: movl $0xffffffff, 0x9d0(%rbp)
18000a8b9: leaq 0x468(%rbp), %r15
18000a8c0: leaq -0x20(%rbp), %r14
18000a8d4: movsd ...
18000a8dd: movl $0xffffffff, %ebx
18000a8e2: movl $0xffffffff, %esi
18000a8e7: xorl %r12d,%r12d
18000a8ea: jmp 0x18000a9f3
```

This is the start of a per-device loop. `%r14` points to a local buffer used for the HIP property query, and `%r15` is reused as a pointer / architecture-string workspace.

### Block B: device-query loop head

```asm
18000a9f3: movl $0x5c0, %r8d
18000a9f9: movq %r14, %rcx
18000a9fc: xorl %edx,%edx
18000a9fe: call 0x180064a00
18000aa03: movq %r14, %rcx
18000aa06: movl %r12d, %edx
18000aa09: call 0x180065a20 ; hipGetDevicePropertiesR0600
18000aa0e: testl %eax,%eax
18000aa10: jne 0x18000a9e1
```

This is the critical HIP query:

- `RCX = local property buffer`
- `RDX = device index`
- `EAX` is checked immediately
- if `EAX != 0`, the function skips the property-processing block and advances to the next index

This is exactly the call pattern expected for `hipGetDevicePropertiesR0600`.

### Block C: architecture string validation

```asm
18000aa12: movq %r15, %rdi
18000aa15: movq 0x468(%rbp), %r15
18000aa1c: movabsq $-0x1000000000001, %rax
18000aa26: andq %rax, %r15
18000aa29: movl 0x468(%rbp), %eax
18000aa2f: movl $0x31786667, %ecx       ; 'gfx1'
18000aa34: xorl %ecx, %eax
18000aa36: movzbl 0x46c(%rbp), %ecx
18000aa3d: xorl $0x31, %ecx
18000aa40: orl %eax,%ecx
18000aa42: jne 0x18000a8f0
18000aa48: movq %rdi, %rcx
18000aa4b: call 0x180064e20
18000aa50: cmpq $0x7, %rax
18000aa54: sete %dl
18000aa57: jmp 0x18000a8f2
```

This block is validating the architecture string in the returned properties object. The pattern is:

- read the first 64 bits / word from the property-name area
- compare it to a `gfx1` prefix pattern
- then compare the next byte to a `0`/`1`-style discriminator
- if it does not match the expected `gfx1...` pattern, it jumps back to the outer loop and continues scanning.

The call to `0x180064e20` is a string-length helper; the comparison `cmpq $0x7, %rax; sete %dl` tests whether the architecture string length is 7 bytes.

### Block D: name-length / family optimization path

```asm
18000a8f0: xorl %edx, %edx
18000a8f2: movq 0xf0(%rbp), %rcx
18000a8f9: testq %rcx,%rcx
18000a8fc: movzbl %r13b, %r13d
18000a900: movl $0x1, %eax
18000a905: cmovnel %eax, %r13d
18000a909: testb %dl, %dl
18000a90b: movl $0x0, %r8d
18000a911: movl $0xf4240, %eax
18000a916: cmovnel %eax, %r8d
18000a91a: movabsq $0x30303231786667, %rax ; 'gfx1200'
18000a924: cmpq %rax, %r15
18000a927: sete %r9b
18000a92b: movl $0x1e8480, %eax
18000a930: cmovel %eax, %r8d
18000a934: movl 0x164(%rbp), %eax
18000a93a: addl %eax, %r8d
18000a93d: orb %dl, %r9b
18000a940: cmpl %esi, %r8d
18000a943: cmovlel %esi, %r8d
18000a947: movl %ebx, %edx
18000a949: cmovgl %r12d, %edx
18000a94d: testb %r9b, %r9b
18000a950: cmovnel %r8d, %esi
18000a954: leaq 0x76660(%rip), %r8 ; 0x180080fbb
18000a95b: leaq 0x77c4f(%rip), %r9 ; 0x1800825b1
18000a962: cmovneq %r9, %r8
18000a966: cmovnel %edx, %ebx
18000a969: cmpq 0x9a0(%rbp), %rcx
18000a970: movl 0x9d0(%rbp), %ecx
18000a976: cmovel %r12d, %ecx
18000a97a: movl %ecx, 0x9d0(%rbp)
18000a980: movq %r9, %rcx
18000a983: leaq 0x71b83(%rip), %rdx ; 0x18007c50d
18000a98a: cmoveq %rdx, %rcx
18000a98e: movsd 0x100(%rbp), %xmm0
18000a996: unpcklps ...
18000a99d: subpd ...
18000a9a5: addsd ...
18000a9a9: mulsd ...
18000a9ae: leal (%rax,%rax), %edx
18000a9b1: movq %rcx, 0x38(%rsp)
... more callback setup omitted ...
18000a9d9: call 0x180007c00
18000a9de: movq %rdi, %r15
18000a9e1: incl %r12d
18000a9e4: movl 0x958(%rbp), %eax
18000a9ea: cmpl %eax, %r12d
18000a9ed: jge 0x18000aae2
18000a9f3: ... loop back to device query
```

This is a per-device tuning / optimization block. It mixes:

- `strlen` result in `%dl`
- `r15` value compared to the `gfx1200` magic word
- `0x164(%rbp)` (a property field likely related to queue / blocks / workgroup capacity)
- `0x100(%rbp)` / floating-point math values used for comparing performance or capability metadata

This is not the final reject code; it is a capability-tuning / optimization pass.

### Block E: `hipGetDevicePropertiesR0600` + follow-up validation

```asm
18000ab8f: xorl %esi,%esi
18000ab91: leaq -0x20(%rbp), %r12
18000ab95: movl $0x5c0, %r8d
18000ab9b: movq %r12, %rcx
18000ab9e: xorl %edx,%edx
18000aba0: call 0x180064a00
18000aba5: movq %r12, %rcx
18000aba8: movl %ebx, %edx
18000abaa: call 0x180065a20 ; hipGetDevicePropertiesR0600
18000abaf: movq %r12, %rcx
18000abb2: call 0x180064e20
18000abb7: leaq 0x90202(%rip), %rcx ; 0x18009adc0
18000abbe: movq %r12, %rdx
18000abc1: movq %rax, %r8
18000abc4: call 0x180035980
18000abc9: movl %ebx,%ecx
18000abcb: call 0x180065ae0
18000abd0: movl %eax, %r14d
18000abd3: testl %eax,%eax
```

This is a second property-fetch pass. It is structurally the same as the earlier loop, but now the code sets `%r14d` from the helper at `0x180065AE0` (which is not a GPU gate; it is an unrelated API call in the final binary). The immediate check is only `testl %eax,%eax` followed by a conditional branch to log/global-choice code.

### Block F: final architecture comparison and global rejection flag

```asm
18000ac4d: movl $0x31786667, %eax       ; 'gfx1'
18000ac52: xorl 0x468(%rbp), %eax
18000ac58: movzbl 0x46c(%rbp), %ecx
18000ac5f: xorl $0x31, %ecx
18000ac62: movabsq $-0x1000000000001, %rdi
18000ac6c: andq 0x468(%rbp), %rdi
18000ac73: orl %eax,%ecx
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
18000ac9d: orb %sil,%bl
18000aca0: movb $0x1, %al
18000aca2: movb %al, 0x90111(%rip)      ; 0x18009ADB9
18000aca8: je 0x18000acb6
18000acaa: leaq 0x7667f(%rip), %rcx     ; 0x180081330
18000acb1: call 0x180011460
```

This is the decisive block. The key facts are:

- `sil` is set if the name-length helper returned 7.
- `bl` is set if the architecture word equals `gfx1200`.
- `al` is set based on `r14d == 0`, but then is overwritten by `movb $0x1, %al` immediately before the state flag write.
- The state byte at `0x18009ADB9` is then assigned to `1` unconditionally in this block.
- `je` at `0x18000ACA8` decides whether to call a secondary logging helper, but does not decide whether the flag is written.

This is the exact location where the initialization state enters the rejection path.

## Every basic block that can reach `0x18000ACA2`

The relevant predecessor blocks are:

1. `0x18000AC75` branch path
   - the preceding compare `orl %eax,%ecx` / `jne 0x18000ac87` determines whether execution falls through to the `gfx1200` checks or to the `strlen` check.

2. The `0x18000AC87` block itself
   - this block sets `bl`, `al`, and then executes the `orb` / `movb` sequence.

3. The unconditional path after `0x18000AC87`
   - there is no other legitimate path to `0x18000ACA2` besides this block.

The immediate relevant branch-tree is:

```text
0x18000AC73: orl %eax, %ecx
0x18000AC75: jne 0x18000AC87
  -> falls through to 0x18000AC77 ... 0x18000AC83
  -> then continues to 0x18000AC87

0x18000AC97: testl %r14d,%r14d
0x18000AC9A: sete %al
0x18000AC9D: orb %sil,%bl
0x18000ACA0: movb $0x1,%al
0x18000ACA2: movb %al, [0x18009ADB9]
0x18000ACA8: je 0x18000ACB6
```

Therefore, every path that reaches the final validation block ends by storing `1` to the state flag.

## Branch conditions that determine whether execution reaches `0x18000ACA2`

The relevant conditions are:

1. `0x18000AA0E`: `testl %eax,%eax` after the HIP device-property call
   - if HIP fails, this skips the architecture validation and advances to the next device.

2. `0x18000AA42`: `jne 0x18000A8F0`
   - if the property-name prefix does not match the expected `gfx1...` pattern, it leaves the current device and continues the loop.

3. `0x18000AA50`: `cmpq $0x7, %rax; sete %dl`
   - if `strlen` of the architecture name is 7, `dl` becomes `1`.

4. `0x18000AA75` (in the block context above, the `jne` at `0x18000AC75` is the final branch): `orl %eax,%ecx; jne 0x18000AC87`
   - if the prefix check fails, it jumps into the `gfx1200` / name-length comparison block.

5. `0x18000ACA8`: `je 0x18000ACB6`
   - this branch is about whether a secondary helper is called; it is not what sets the rejection flag.

## Variables tracked through the branches

### `r14d`

- Set at `0x18000ABD0`: `movl %eax, %r14d` after a call to `0x180065AE0`.
- Later tested at `0x18000AC97`:

```asm
18000ac97: testl %r14d,%r14d
18000ac9a: sete %al
```

This means `r14d == 0` yields `al = 1`.

However, the next instruction immediately overwrites `al`:

```asm
18000aca0: movb $0x1, %al
```

So `r14d` is consulted, but it does not decide whether the global flag is set. It only affects the previous branch condition and the secondary logging choice.

### `r15`

- At `0x18000A8B9`: `leaq 0x468(%rbp), %r15`
- At `0x18000AA15`: `movq 0x468(%rbp), %r15`
- At `0x18000A924`: `cmpq %rax, %r15` compares the first 8 bytes of the architecture-string area against a special constant.
- This is the first part of the `gfx1200` / architecture string test.

### `rdi`

- `rdi` is reused as the string / word workspace around the architecture checks.
- At `0x18000AA12`: `movq %r15, %rdi`
- At `0x18000AA48`: `movq %rdi, %rcx; call 0x180064E20`
- At `0x18000AC87`: `cmpq %rax, %rdi` compares the name word to `0x30303231786667` (`gfx1200`)

This indicates that the code is reusing `rdi` as a temporary representation of the architecture name word in the device property struct.

### `r13d`

- `r13d` is a per-loop state value. It is updated at `0x18000A8FC` and again at `0x18000A905` using `cmovnel`.
- This appears to be a loop- or device-state boolean, not the final rejection criterion.

### `bl`

- `bl` is set by `sete %bl` at `0x18000AC94` after comparing `rdi` to the magic value for `gfx1200`.
- Then `orb %sil, %bl` combines it with the result of the length check.

Thus, `bl` is a logical OR of:

- `arch == gfx1200`, or
- `strlen == 7`

This is a mixed capability/optimization predicate, not a direct rejection predicate.

### `sil`

- `sil` is set by `sete %sil` at `0x18000AC83` after `strlen(...) == 7`.
- It is then ORed into `%bl` with `orb %sil, %bl`.

So `sil` is a string-length test result.

### `eax`

- `eax` is used repeatedly as a scratch register in the property validation checks.
- At `0x18000AC97`, `testl %r14d,%r14d; sete %al` sets `al` based on `r14d == 0`.
- Immediately afterward, `movb $0x1,%al` overwrites it.

This is decisive evidence that the final state writes are not controlled by the earlier `eax` result; the state byte is written to `1` regardless of the earlier `eax` condition.

## Architecture comparisons and their roles

| Address | Compared value / string | Branch target | Role |
|---|---|---|---|
| `0x18000AA2F` | `0x31786667` = `gfx1` | `0x18000AA42` if mismatch | Prefix validation |
| `0x18000AA50` | `strlen(...) == 7` | sets `%dl` | String length check |
| `0x18000A91A` | `0x30303231786667` = `gfx1200` | `sete %r9b` | Special-case architecture optimization |
| `0x18000AC62`..`0x18000AC75` | masked name word vs `gfx1` pattern | `jne 0x18000AC87` | Architecture prefix / name sanity check |
| `0x18000AC87` | `0x30303231786667` = `gfx1200` | `sete %bl` | Special-case arch match |
| `0x18000AC83` | `strlen == 7` | `sete %sil` | Allow-length condition |
| `0x18000AC9D` | `sil | bl` | `je 0x18000ACB6` | Final branch after `or` |

Important interpretation:

- These are not a simple allow-list or raw block-list check.
- They are a combination of:
  - architecture-prefix sanity checks,
  - string-length tests,
  - special-case checks for a specific family (`gfx1200`),
  - and then a global state flag assignment.

This is why the DLL can contain `gfx1032` in the architecture list and still reject the actual RX 6600 at runtime: the code is not using the architecture list as a guarantee of support; it is using the returned HIP property data as a runtime capability gate and then setting a rejection state even when the architecture name is known to be present in the fat binary.

## What the code expects from the HIP device-properties structure

The relevant call is:

```asm
18000aa09: call 0x180065a20 ; hipGetDevicePropertiesR0600
```

The device properties buffer is a local stack buffer at `%r14` (`leaq -0x20(%rbp), %r14`), and the code later reads fields from the stack-local struct using offsets such as:

- `0x468(%rbp)`
- `0x46c(%rbp)`
- `0x164(%rbp)`
- `0x100(%rbp)`
- `0x9a0(%rbp)`
- `0x9d0(%rbp)`
- `0x9a0(%rbp)` / `0x9a8(%rbp)` / `0x9b0(%rbp)` / `0x9b8(%rbp)`

These offsets are consistent with a `hipDeviceProp_t`-style structure or a custom local wrapper around it.

The reads that are clearly relevant are:

1. `0x468(%rbp)` and `0x46c(%rbp)`
   - this is the architecture-name / `char name[256]`-style region, or the first bytes of the architecture string in the device prop structure.
   - the code compares it to `gfx1...` and then to `gfx1200`.

2. `0x164(%rbp)`
   - likely a numeric field in the HIP properties object, such as a block count / CUs / threads-per-block-related value.
   - the code does `movl 0x164(%rbp), %eax; addl %eax, %r8d` and later uses it in the capability tuning calculation.
   - this is not obviously the final rejection determinant, but it is definitely a numeric device-property read.

3. `0x100(%rbp)`
   - the code computes a floating-point expression using `movsd 0x100(%rbp), %xmm0` and subtracts constants, which strongly suggests a property like a clock/ratio metric or an execution-time value.

4. `0x9a0(%rbp)` and related offsets
   - these are part of a local buffer or queue object rather than direct device-property numeric fields.

The architecture-name read is the most concrete and most relevant signal because the code explicitly compares the string to `gfx1` and to `gfx1200`.

## Exact property offsets inside HIP device properties

From HIP ABI knowledge, the `hipDeviceProp_t` / `hipDeviceProperties_t` structure includes fields such as:

- `char name[256];`
- `uuid_t uuid;`
- `char luid[8];`
- `unsigned int luidDeviceNodeMask;`
- `unsigned int pciBusID;`
- `unsigned int pciDeviceID;`
- `unsigned int pciDomainID;`
- `int multiProcessorCount;`
- `int maxThreadsPerMultiProcessor;`
- `int maxThreadsPerBlock;`
- `int warpSize;`
- `int clockRate;`
- `int memoryClockRate;`
- `int memoryBusWidth;`
- `int totalConstMem;`
- `int sharedMemPerBlock;`
- `int totalGlobalMem;`
- `int regsPerBlock;`
- `int major;`
- `int minor;`
- `int computeMode;`
- `int clockInstructionRate;`
- `int unifiedAddressing;`
- `int concurrentManagedAccess;`
- `int pageableMemoryAccess;`
- `int asyncEngineCount;`
- `int cooperativeLaunch;`
- `int cooperativeMultiDeviceLaunch;`
- `...`

The relevant offsets here are most likely near:

- `name` (string area) at the start of the property block
- `major/minor` or `arch` field around the later numeric area
- perhaps `multiProcessorCount`, `maxThreadsPerMultiProcessor`, or `warpSize`

The chain of checks does not isolate a single property with full certainty from the assembly alone, but the code clearly expects a string-form architecture name to be present and then uses a secondary numeric field / helper result before setting the rejection flag.

## Failure block `0x18000AD00` to `0x18000AD53`

This block is the final cleanup / failure marker before the unsupported message logic:

```asm
18000ad01: leaq -0x20(%rbp), %rsi
18000ad05: movl $0x5c0, %r8d
18000ad0b: movq %rsi, %rcx
18000ad0e: xorl %edx,%edx
18000ad10: call 0x180064a00
18000ad15: movq %rsi, %rcx
18000ad18: xorl %edx,%edx
18000ad1a: call 0x180065a20 ; hipGetDevicePropertiesR0600
18000ad22: call 0x180064e20
18000ad27: leaq 0x90092(%rip), %rcx ; 0x18009ADC0
18000ad2e: movq %rsi,%rdx
18000ad31: movq %rax,%r8
18000ad34: call 0x180035980
18000ad39: leaq 0x468(%rbp), %rdx
18000ad40: leaq 0x725e8(%rip), %rcx ; 0x18007D32F
18000ad47: call 0x180007c00
18000ad4c: movb $0x0, [0x18009ADB9]
18000ad53: cmpb $0x0, [0x18009ADB9]
```

This path is important because it clarifies two facts:

- `0x18009ADB9` is only a runtime state bit; it is explicitly cleared in a failure path.
- The unsupported message occurs later when the flag is seen as set, which means the flag is a state machine bit rather than a direct per-arch constant.

## Why the code can contain `gfx1032` in the architecture list while still rejecting the actual RX 6600 at runtime

This is the key point: the architecture list in `.rdata` is not itself a final allow-list for the currently connected device.

The binary contains the strings:

```text
gfx1201,gfx1200,gfx1100,gfx1101,gfx1102,gfx11-generic,gfx1032,gfx9-generic
```

But the runtime `hipGetDevicePropertiesR0600` result is then used in a separate validation block. That validation block:

- checks the returned architecture string for a prefix / length / family pattern,
- checks a secondary numeric/helper condition,
- and then writes `1` to `[0x18009ADB9]`.

This means the binary can include support for `gfx1032` in the fat binary and even in the string table, while still rejecting a specific device because the runtime property values for that user’s hardware do not satisfy the code’s later family / capability gate.

In other words:

- the embedded fat binary says “this build has a GFX1032 object”
- the runtime validation says “this device’s actual properties fail the later support check”
- the unsupported message is downstream of the later state bit, not the raw embedded object list

## What still needs to be traced to prove the exact field

The disassembly establishes the control flow and the rejection state, but the exact property member inside the HIP device-properties structure that flips the final branch is not yet isolated to a single field with absolute certainty.

The remaining untraced value is the specific field(s) backing:

- `0x468(%rbp)` / `0x46c(%rbp)`
- `0x164(%rbp)`
- and the result of `0x180065AE0` stored in `r14d`

The next necessary step is to dump the live `hipDeviceProp_t` returned by `hipGetDevicePropertiesR0600` for a real RX 6600 and correlate those offsets to the actual fields in the structure.

## ROOT CAUSE

The direct evidence shows that the rejection flag is written at `0x18000ACA2` (`mov byte ptr [0x18009ADB9], 1`) after the code has validated the HIP device property block and the architecture string in the returned `hipGetDevicePropertiesR0600` object. The path is entered after `0x18000AA09` calls `hipGetDevicePropertiesR0600`, `0x18000AA0E` checks `EAX`, and then the architecture/name checks at `0x18000AA2F`, `0x18000AA50`, `0x18000AC87`, and `0x18000AC97` drive the later state assignment.

The root cause is therefore not a missing `gfx1032` object; it is a later runtime property gate that sets the global unsupported flag once the returned HIP properties fail the block’s architecture / capability validation. The remaining missing proof is which exact `hipDeviceProp_t` member (most likely the architecture-name field and / or a numeric capability field) is causing the RX 6600 / gfx1032 case to satisfy the reject condition. The exact value or function still needing to be traced is the live `hipDeviceProp_t` field values returned by `hipGetDevicePropertiesR0600`, especially the fields behind `0x468(%rbp)`, `0x46c(%rbp)`, and `0x164(%rbp)`.
