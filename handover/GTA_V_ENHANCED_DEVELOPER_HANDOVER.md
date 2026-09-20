# Developer handover: first verifiable GTA V Enhanced neural frame on gfx1030

Prepared 20 September 2026 for the supervisor of **Sakushey/gfx1030-dlss-nr-research**. All proposed work below targets the RX 6900 XT / Navi 21 / `gfx1030`. The supplied Discord report concerns a different device, RX 6600 / `gfx1032`; its results are not transferable device qualification.

**Decision:** the project is not one DLL setting away from a demonstrable neural frame. There are specific, fixable build, parser, validation and evidence-reporting defects, followed by uncompleted physical kernel, complete-job and presentation milestones. A separate local experiment now supplies a useful starting point: one translated final-head kernel produced 32,768 correct output bytes across two workgroups on the RX 6900 XT. That result does not resolve the repository's stalled J3/SWIN path.

The proposed target is deliberately narrow: **capture one authentic GTA input packet, compute the complete selected network job correctly, apply its result with the correct image contract, and present and save one identifiable frame.** Neither playable frame rate nor a large perceptual improvement is established or promised. Correct execution enables an image-quality experiment; it does not predetermine its outcome.

## 1. Audit boundary and evidence rules

A fresh upstream clone was inspected at commit [`dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8`](https://github.com/Sakushey/gfx1030-dlss-nr-research/tree/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8), titled `Remove one-shot release bootstrap`. This is the inspected public state, not an assumption about unpublished developer files or subsequent commits.

The fresh checkout passed **44 host unittest tests** during this audit. The test output is included in `evidence/host-tests.txt`. No new GPU or GTA execution was performed for this handover. Earlier local GPU logs and original experimental source are included separately. GitHub issue comments were not retrieved and are not used as evidence.

Evidence labels used here:

| Label | Meaning |
|---|---|
| Public source | Directly inspected implementation in the pinned commit |
| Project-reported | The repository's own status of physical experiments, not independently rerun here |
| Local experiment | Source, hashes and retained logs from this workspace; not merged upstream |
| Supplied report | User-supplied Discord text/log from another machine; not an independently reproduced result |
| Proposed work | An implementation prescription or acceptance criterion, not a claim that code already exists |

Only established omissions and defects are described as findings. Suspected watchdog mechanisms, particular missing instructions, driver faults and dropped stores are **not** findings. An absent public artifact also does not prove that the author lacks it privately.

## 2. What is actually established

The pinned [STATUS.md](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/STATUS.md) reports that a one-workgroup translation reaches HIP but fails to finish before watchdog recovery. A midpoint experiment narrows the failure to the first approximately 55.6% of modeled execution. It explicitly does not identify the mechanism. Complete-job, D3D12/HIP and presented-frame milestones remain unreached. The bridge has host/mock evidence; that is not physical interoperability evidence.

The supplied GTA log reports FFX capture, shared inputs/output, asynchronous residual application and completed network-job counters on RX 6600. It also identifies a stub architecture path. Those messages establish what that executable logged, not correct neural output. In particular, `ready`, elapsed milliseconds and a nonzero pre-block buffer do not validate downstream network computation or prove presentation. The modified executable producing that log was not supplied; do not equate it with the installer inspected here.

The locally inspected installer contains eight GPU bundle entries plus an empty host entry. In its generic RDNA2 object, `_Z13k_align_probePh` immediately returns and `_Z12k_final_head10HeadParams` stores zero values. Thus unchanged probe bytes or zero final-head output have a concrete code-level explanation **if that object is selected**. This does not prove which bytes were selected by the other user's modified DLL. It does not prove every function in the generic object is a stub.

The local RX 6900 XT experiment instead translated the numerical GFX1100 final-head function. It obtained:

| Test | Dispatch | Compared output | Result |
|---|---|---:|---|
| Hidden-argument probe | 1 workgroup, 256 threads | Grid/group ABI fields | Expected fields observed |
| Constant finite input and weights | 1 × 256 | 16,384 bytes | Exact match |
| Patterned signed input and weights | 1 × 256 | 16,384 bytes | Exact match, 87 distinct byte values |
| Patterned signed input and weights | 2 × 256 | 32,768 bytes | Exact match, 88 distinct byte values |

The patterned CPU reference computes independent integer matrix multiplication and E4M3 quantization. Allocation guards remained intact and input/weight writes were checked. Inputs were small dyadic values with exactly representable intermediate values over this test range. There was **no native RDNA3 reference device**, no authentic network weights/input packet, no whole-network run and no presented image. These limitations must travel with the result.

Identity anchors:

```text
Input installer SHA-256:
cf7ada1486b499700a84846b342ca2b1defdb4db622843f812151f255f2ad63c

Extracted GFX1100 ELF SHA-256:
51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7

Passing isolated final-head code object SHA-256:
8b0243689bf2a400459d9d988b8dbad361c614fa563538ff1ec12a96c8bac7c3
```

These are three different artifacts. None is the full-bundle hash expected by the public bridge.

## 3. Proven implementation gaps, with exact repairs

### F1. The documented GPU smoke-test command resolves a nonexistent source path

**Public source:** [`scripts/run_test.ps1:4`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/scripts/run_test.ps1#L4) joins the script directory to `soft_wmma_test.cpp`. The file is in `tests/`, not `scripts/`. Its explicit existence check therefore fails in the published layout.

**Repair:** resolve repository root from the script directory; select `tests/soft_wmma_test.cpp`; write artifacts under a declared build directory. Discover MSVC with `vswhere` and the C++ component requirement, or accept an explicitly supplied toolchain. Do not assume the `MSFT_VSInstance` CIM class exists. Make HIP root configurable while recording the selected version and absolute backend path. Provide `-BuildOnly` so path/compiler failures can be resolved without launching a kernel.

**Acceptance:** a clean checkout resolves every source dependency, builds from a working directory outside the repository, and reports missing prerequisites before GPU work. The existing local changes to the smoke launcher are useful input, not automatically a complete upstream patch.

### F2. The smoke test and benchmark can accept NaN output

**Public source:** [`tests/soft_wmma_test.cpp:163`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/tests/soft_wmma_test.cpp#L163) updates a maximum only when `abs(got-ref) > max_abs`. NaN makes that comparison false. If all differences are NaN, the maximum remains zero and the final threshold passes. [`tests/soft_wmma_bench.cpp:95`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/tests/soft_wmma_bench.cpp#L95) has the corresponding `std::max` problem.

**Repair:** use a shared comparator that checks expected length, every expected output, and finiteness before tolerance arithmetic. For these finite fixtures, any NaN or infinity is failure. Initialize outputs to a sentinel that cannot equal the expected fixture and check guards independently. Retain mismatching indices and actual/expected values.

**Acceptance:** host-only negative controls reject an injected NaN, infinity, untouched sentinel, single wrong element and corrupted guard. Passing synchronization or a small finite subset must never satisfy the whole-output check. For future deliberately nonfinite fixtures, implement explicit class/sign rules rather than silently reusing this finite comparator.

### F3. The dense smoke test does not verify native WMMA register ownership

**Public source:** [`tests/soft_wmma_test.cpp:35`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/tests/soft_wmma_test.cpp#L35) claims native C-fragment ownership, while its implementation assigns `row = 8*(lane>>4)+r`. This can compute a correct dense matrix because its stores use the same custom layout. It does not establish equivalence at a native WMMA instruction boundary.

**Repair:** retain the dense test with an accurate description and add a register-fragment test. For the specific FP16-input/FP32-output wave32 form used by the local replacement, reconstruct native A/B fragments, including replicated half-waves, and compare each D register independently. The local experiment uses `D[lane][r] = C[2*r + (lane>>4)][lane & 15]`. Validate that contract against the instruction form, not against the existing smoke comments. AMD's [WMMA explanation](https://gpuopen.com/learn/wmma_on_rdna3/) describes fragment distribution and input replication.

**Acceptance:** identity, one-hot, signed nonsymmetric and nonzero-accumulator cases catch row/column swaps, half-wave mistakes and lost accumulators. Keep full-EXEC and supported-wave assumptions explicit. Do not claim arbitrary EXEC-mask, alternate WMMA form or rounding-edge equivalence from this test.

### F4. The bundle parser uses the wrong structural contract

**Public source:** [`registry_ident.h:33`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/src/bridge/registry_ident.h#L33) calls the 64-bit value after the 24-byte magic a version and requires five. [`registry_ident.cpp:134`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/src/bridge/registry_ident.cpp#L134) then reads entries until an all-zero triple rather than using that value as a count.

The [Clang binary bundle format](https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout) defines that field as the **number of bundle entries**. Reading the supplied installer at bundle offset `0xE3200` gives nine. Consequently, this input fails the current check before architecture matching or hashing. The four expected GPU architectures in the header also describe a different input than the supplied eight-architecture bundle. This is a verified compatibility mismatch, independent of any GPU execution.

**Repair:**

1. Separate parsing from supported-payload policy. Parse a byte span with a real length, verify magic, read `entry_count`, and iterate exactly that count.
2. Validate each descriptor and identifier against the actual span, with overflow-safe subtraction checks; validate payload ranges, duplicate IDs and permitted overlap. A 16 MiB maximum is a cap, not proof that a pointer has 16 MiB of readable storage. Derive a readable owner/span at the runtime boundary or decline substitution when one cannot be established.
3. Do not require a zero terminator or infer table extent from padding. Accept structurally valid tables independently of whether their payload is supported.
4. Apply a separate allowlisted manifest identifying the original bundle hash, exact target IDs, source ELF hashes, model/runtime profile and replacement bundle hash. Generate constants from that manifest.
5. Keep unknown applications' registrations unchanged. For the selected test profile, report an explicit identity mismatch and keep neural execution disarmed instead of allowing a misleading “enabled” status.

**Acceptance:** synthetic host-only bundles with one, five and nine entries parse; valid nonzero bytes immediately after the table do not become extra descriptors. Truncated tables, truncated payloads, huge counts, duplicate IDs, overflowed spans and changed hashes fail predictably. A nine-entry fixture must not be reported as “version mismatch.” Production substitution still requires the exact approved identity. Do not merely change `5` to `9` or disable hashing.

### F5. There is no complete public, clean-checkout build recipe for the replacement payload

**Public source:** [`src/bridge/README.md:22`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/src/bridge/README.md#L22) explicitly requires the intentionally unpublished generated `bridge_gfx1030_fatbin.h`. [`p14d_rebuild.py:1`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/src/isa/p14d_rebuild.py#L1) is descriptor-only normalization and expects `phase9_final_module/asm` and `phase5_exact_fragment/gfx1100_code_object.o`; neither exists in the fresh checkout. It is not a complete initial translator merely because its name contains “rebuild.”

**Repair:** publish original generation tooling and a dependency graph, while continuing to exclude proprietary inputs and generated payload bytes. A single build-only command should take locally supplied input paths plus a profile, extract and hash inputs, generate disassembly/metadata, translate the selected symbol closure, assemble/link, validate descriptors/relocations, pack the replacement and emit the header and manifest. Each step must name its producer and exact tool version. Reject missing inputs before compiling the bridge. Separate a synthetic mock fixture from a real payload; never ship a production placeholder as if it were qualified.

**Acceptance:** a second developer can regenerate the same qualified inputs/outputs from the documented local inputs without legacy private phase directories. Reproducibility differences must be explained rather than hidden behind a filename. Compilation and physical qualification remain distinct results.

### F6. Native WMMA execution in the emulator remains a value stub

**Public source:** [`src/emulator/emu.py:1250`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/src/emulator/emu.py#L1250) iterates active lanes and leaves accumulators unchanged. The older oracle-base copy also has a stub. This does not invalidate every scalar/control check; it limits what these particular native-WMMA paths can prove numerically.

**Repair:** make control-only emulation an explicit mode whose reports cannot receive a numerical-validation verdict. A numerical mode must fail on an unsupported/value-stub instruction, or implement the relevant WMMA fragment operation and its documented arithmetic behavior. Use an independently expressed matrix reference and, where available, native hardware traces. Do not compare two models inheriting the same stub and label agreement independent numerical evidence.

**Acceptance:** executing the instruction with a known nonzero product changes the expected accumulators; a stub triggers `UNSUPPORTED_NUMERIC` rather than success. Emit the dynamically encountered unsupported forms and coverage per kernel.

### F7. The readiness script cannot advance from newly produced evidence

**Public source:** [`tools/k_readiness.py:123`](https://github.com/Sakushey/gfx1030-dlss-nr-research/blob/dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8/tools/k_readiness.py#L123) assigns F/G/J tracks `NOT_BUILT` and then requires them not to equal `NOT_BUILT` for J3 readiness. It also hardcodes several status outputs. This is a frozen phase report, not an evolving release gate.

**Repair:** preserve historical evidence, but implement an evidence-derived qualification manifest. Key results by original/replacement hashes, device, runtime, fixture, shape, launch configuration and test version. Give each required stage a typed result such as `NOT_RUN`, `PASS`, `FAIL` or `UNSUPPORTED`; verify referenced artifacts. Track host, isolated physical kernel, authentic dispatch, complete job, transport and presentation separately.

**Acceptance:** host tests prove that missing, failed, stale and wrong-profile records cannot produce readiness, and that a complete matching synthetic evidence set can. A final-head record must not advance J3, another shape, gfx1032, or presentation readiness. Do not “repair” this by replacing literal failure strings with literal success.

### F8. The physical J3/SWIN blocker is unresolved, not diagnosed

**Project-reported:** the status and roadmap establish non-completion and the checkpoint interval. They do not prove a barrier, wait counter, descriptor or scheduling root cause.

**Necessary work:** retain the failing artifact identity and input fixture; produce a new bounded experiment distinguishing specific hypotheses inside the narrowed interval. Verify the relationship between modeled instruction position and a real control-flow checkpoint. Use only checkpoints that all participating waves can reach consistently; an early exit in one wave before a shared barrier can manufacture a hang. Include small independent probes for any instruction/ABI contract implicated by evidence.

**Acceptance:** a numerically meaningful one-workgroup result, checked output/guards and completion evidence from the exact tested artifact, followed by the selected authentic dispatch. A timeout reduction, returning synchronization call, or corrected descriptor alone is not completion. Do not assert that the final-head table fix below explains this different kernel's failure.

### F9. The local final-head port is not a production module or full-network translator

**Local source:** `rdna2/translate_final_head.py` explicitly translates one hash-matched symbol, uses a fixed register budget and descriptor, consumes a saved metadata dump and materializes copied constant data in its standalone module. It does not provide the original full registration/global-symbol contract.

Its first numerical run returned all zeros because the extracted `g_e4m3_lut` contained zero initialization data and the isolated harness had not reproduced runtime initialization. Populating that table fixed this experiment. **This is not evidence that the original application fails to initialize its table.** The public bridge already forwards `__hipRegisterVar` and `hipMemcpyToSymbol`; those mechanisms must be preserved and verified.

**Repair:** promote the local port as a versioned regression fixture, then replace its single-kernel assumptions with per-kernel metadata, relocation and global-initialization handling. Preserve externally registered variable names, sizes, storage, lifetime and copy offsets. Reproduce each initialization before dependent kernels launch. Do not bake this one fixture's constant table into every translated module or silently redirect symbol writes elsewhere.

**Acceptance:** the final-head test still passes after packaging through the real registration path, including an initialization/readback check. Removing the lookup-table initialization must reliably make the controlled nonzero fixture fail. Qualification of other network functions remains separate.

### F10. A complete authentic job and presented frame are not yet qualified

**Project-reported:** the pinned status explicitly leaves these milestones unreached. The public HIP forwarding code is not a complete GTA hook/composition implementation. The supplied external log shows another host integration exists, but its source and exact modified binary were not available for this audit.

**Necessary work:** establish the selected host/runtime integration, capture an authentic job contract, qualify its entire executed dependency closure, and verify transport and composition. The next sections specify this work. It is an evidence/integration gap; do not claim to have proven that every component is entirely unwritten in the maintainer's private workspace.

## 4. Implementation sequence for the first frame

### Stage A — Freeze one profile and make the evidence reproducible

Supervisor instruction: **choose one explicit source profile before asking the agent to generalize anything.** The repository's historical bundle and the supplied installer are different profiles. Prefer the supplied hash-matched input if adopting the local final-head evidence; otherwise reproduce the experiment against the maintainer's chosen input and label it a new result.

Deliver a profile containing:

- Host runtime binary hash/build, original bundle hash and its counted entries, source ELF hash, weight/model identity, replacement module/bundle hashes.
- RX 6900 XT `gfx1030`, exact driver, HIP runtime and compiler versions, MSVC/SDK versions, wave size and selected device identity.
- GTA executable version/hash, FFX/OptiScaler configuration if actually used, selected route, input/output dimensions and temporal/history policy.
- Explicit required kernels and globals once capture establishes them; initially mark these unknown, not guessed.

Fix F1–F7 first. The build should remain host-only by default and have a distinct physical test command. Maintain the repository's publication manifest properly when changing tracked source; do not delete provenance checks to make tests green.

### Stage B — Adopt the passing final-head fixture as the first physical reference point

The support directory includes original scripts and logs, not generated proprietary ELF/assembly/payloads. Import the useful logic into the repository's own layout and replace hardcoded `research/` and `analysis/` assumptions with explicit inputs.

The existing local reproduction sequence, **after locally regenerating its inputs**, is:

```powershell
python rdna2/translate_final_head.py --abi-probe
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode abi
python rdna2/translate_final_head.py
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode patterned
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode multigroup
```

These are commands from the local experiment, **not clean-checkout upstream commands**. Its generator needs the hash-matched extracted ELF, generated `analysis/gfx1100-notes.txt`, ROCm 6.4 tools and the repository's `p14d_kd` parser. The support code exposes these requirements; Stage A must make their production deterministic. Do not download a mystery replacement DLL to satisfy them.

The local code illustrates three necessary translation details worth retaining: separate scratch registers for WMMA lowering, temporary results preserving simultaneous reads when splitting dual operations, and explicit relocation of branch/PC-relative references. Its fixed 96-VGPR/8,192-byte-LDS descriptor and entry SGPR remap are facts about this artifact, **not defaults for the remaining kernels**.

Add authentic-range and boundary fixtures before treating it as qualified for a real job. Small exact matrices do not cover FP16 overflow, denormals, NaNs, all FP8 rounding ties or accumulated error through a network. Document numerical tolerances from an independent reference instead of adjusting them until a test passes.

### Stage C — Capture the real job contract before selecting the remaining kernel work

Supervisor instruction: **obtain access to the chosen host integration's source or an observable runtime boundary.** Public research code alone does not supply the complete model scheduling, tensor preparation and GTA residual application contract. If source is available, instrument it directly. Otherwise instrument the existing registration/launch/memory boundary for the precisely identified binary. Report any unobservable contract as a blocker rather than inventing it.

A capture mode should collect a single immutable packet while neural execution remains disarmed on unqualified hardware. Capture the actual FFX input route, not merely the swapchain backbuffer: the supplied log distinguishes those routes. Job scheduling metadata may be recorded with launches suppressed, but **valid intermediate tensors and numerical golden output require a functioning implementation or independent reference**. A mock run cannot manufacture them.

Record at least:

| Area | Required information |
|---|---|
| Frame | Capture/job ID, dimensions, crop/padding, jitter, exposure, history/reset state, input hashes |
| Resources | Pixel/tensor formats, pitches, planes/subresources, strides, signedness, packing and coordinate conventions |
| Registration | Full mangled function names, globals and sizes, source module identity, function-pointer-to-symbol mapping |
| Memory | Stable allocation IDs, byte sizes, pointer arguments as allocation ID plus offset, lifetimes and initialization writes |
| Launches | Ordered dispatch list, symbol, grid/block, dynamic LDS, argument schema/bytes, stream and predecessor dependencies |
| Constants | Weights/scales and LUT initialization, copy offsets, source hashes and initialization order |
| Output | Whether output is a residual or replacement image, scale, color domain, dimensions, format, clamping and exposure rules |

Do not serialize raw process addresses as replayable pointers. Record relocation fields explicitly and rebuild addresses when replay allocates buffers. Decode hidden kernargs and descriptor-implied inputs for each dispatch, not only visible C++ struct fields.

The capture determines the mandatory kernel set. The local static census finds 203 WMMA sites across 24 of 35 disassembled functions; those numbers are **not 35 launched kernels**, and they are not a job dependency graph. Helpers and indirect control flow matter. Conversely, it is unnecessary to qualify every unused variant before one fixed-profile frame, provided unsupported routes are explicitly rejected.

**Deliverables:** `job.json`, original input tensors, initialized constants or locally reproducible references to them, dispatch/dependency inventory and a source-to-output dependency map. Store private assets locally and publish only original tooling and nonproprietary metadata.

### Stage D — Close the selected kernel dependency set and resolve liveness

Generate a per-symbol ledger from Stage C with translation status, descriptor, register/scratch/LDS requirements, called helpers, global references, instruction coverage, fixture and physical result. Use full symbol identity, not merely substrings from the older 33-entry policy table.

For every function actually required by this profile:

1. Decode the exact instruction forms and preserve data/control side effects, including EXEC/VCC/SCC, carry, aliases and packed operand selectors. Refuse unknown forms; do not NOP them or emit silent zero paths.
2. Expand only the WMMA forms with a tested fragment/arithmetic contract. Determine scratch registers from liveness and allocation; preserve caller/callee requirements and all uses of the original accumulator.
3. Reassemble for `gfx1030`. Resolve direct branches, PC-relative constants, helper calls and indirect targets; verify reachable instruction decoding after linking.
4. Generate each descriptor and metadata together, preserving or deliberately adapting dispatch-pointer requirements, workgroup-ID placement, hidden args, private segment/scratch, LDS and wave size. Prove any adaptation with a targeted probe.
5. Run the smallest meaningful checked fixture before its authentic shape. For cooperative kernels, retain the necessary participating waves/workgroups: arbitrarily reducing launch dimensions can break the algorithm.
6. For the reported J3/SWIN failure, use a bounded, hypothesis-specific checkpoint plan. Record one artifact per change and the exact last completed checkpoint. Compare relevant memory/flag/barrier dependencies and independent instruction probes; do not treat the final-head success as J3 evidence.
7. Qualify the authentic dimensions and participating workgroups selected by Stage C, including boundary tiles and inter-dispatch consumers.

A host timeout is an observation limit, not a guaranteed means of cancelling a stuck GPU kernel. Do not change watchdog settings or run automatic reset/retry loops. After a timeout/device loss, stop the sequence and preserve evidence; resume only with a distinct reviewed experiment and a usable device.

**Exit:** every required symbol and dispatch in the selected profile has matching physical evidence and a numerical check. Merely completing a dispatch is insufficient. Native supported-GPU outputs would strengthen numerical validation; if unavailable, state which independent operator/model reference is used and its limitations.

### Stage E — Build and validate a complete offline job

Implement a standalone replay runner before debugging inference inside GTA. Reconstruct allocations, constants, arguments and dependencies from the frozen packet, run exactly one job, and save named intermediate boundaries and the final tensor. No game DLL or D3D12 sharing is necessary for this phase.

Choose reference comparisons per operator/tensor before running. Exact comparison is appropriate for copies and exact fixtures; floating network tensors may need justified absolute/relative tolerances, finite checks and error distributions. Compare independently computed or known-good values at meaningful boundaries. A checksum proves repeatability/identity, not mathematical correctness.

Check that inputs/weights remain unmodified where required, every required output region is written, buffers stay within bounds, and output belongs to this job rather than a previous invocation. Completion markers must be observed after the relevant work is complete; a single wave reaching a marker cannot stand in for all workgroups. Pair stream completion with complete expected-output coverage and guards.

Include an input-dependency test with a change whose expected effect is known from the chosen reference. Nonzero output alone and arbitrary visual noise are not sufficient. Run the frozen packet again in a fresh process to expose dependence on stale initialization. If stage comparisons diverge, stop at the first bad boundary rather than proceeding to presentation.

**Exit:** one authentic complete job on RX 6900 XT, with verified output and enough intermediate evidence to localize discrepancies. This is still not a rendered game frame.

### Stage F — Package the qualified job through the actual bridge/runtime

Audit the exact chosen host binary's HIP imports and declarations against the backend. The public `exports.def` describes an earlier audited set; its existence does not prove ABI compatibility with the supplied v0.3.1/HIP 7.2 environment. Equally, absence of a particular export is not automatically a defect unless the selected path needs it.

Use the existing ABI probes and mock backend to verify called signatures and structure layouts, then perform a physical registration test with the qualified module. Preserve full function and variable registrations and `hipMemcpyToSymbol` behavior. Log selected source/replacement hashes and actual backend device. Confirm that the compiled code object, not an unchanged generic stub, is loaded for each launch.

Maintain both the original identity gate and explicit launch gate. The current default-off `DLSSNR_GFX1030_ALLOW_KERNEL_LAUNCH` behavior is an intentional safeguard, not a bug. Enable a **specific qualified profile** only after its evidence matches. If the host's architecture policy prevents that profile, add a narrow source-aware compatibility decision with an explicit profile check. Blind binary patches or spoofing `gfx1100` do not implement missing RDNA2 computation.

Replay Stage E through this real registration path and compare the same tensor outputs. This detects mistakes hidden by standalone `hipModuleLoad` success, particularly globals, symbol mapping and ABI differences.

### Stage G — Verify transport separately, then carry one neural result to GTA

Do not make zero-copy interoperability a prerequisite for the first frame if it is the remaining blocker. Implement a deliberately slow **CPU-staged diagnostic route** using the existing host integration, then optimize only after correct presentation. This is a proposed fallback, not a claim that the external application's current zero-copy route is faulty.

The diagnostic flow is:

```text
GTA/FFX input resources
  -> fenced D3D12 readback and immutable frame packet
  -> owned HIP allocations and one qualified complete job
  -> checked CPU output
  -> D3D12 upload and verified output/residual texture
  -> existing, contract-correct composition
  -> identified presented frame + saved captures
```

Use `GetCopyableFootprints` and `CopyTextureRegion` for supported texture transfer layouts, copying rows using returned pitches rather than assuming tightly packed GPU textures. Fence completion must precede CPU consumption; mapping a readback resource does not itself synchronize GPU work. Retain resources and command allocators until their work completes. Restore states required by the surrounding integration. These requirements follow Microsoft's [readback guidance](https://learn.microsoft.com/en-us/windows/win32/direct3d12/readback-data-using-heaps), [copy-footprint API](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getcopyablefootprints) and [texture upload guidance](https://learn.microsoft.com/en-us/windows/win32/direct3d12/upload-and-readback-of-texture-data).

Depth/motion/exposure resources require their actual format and sampling conventions; do not reinterpret every plane as RGBA. Resolve or convert only according to the captured host contract. Verify the D3D12 adapter and HIP device identities, not just device index zero. Stage this off the critical Present path and provide a finite CPU wait/cancellation policy; do not put a long spin loop into Present to wait for unqualified inference.

First test transport with deterministic identity and asymmetric diagnostic patterns. Label these **transport tests**, never neural frames. Then send the already verified neural tensor through exactly the same route, checking its readback/hash before composition.

If retaining zero-copy, separately establish supported external-memory/fence handle types, ownership, lifetimes and synchronization on the exact Windows runtime/device combination. AMD's [external-resource API documentation](https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.1/doxygen/html/group___external.html) describes relevant interfaces; it is not proof that a particular D3D12/HIP 6.4 combination works. Add exports/adapters only when the actual route requires them.

### Stage H — Compose and present one identifiable frozen frame

The supplied log says `residual on`, tonemapping and exposure handling are active. It does not supply the exact math. Obtain that math from the selected host implementation or a valid reference. Do not assume the network output is standalone RGB or simply add it to the swapchain image.

For a manually testable first frame:

1. Launch the selected GTA V Enhanced build in a controlled single-player scene, with the exact captured FFX route/settings. Keep normal rendering available as the baseline.
2. Arm one capture by explicit action. Save its ID and all required resources. Use an existing supported history-reset/history-off path if available; otherwise supply the real required history. Do not fabricate zero history without checking the contract.
3. Preserve a frozen baseline image and the corresponding composition inputs. A paused camera alone may not freeze exposure, jitter or temporal resources. The result must be applied to the same captured inputs, or clearly shown as a frozen diagnostic image.
4. Run the already qualified job and verify output before enabling application. Apply the real residual/replacement convention, exposure, color conversion, crop and output dimensions from Stage C.
5. Present that composed image through the game's controlled diagnostic path and identify it with its frame/job ID. Capture the composed target before UI and the displayed frame after presentation. A debug window displaying a tensor is useful but does not meet the in-game presented-frame criterion.
6. On failure, retain the baseline and show a precise stage failure. Prevent automatic inference resubmission. Restore normal rendering and release resources only after their work completes.

**The first-frame evidence package must contain:** the pinned source and binary/profile hashes; selected device/runtime; immutable input packet IDs/hashes; executed dispatch list; complete-job checks; final raw tensor; baseline image; composed pre-UI image; presented capture; and frame/job association. Include a pixel-difference image for inspection, while stating that visible difference alone does not prove enhancement or correctness.

## 5. Supervisor's concrete task split and review gates

Use the accompanying `CODING_AGENT_BRIEF.md` as the initial instruction. Review the following deliverables in order; Stage C metadata discovery can proceed while bounded kernel work is being prepared, but no later stage inherits an earlier stage's pass without evidence.

| Task | Required deliverable | Gate before proceeding |
|---|---|---|
| 1. Repair evidence/build foundation | F1–F7 fixes, synthetic regression tests, input manifest and build-only entry point | Fresh host tests and reproducible dependency resolution |
| 2. Import final-head reference | Original source integration, regenerated local inputs, paired negative controls and logs | Same controlled numerical results under the selected profile |
| 3. Freeze a real job | Capture schema/tooling, input packet, initialization and dispatch dependency map | No unknown required pointer/layout/scheduling contract |
| 4. Qualify selected kernels | Symbol ledger, J3 diagnosis evidence, per-kernel/authentic-dispatch checks | Every executed dependency has matching physical/numerical evidence |
| 5. Replay full job | Standalone one-shot runner and intermediate/final comparisons | Authentic complete output passes independent checks |
| 6. Integrate registration | Counted bundle parser, exact identity, ABI/global-registration tests | Same full-job output through the production registration path |
| 7. Transport and first presentation | Staging or proven shared-memory route, composition contract, one-shot GTA diagnostic | Saved, identified in-game frame backed by complete-job evidence |

Do not ask the agent to “make it work and report success.” Ask it to deliver each gate's raw artifacts, the precise unsupported cases, and the next smallest falsifiable experiment when a gate fails. The major remaining work is executable kernel coverage and authentic integration, not renaming a DLL or changing a readiness flag.

## 6. Inputs the supervisor must supply or deliberately reconstruct

The audit cannot establish these unavailable private details, and they must not be guessed:

- The exact host integration chosen for the first frame, and source access or a sufficiently observable boundary for its scheduler and composition contract.
- Its matching model weights, metadata and required initialization, supplied locally.
- Authentic frame/job inputs and a valid reference for required intermediate/final outputs, obtained from a working implementation or independently implemented math.
- The actual failing J3 artifact and fixture if continuing the upstream watchdog investigation. A percentage in a status report is insufficient to reproduce it.
- A GTA V Enhanced test installation and a machine/session on which the bounded physical tests and diagnostic integration can run. This audit did not establish a local GTA installation or perform any game test.

When one of these is unavailable, continue independent host/build/capture work and name the blocked stage exactly. Do not replace missing mathematical or host contracts with plausible-looking tensor output.

## 7. Package contents and reuse

`support/rdna2/` contains the local experiment's original sources and retained logs. `support/inspect_bundles.py` extracts locally supplied inputs without running the installer. `support/rdna2/FINAL_HEAD_RESULTS.md` gives detailed experimental context. `evidence/inspection.json` records selected audit observations and identities. `evidence/package-files.json` hashes the packaged files.

Generated vendor ELF/code objects, their disassembly, embedded headers, model weights, installer/runtime DLLs and game files are not included. The support files are research inputs for integration, not a drop-in game build. The public repository's code and license must be obtained from its pinned checkout. No GitHub issue, message or pull request was posted as part of this handover.
