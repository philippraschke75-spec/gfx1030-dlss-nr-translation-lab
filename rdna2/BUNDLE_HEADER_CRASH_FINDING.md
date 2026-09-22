# Reproduced HIP initialization failure, 2026-09-21

Deployment update: the corrected copy is now active at the existing runtime
path, preserving its filename, directory, settings and resource lookup location.
The original is preserved alongside it as version.before-bundle-header-repair.dll.
The launcher manifest now pins c206fb29a1265200b12dcb46b6123154fc549506ba4849c76b0742e5d7fe2979;
full Vortex CheckOnly preflight passes. No additional game-folder changes were
needed because the installed bootstrap already loads this absolute path.
Capture-only launch blocking and exception tracing remain active.
An in-game test of the repaired runtime is still pending. The investigation
and pre-deployment status below are retained as history.

The active diagnostic runtime is the imported version.dll with SHA256
fa616204dd68521217875fb41835d3189c89e7c87c3050e069df16bafc458a05.
Its bundle begins at file offset 0x9b600. It declares nine entries.
After the gfx1032 entry the next descriptor should begin at bundle offset 142,
but eight zero bytes precede the actual descriptor at offset 150. Parsing at
142 yields offset=0, size=569344, name_length=1172152. This is a malformed
descriptor stream. The reason those bytes were introduced is not established.

## Controlled reproduction

capture_registration_probe.cpp loads the capture-only bridge, registers the
extracted original bundle, and calls hipGetDeviceCount. It does not load the
research runtime, hook the game, allocate GPU buffers, or launch kernels.
The original bundle produces a caught std::exception with message
`bad allocation`. Its AMD backend stack offsets match the game's observed
exception chain (0x8388ff, 0x837e77, 0x8368b5, 0x3539, 0x1605f,
0x8d088, 0x15cd9, 0x2a4701, 0x43323, 0x5998b).

repair_capture_bundle.py compacts only the remaining descriptor header by
eight bytes, retaining all payload offsets and byte-for-byte identical payload
data. It pins the input SHA256 and checks all nine repaired descriptors.
AMD's installed clang-offload-bundler --type=o --list accepts the repaired
bundle and lists all nine targets. The same registration probe then returns
hipGetDeviceCount rc=0, count=1.

The repaired DLL copy is build/registration-probe/version-header-repaired.dll,
SHA256 c206fb29a1265200b12dcb46b6123154fc549506ba4849c76b0742e5d7fe2979.
The original remains unchanged. The bootstrap still points at the original:
the repaired copy has NOT been deployed or tested in Cyberpunk.

## Limits and next step

This establishes a reproducible malformed-bundle initialization defect and a
successful isolated repair. It does not prove that every in-game failure is
fixed. Next stage the repaired copy with the intended settings location, update
the bootstrap's fixed runtime path and manifest, rebuild/verify it, then perform
one capture-only game attempt. Do not rerun the current launcher unchanged.
The bundle still names gfx1032 rather than the user's gfx1030; this header repair
does not provide the RX6900XT kernel translation or authorize kernel execution.
All research launches must remain blocked during the next capture attempt.
