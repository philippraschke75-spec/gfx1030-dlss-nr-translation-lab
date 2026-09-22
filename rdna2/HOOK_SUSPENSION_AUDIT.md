# Hook setup suspension/allocation audit

Inspected runtime SHA256:
`c206fb29a1265200b12dcb46b6123154fc549506ba4849c76b0742e5d7fe2979`.
Evidence: `build/closed-20260921-1845/game-stacks.txt` and
`thread-enumeration-disassembly.txt`. Addresses below are RVAs.

## Confirmed facts

* IAT 0x8da28 resolves to KERNEL32!OpenThread.
* IAT 0x8daf8 resolves to KERNEL32!SuspendThread; thunk 0x65c10 jumps there.
* IAT 0x8d988 resolves to KERNEL32!HeapAlloc.
* Thread-enumeration loop calls helper 0x36370 at 0x9d24.
* Helper 0x36370 allocates a 16-byte bookkeeping node at 0x363a5, then
  calls SuspendThread via 0x65c10 at 0x36413 and links its bookkeeping node.
* After helper return, the caller appends the handle to a container. If full,
  it calls container growth at 0x9d48, returning at 0x9d4d. Growth calls the
  allocator at 0x2130c, returning at 0x21311. The allocator reaches HeapAlloc
  at 0x4e81e, returning at 0x4e824.
* The captured live worker (TID 9380) is waiting inside ntdll with exactly
  those container/allocator return addresses on its stack. Many peer threads
  already have suspension count 1 before inspection.

Thus the implementation performs potentially blocking allocation after it has
suspended other threads. Reserving only the caller's container is insufficient:
the helper allocates another node for every thread, including after earlier
threads have already been suspended. This is a concrete deadlock hazard and
matches the observed blocked path. The identity of the heap-lock owner is not
established; do not present a fully proven wait cycle or a tested fix.

## Required implementation change

Obtain the source/build inputs corresponding to this runtime's hook transaction.
The inspected workspace sources do not contain this implementation. Do not
patch away SuspendThread or blindly resume peers during a hook transaction.

Separate preparation from suspension:

1. Enumerate candidates and open handles before suspending anyone. Allocate
   the complete handle list, transaction nodes, relocation/trampoline storage,
   saved contexts, and rollback state before suspension. Resolve imported calls
   and any lazy initialization needed for the critical phase beforehand.
2. Change the helper as well as its caller to consume preallocated records.
   Do not allocate/free containers or bookkeeping nodes, write buffered logs,
   invoke callbacks, or acquire locks that a suspended peer could own in the
   suspended interval. Audit commit and abort paths, not just enrollment.
3. Record only successful suspensions and balance each exactly once on success
   and failure; preserve pre-existing suspend counts. Handle disappearing
   threads and failed context operations with a defined rollback.
4. Account for thread creation between enumeration and patching using the hook
   implementation's supported synchronization strategy. A fixed-size array
   alone does not solve this race.
5. Resume peers before cleanup/freeing storage. Keep graphics patch visibility,
   instruction-pointer adjustment and instruction-cache handling correct.

Test outside Cyberpunk using a worker holding an allocation-related lock while
another thread initiates hook setup. Instrument the suspended interval to fail
on allocation/free, logging or lazy initialization. Add failure injection at
every suspension/context/patch step, verifying exact suspension balance and
rollback, plus repeated concurrent thread creation. Then perform one
capture-only game test with exception logging and a hang stack snapshot.

No hook binary repair has been installed. The independent bundle-header repair
remains active; it does not address this suspension/allocation hazard.
