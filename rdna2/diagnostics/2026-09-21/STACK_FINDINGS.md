# Live stack inspection

Inspected Cyberpunk PID 17052 and reporter PID 3228 with
rdna2/inspect_process_stacks.cpp. Unrestricted process inspection was required
after sandboxed SymInitialize returned access denied. The helper adds one
temporary suspension per inspected thread and removes exactly that increment;
existing suspensions remain intact. No processes were terminated or code patched.

game-stacks.txt and reporter-stacks.txt contain module-relative frames.
Numerous game threads already had suspend count 1. Research-runtime thread
9380 had count 0 and was waiting in ntdll with callers:
version.dll+0x4e824, +0x3b9d3, +0x21311, +0x9d4d, +0x90ba.
Disassembly shows the +0x21311 return site follows allocation during growth of
an eight-byte-element container. The +0x9d4d caller is in a thread-enumeration
loop, after a call to +0x36370 operating on the opened thread handle.
The latter helper also allocates memory and calls another thread operation
at +0x65c10. Many peers are suspended while this allocation is waiting.

This supports investigating an allocation/lock deadlock during thread suspension
in graphics-hook setup. Exact lock ownership and the imported operation behind
+0x65c10 were not yet resolved, so do not claim a fully proven lock cycle.
There is no HIP frame on this blocked worker's captured stack.
The reporter's main stack waits in Windows with caller reporter+0x99ad.

Next: resolve the thread-operation import and allocator wait/lock owner, then
review the hook transaction so all storage is prepared before suspending peers.
Do not blindly resume all threads: that could interrupt an active hook update.
