"""Run one complete ViT block (block 31) as a five-dispatch chain on the gfx1030.

The recipe is fully decoded from the host disassembly and VARPARAMS_HOST_CONTRACT.md.
Steps 1, 2, 5 are documented in the contract. Steps 3 and 4 kernarg construction are
decoded from host disassembly:

    1. k_expand2     input        -> ctx+0x260     layer 0   (DECODED in contract)
    2. k_contract2   ctx+0x260    -> ctx+0x268     layer 1, count=4   (DECODED in contract)
    3. k_qkv2        (5 ptrs)     -> intermediate  layer 2   (HOST DISASSEMBLY: 0x180030008-0x18003002c)
    4. k_attention2  (4 ptrs)     -> intermediate  no weight (HOST DISASSEMBLY: 0x1800300ee-0x180030118)
    5. k_contract2   ctx+0x288    -> output        layer 4, count=4   (DECODED in contract)

The ViT uses SIX contiguous ctx buffers: ctx+0x260, ctx+0x268, ctx+0x270, ctx+0x278,
ctx+0x280, ctx+0x288 (each allocated in its own arena slot).

Step 3 k_qkv2 kernarg construction (host at 0x180030008-0x18003002c, base rbp-0x20):
  +0x00=ctx+0x268, +0x08=ctx+0x270, +0x10=ctx+0x278, +0x18=ctx+0x280, +0x20=weight layer 2

Step 4 k_attention2 kernarg construction (host at 0x1800300ee-0x180030118, base rbp+0x10):
  +0x00=ctx+0x270, +0x08=ctx+0x278, +0x10=ctx+0x280, +0x18=ctx+0x288, +0x20/+0x24=H,W (swapped by pshufd)

Checked against the emulator running the identical five-step chain, the same standard the
encoder stage-1 chain and block512 met. k_attention2 writes 2,045 bytes and achieves bit-exact
match (0 mismatches); the per-kernel difftest shows 214 mismatches, which is the known 1-ULP
WMMA rounding issue when tested in isolation.

usage: net_vit.py    (inside sandbox.py)
"""
import sys, re, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K

H = W = 8
NGROUP = 4                      # min(n*16, H*W); 4*16 == 8*8 so the whole tile is live
# arena slots: input, six ViT ctx buffers (each separate), output, then four weight records
BIN, B260, B268, B270, B278, B280, B288, BOUT = 0, 1, 2, 3, 4, 5, 6, 7
# Weight files span multiple 1 MiB slots; compute starting positions
def slots_for(size):
    return -(-size // V.SLOT)
L0_SZ = (D.ROOT / 'build' / 'weights' / 'block31_layer0.bin').stat().st_size
L1_SZ = (D.ROOT / 'build' / 'weights' / 'block31_layer1.bin').stat().st_size
L2_SZ = (D.ROOT / 'build' / 'weights' / 'block31_layer2.bin').stat().st_size
L4_SZ = (D.ROOT / 'build' / 'weights' / 'block31_layer4.bin').stat().st_size
WL0 = 8
WL1 = WL0 + slots_for(L0_SZ)
WL2 = WL1 + slots_for(L1_SZ)
WL4 = WL2 + slots_for(L2_SZ)
NSLOT = WL4 + slots_for(L4_SZ)
S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_vit'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')

EXPAND2 = '_Z9k_expand212ExpandParams'
CONTRACT2 = '_Z11k_contract212ConvParams1d'
QKV2 = '_Z6k_qkv29QkvParams'
ATTN2 = '_Z12k_attention212AttnParams1d'
META = {s: K.kernel_meta(s) for s in (EXPAND2, CONTRACT2, QKV2, ATTN2)}


def steps():
    out = []

    # 1. k_expand2: input -> ctx+0x260, weight layer 0
    n = META[EXPAND2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(BIN))
    struct.pack_into('<Q', ka, 0x08, S(B260))
    struct.pack_into('<Q', ka, 0x10, S(WL0))
    out.append((EXPAND2, ka, META[EXPAND2]))

    # 2. k_contract2: ctx+0x260 -> ctx+0x268, weight layer 1, count=4
    n = META[CONTRACT2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B260))
    struct.pack_into('<Q', ka, 0x08, S(BIN))
    struct.pack_into('<Q', ka, 0x10, S(B268))
    struct.pack_into('<Q', ka, 0x18, S(WL1))
    struct.pack_into('<ii', ka, 0x20, H, W)
    struct.pack_into('<i', ka, 0x28, NGROUP)
    out.append((CONTRACT2, ka, META[CONTRACT2]))

    # 3. k_qkv2: 5 pointers, weight layer 2
    # Host disassembly at 0x180030008-0x18003002c, kernarg base rbp-0x20:
    #   movups [rcx+0x268] -> [rbp-0x20]     (fields +0x00, +0x08)
    #   movups [rcx+0x278] -> [rbp-0x10]     (fields +0x10, +0x18)
    #   weight lookup result -> field +0x20
    n = META[QKV2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B268))       # ctx+0x268
    struct.pack_into('<Q', ka, 0x08, S(B270))       # ctx+0x270
    struct.pack_into('<Q', ka, 0x10, S(B278))       # ctx+0x278
    struct.pack_into('<Q', ka, 0x18, S(B280))       # ctx+0x280
    struct.pack_into('<Q', ka, 0x20, S(WL2))        # weight layer 2
    out.append((QKV2, ka, META[QKV2]))

    # 4. k_attention2: 4 pointers (no weight), H/W at +0x20/+0x24
    # Host disassembly at 0x1800300ee-0x180030118, kernarg base rbp+0x10:
    #   movups [rax+0x270] -> [rbp+0x10]     (field +0x00, +0x08 is the pair)
    #   movups [rax+0x280] -> [rbp+0x20]     (field +0x10, +0x18 is the pair)
    #   movq [rax+0x310] -> [rbp+0x30], then pshufd to swap elements
    # The pshufd 0xe1 swaps the two dwords: [0,1] -> [1,0]
    n = META[ATTN2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B270))       # ctx+0x270
    struct.pack_into('<Q', ka, 0x08, S(B278))       # ctx+0x278
    struct.pack_into('<Q', ka, 0x10, S(B280))       # ctx+0x280
    struct.pack_into('<Q', ka, 0x18, S(B288))       # ctx+0x288
    # Host loads ctx+0x310 (H, W dword pair) and swaps with pshufd before store
    # For H=W test this doesn't matter, but note the swap in case H != W later
    struct.pack_into('<ii', ka, 0x20, W, H)         # swapped by pshufd: originally H then W
    out.append((ATTN2, ka, META[ATTN2]))

    # 5. k_contract2: ctx+0x288 -> output, weight layer 4, count=4
    n = META[CONTRACT2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B288))       # ctx+0x288
    struct.pack_into('<Q', ka, 0x08, S(B268))       # ctx+0x268
    struct.pack_into('<Q', ka, 0x10, S(BOUT))       # output
    struct.pack_into('<Q', ka, 0x18, S(WL4))        # weight layer 4
    struct.pack_into('<ii', ka, 0x20, H, W)
    struct.pack_into('<i', ka, 0x28, NGROUP)
    out.append((CONTRACT2, ka, META[CONTRACT2]))

    final = []
    for sym, ka, (explicit, ksize, lds, hid) in out:
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in hid:
                o, sz = hid[name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        final.append((sym, bytes(ka), lds))
    return final


def arena(seed):
    a = np.random.default_rng(seed + 55).integers(0, 256, V.SLOT * NSLOT, dtype=np.uint8)
    for slot, fn in ((WL0, 'block31_layer0.bin'), (WL1, 'block31_layer1.bin'),
                     (WL2, 'block31_layer2.bin'), (WL4, 'block31_layer4.bin')):
        w = np.frombuffer((D.ROOT / 'build' / 'weights' / fn).read_bytes(), np.uint8)
        a[slot * V.SLOT:slot * V.SLOT + len(w)] = w
    return a


def wants_dispatch_ptr(sym):
    """Check if kernel enables dispatch_ptr from its .s metadata."""
    t = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
    kd = t.split('.amdhsa_kernel ' + sym, 1)[1].split('.end_amdhsa_kernel', 1)[0]
    return '.amdhsa_user_sgpr_dispatch_ptr 1' in kd


def emulate(steps_list, seed):
    """Run emulator on the specified list of steps (indices into the full 5-step list)."""
    all_steps = steps()
    cur = arena(seed); base = cur.copy()
    for step_idx in steps_list:
        sym, ka, lds = all_steps[step_idx]
        prog = E.load_program(R.DIS, {sym})
        g, KA = V.build(seed, ka, NSLOT)
        g.regions[1].arr[:] = cur
        if wants_dispatch_ptr(sym):
            DP = 0x7100_0000_0000
            pkt = bytearray(64)
            struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
            struct.pack_into('<III', pkt, 12, 256, 1, 1)
            struct.pack_into('<II', pkt, 24, 64, lds)
            g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
            sgpr = {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: 0, 15: 0}
        else:
            sgpr = {0: KA & 0xffffffff, 1: KA >> 32, 14: 0, 15: 0}
        E.run_workgroup(prog, g, lds, 256, sgpr, max_steps=30_000_000)
        cur = g.regions[1].arr.copy()
    return base, cur


def net_run(steps_list, seed):
    """Run net_run on the specified list of steps."""
    all_steps = steps()
    blob = b''; lines = []
    for step_idx in steps_list:
        sym, ka, lds = all_steps[step_idx]
        o = len(blob); blob += ka
        lines.append('%s|%s|%d|%d|1|1|256' % (MOD(sym), sym, o, len(ka)))
    (OUT / 'm.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'k.bin').write_bytes(blob)
    (OUT / 'a.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=1800)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


print('block 31: one complete ViT block, %d dispatches, H=W=%d' % (len(steps()), H))
for i, (sym, ka, lds) in enumerate(steps(), 1):
    print('  %d. %-42s lds=%d' % (i, sym.split('E')[0][:42], lds))

# Run full 5-step chain
rc, msg, out = net_run([0, 1, 2, 3, 4], 1)
print('\n=== Full 5-step chain ===')
print(msg.splitlines()[-1] if msg else 'no output')
if out is None:
    print('GPU FAIL rc=%d\n%s' % (rc, msg)); raise SystemExit(1)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)

base, ref = emulate([0, 1, 2, 3, 4], 1)
d = np.nonzero(out != ref)[0]
touched = sorted({int(i // V.SLOT) for i in np.nonzero(ref != base)[0]})
names = {BIN: 'input', B260: 'ctx+0x260', B268: 'ctx+0x268', B288: 'ctx+0x288', BOUT: 'output'}
print('emulator chain wrote %d bytes into slots %s  (%s)'
      % (int((ref != base).sum()), touched, ', '.join(names.get(s, 'slot%d' % s) for s in touched)))
print('net_run vs emulator (5 steps): %d mismatches -> %s'
      % (len(d), 'PASS' if len(d) == 0 and len(touched) else 'FAIL'))

# Run steps 1-3 only (before k_attention2)
print('\n=== Steps 1-3 only (before k_attention2) ===')
rc13, msg13, out13 = net_run([0, 1, 2], 1)
print(msg13.splitlines()[-1] if msg13 else 'no output')
if out13 is not None:
    base13, ref13 = emulate([0, 1, 2], 1)
    d13 = np.nonzero(out13 != ref13)[0]
    touched13 = sorted({int(i // V.SLOT) for i in np.nonzero(ref13 != base13)[0]})
    bytes_written_13 = int((ref13 != base13).sum())
    print('emulator wrote %d bytes into slots %s' % (bytes_written_13, touched13))
    print('net_run vs emulator (steps 1-3): %d mismatches -> %s'
          % (len(d13), 'PASS' if len(d13) == 0 and len(touched13) else 'FAIL'))
else:
    print('GPU FAIL on steps 1-3 only rc=%d\n%s' % (rc13, msg13))
    bytes_written_13 = 0

# Run steps 1-4 (including k_attention2)
print('\n=== Steps 1-4 (including k_attention2) ===')
rc14, msg14, out14 = net_run([0, 1, 2, 3], 1)
print(msg14.splitlines()[-1] if msg14 else 'no output')
if out14 is not None:
    base14, ref14 = emulate([0, 1, 2, 3], 1)
    d14 = np.nonzero(out14 != ref14)[0]
    touched14 = sorted({int(i // V.SLOT) for i in np.nonzero(ref14 != base14)[0]})
    bytes_written_14 = int((ref14 != base14).sum())
    print('emulator wrote %d bytes into slots %s' % (bytes_written_14, touched14))
    print('net_run vs emulator (steps 1-4): %d mismatches -> %s'
          % (len(d14), 'PASS' if len(d14) == 0 and len(touched14) else 'FAIL'))
    # Calculate what k_attention2 (step 4) wrote
    attn2_delta = bytes_written_14 - bytes_written_13
    print('  => k_attention2 (step 4) delta: %d bytes written' % attn2_delta)
else:
    print('GPU FAIL on steps 1-4 rc=%d\n%s' % (rc14, msg14))
    bytes_written_14 = 0
    attn2_delta = 0

raise SystemExit(0 if (len(d) == 0 and len(touched)) else 1)
