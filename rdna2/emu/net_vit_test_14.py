"""Test steps 1-4 isolation for k_attention2 verification."""
import sys, re, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K

H = W = 8
NGROUP = 4
BIN, B260, B268, B270, B278, B280, B288, BOUT = 0, 1, 2, 3, 4, 5, 6, 7

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
    n = META[EXPAND2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(BIN))
    struct.pack_into('<Q', ka, 0x08, S(B260))
    struct.pack_into('<Q', ka, 0x10, S(WL0))
    out.append((EXPAND2, ka, META[EXPAND2]))

    n = META[CONTRACT2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B260))
    struct.pack_into('<Q', ka, 0x08, S(BIN))
    struct.pack_into('<Q', ka, 0x10, S(B268))
    struct.pack_into('<Q', ka, 0x18, S(WL1))
    struct.pack_into('<ii', ka, 0x20, H, W)
    struct.pack_into('<i', ka, 0x28, NGROUP)
    out.append((CONTRACT2, ka, META[CONTRACT2]))

    n = META[QKV2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B268))
    struct.pack_into('<Q', ka, 0x08, S(B270))
    struct.pack_into('<Q', ka, 0x10, S(B278))
    struct.pack_into('<Q', ka, 0x18, S(B280))
    struct.pack_into('<Q', ka, 0x20, S(WL2))
    out.append((QKV2, ka, META[QKV2]))

    n = META[ATTN2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B270))
    struct.pack_into('<Q', ka, 0x08, S(B278))
    struct.pack_into('<Q', ka, 0x10, S(B280))
    struct.pack_into('<Q', ka, 0x18, S(B288))
    struct.pack_into('<ii', ka, 0x20, W, H)
    out.append((ATTN2, ka, META[ATTN2]))

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
    t = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
    kd = t.split('.amdhsa_kernel ' + sym, 1)[1].split('.end_amdhsa_kernel', 1)[0]
    return '.amdhsa_user_sgpr_dispatch_ptr 1' in kd

def emulate(steps_list, seed):
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
                       capture_output=True, text=True, timeout=30)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out

print('ISOLATED TEST: Steps 1-3 and Steps 1-4')
print('=' * 60)

# Steps 1-3
print('\n[1/4] Running net_run([0,1,2])...')
rc13, msg13, out13 = net_run([0, 1, 2], 1)
print(msg13.splitlines()[-1] if msg13 else 'FAIL')

if out13 is None:
    print('GPU FAIL'); sys.exit(1)

V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg13)[1], 16)
print('[2/4] Running emulate([0,1,2])...')
base13, ref13 = emulate([0, 1, 2], 1)
d13 = np.nonzero(out13 != ref13)[0]
touched13 = sorted({int(i // V.SLOT) for i in np.nonzero(ref13 != base13)[0]})
bytes_written_13 = int((ref13 != base13).sum())
print('emulator wrote %d bytes into slots %s' % (bytes_written_13, touched13))
print('net_run vs emulator (steps 1-3): %d mismatches' % len(d13))

# Steps 1-4
print('\n[3/4] Running net_run([0,1,2,3])...')
rc14, msg14, out14 = net_run([0, 1, 2, 3], 1)
print(msg14.splitlines()[-1] if msg14 else 'FAIL')

if out14 is None:
    print('GPU FAIL'); sys.exit(1)

print('[4/4] Running emulate([0,1,2,3])...')
base14, ref14 = emulate([0, 1, 2, 3], 1)
d14 = np.nonzero(out14 != ref14)[0]
touched14 = sorted({int(i // V.SLOT) for i in np.nonzero(ref14 != base14)[0]})
bytes_written_14 = int((ref14 != base14).sum())
print('emulator wrote %d bytes into slots %s' % (bytes_written_14, touched14))
print('net_run vs emulator (steps 1-4): %d mismatches' % len(d14))

# Analysis
print('\n' + '=' * 60)
print('RESULTS:')
print('  bytes_written_13 (steps 1-3):  %d' % bytes_written_13)
print('  bytes_written_14 (steps 1-4):  %d' % bytes_written_14)
attn2_delta = bytes_written_14 - bytes_written_13
print('  => k_attention2 delta:         %d' % attn2_delta)
print('  mismatches_14 (steps 1-4):     %d' % len(d14))
print('\nDID k_attention2 WRITE SUBSTANTIAL AMOUNT AND MATCH?')
if attn2_delta > 1000 and len(d14) == 0:
    print('  YES: %d bytes written, 0 mismatches' % attn2_delta)
elif attn2_delta > 1000:
    print('  PARTIAL: %d bytes written, but %d mismatches' % (attn2_delta, len(d14)))
else:
    print('  NO: only %d bytes written' % attn2_delta)
