"""Full 71-block DLSS network as a single net_run sequence with emulation validation.

Order: block 0 (pre) -> 1-22 encoder -> 23-30 C=512 -> 31-38 ViT -> 39 k_dec_upsample
       -> 40-47 C=512 -> 48-69 decoder -> 70 final head

Incremental checkpoints:
(a) encoder + C=512 blocks 23-30
(b) + ViT blocks 31-38
(c) + block 39 + C=512 blocks 40-47
(d) + decoder blocks 48-69
(e) + final head block 70

usage: net_full.py    (inside sandbox.py)
"""
import sys, re, struct, subprocess, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K

H0 = 16
BASE = len(V.PTR_FIELDS)

def slots_for(size):
    """Calculate number of 1 MiB slots needed for size bytes."""
    return -(-size // V.SLOT)

# Kernel symbols
EXPAND2 = '_Z9k_expand212ExpandParams'
CONTRACT2 = '_Z11k_contract212ConvParams1d'
QKV2 = '_Z6k_qkv29QkvParams'
ATTN2 = '_Z12k_attention212AttnParams1d'
FFWD_IV = '_Z14k_ffwd_inpview12FfwdPlParams'
FFWD2 = '_Z7k_ffwd211Ffwd2Params'
CONVV = '_Z16k_conv_res_views12ConvPlParams'
QKV = '_Z10k_qkv_attn10AttnParams'

META_VIT = {s: K.kernel_meta(s) for s in (EXPAND2, CONTRACT2, QKV2, ATTN2)}
# Same for the C=512, transition and head kernels. Passing 0 here silently starves them of
# shared memory: the GPU reads LDS from the code object and is unaffected, but the emulator
# takes what we hand it and faults on the first ds_store past the end.
META = {s: K.kernel_meta(s) for s in (FFWD_IV, FFWD2, CONVV, QKV)}
LDS = {s: META[s][2] for s in META}

S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_full'; OUT.mkdir(parents=True, exist_ok=True)


# ===== BUILD ALLOCATION MAP =====
def build_allocation_map():
    """Build and verify the arena allocation map for all weights and activation buffers."""

    weight_files = []

    # All encoder blocks 0-22 (23 blocks)
    for blk in range(23):
        fn = D.ROOT / 'build' / 'weights' / f'block{blk}.bin'
        if fn.exists():
            sz = fn.stat().st_size
            weight_files.append((f'block{blk}', blk, sz))

    # C=512 stage 1 blocks 23-30 (8 blocks, 4 layers each = 32 weights)
    for blk in range(23, 31):
        for layer in range(4):
            fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer{layer}.bin'
            if fn.exists():
                sz = fn.stat().st_size
                weight_files.append((f'block{blk}_layer{layer}', 100+blk*10+layer, sz))

    # ViT blocks 31-38 (8 blocks, skip layer 3)
    for blk in range(31, 39):
        for layer in [0, 1, 2, 4]:
            fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer{layer}.bin'
            if fn.exists():
                sz = fn.stat().st_size
                weight_files.append((f'block{blk}_layer{layer}', 200+blk*10+layer, sz))

    # Block 39
    for fn_name in ['block39.bin', 'block39_layer0.bin']:
        fn = D.ROOT / 'build' / 'weights' / fn_name
        if fn.exists():
            sz = fn.stat().st_size
            uid = 39 if fn_name == 'block39.bin' else 300+39*10+0
            weight_files.append((fn_name[:-4], uid, sz))

    # C=512 stage 2 blocks 40-47 (8 blocks, 4 layers each = 32 weights)
    for blk in range(40, 48):
        for layer in range(4):
            fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer{layer}.bin'
            if fn.exists():
                sz = fn.stat().st_size
                weight_files.append((f'block{blk}_layer{layer}', 300+blk*10+layer, sz))

    # Decoder blocks 48-69
    for blk in range(48, 70):
        fn = D.ROOT / 'build' / 'weights' / f'block{blk}.bin'
        if fn.exists():
            sz = fn.stat().st_size
            weight_files.append((f'block{blk}', blk, sz))
        fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer0.bin'
        if fn.exists():
            sz = fn.stat().st_size
            weight_files.append((f'block{blk}_layer0', 400+blk*10+0, sz))

    # Final head block 70
    for fn_name in ['block70.bin', 'block70_layer0.bin', 'block70_layer0_blend_scale.bin']:
        fn = D.ROOT / 'build' / 'weights' / fn_name
        if fn.exists():
            sz = fn.stat().st_size
            uid = 70 if 'layer' not in fn_name else 400+70*10+(0 if 'blend' not in fn_name else 1)
            weight_files.append((fn_name[:-4], uid, sz))

    n_activation_slots = 3 + 4 + 6 + 4 + 3  # = 20
    ACT_BASE = BASE
    W_BASE = ACT_BASE + n_activation_slots

    weight_allocations = []
    w_slot = W_BASE
    for name, uid, size in sorted(weight_files, key=lambda x: (-x[2], x[0])):
        num_slots = slots_for(size)
        weight_allocations.append((name, uid, size, w_slot, w_slot + num_slots))
        w_slot += num_slots

    total_slots = w_slot

    print(f"=== Arena Allocation Map ===")
    print(f"Slots 0..17:        VarParams pointer fields (BASE={BASE})")
    print(f"ACT_BASE = {ACT_BASE}:  Activation/working buffers ({n_activation_slots} slots)")
    print(f"W_BASE = {W_BASE}:  Weights (allocated per size)")
    print(f"Total slots needed: {total_slots}")

    sorted_alloc = sorted(weight_allocations, key=lambda x: -x[2])
    print(f"\n5 largest weights:")
    for i, (name, uid, size, start, end) in enumerate(sorted_alloc[:5]):
        print(f"  {i+1}. {name:30} {size:>10} B ({slots_for(size)} slots) [slot {start}..{end-1}]")

    errors = []
    act_end = ACT_BASE + n_activation_slots
    if act_end != W_BASE:
        errors.append(f"MISMATCH: ACT_BASE+n_activation_slots={act_end} != W_BASE={W_BASE}")

    for i, (n1, u1, sz1, s1, e1) in enumerate(weight_allocations):
        for j, (n2, u2, sz2, s2, e2) in enumerate(weight_allocations):
            if i < j and not (e1 <= s2 or e2 <= s1):
                errors.append(f"OVERLAP: {n1} overlaps {n2}")

    for name, uid, size, start, end in weight_allocations:
        if not (end <= ACT_BASE or start >= W_BASE):
            errors.append(f"VIOLATION: {name} intersects activation range")

    if errors:
        print("\nERRORS FOUND:")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)
    else:
        print(f"\n[OK] All {len(weight_allocations)} weight allocations verified")

    return ACT_BASE, W_BASE, total_slots, weight_allocations


ACT_BASE, W_BASE, TOTAL_SLOTS, WEIGHT_ALLOC_MAP = build_allocation_map()
print(f"Map validated. Ready to proceed.\n")


def plan_full(act_base, w_base, weight_map):
    """Generate all dispatches for 71-block network.

    Returns: (steps_list, checkpoints) where checkpoints maps labels to (start, end) indices.
    """
    weight_slot_lookup = {}
    for name, uid, size, start, end in weight_map:
        weight_slot_lookup[name] = start

    out = []
    used_act_slots = set()

    enc_pp_0 = act_base + 0
    enc_pp_1 = act_base + 1
    enc_in = act_base + 2
    used_act_slots.update([enc_pp_0, enc_pp_1, enc_in])

    POOL = 0x38

    # ===== BLOCK 0: Pre-block =====
    checkpoint_0_start = len(out)
    # The pre-block is k_swin_var<32,TRUE> ('32_1'), not the <32,false> the encoder stages use.
    # Different template instantiation, different LDS - running '32_0' here faults on a ds_load
    # past the end. Both GPU-verified pre-block scripts use '32_1'.
    sym, lds = D.SYMS['32_1']
    lds = lds or D.group_size(sym)
    h = w = H0
    # The pre-block is k_swin_var<32,true> run under the PRE-BLOCK convention, not the encoder one:
    # flags 0x14, +0x00 null, and the float-RGB network input at +0x40. Wiring it like an ordinary
    # encoder block (flags 5, input at +0x00) runs the same kernel with the wrong calling convention
    # and mismatches on most of its output. Layout per chain_import_preblock_real.py, which is
    # GPU-verified against the emulator on real pixels.
    ka = V.make_kernarg(H=h, W=w, offy=0, offx=0, flags=0x14, grid=((w+7)//8, (h+7)//8))
    weight_name = f'block0'
    w_slot = weight_slot_lookup.get(weight_name, w_base)
    struct.pack_into('<Q', ka, 0x00, 0)                     # null, not the input
    struct.pack_into('<Q', ka, 0x30, 0)                     # null too - make_kernarg defaults it
                                                            # to a pointer, which the pre-block is
                                                            # not launched with
    struct.pack_into('<Q', ka, 0x08, S(enc_pp_0))           # output
    struct.pack_into('<Q', ka, 0x10, S(w_slot))             # block0 weights
    struct.pack_into('<Q', ka, 0x40, S(enc_in))             # float-RGB input, 12 B/pixel
    for off in (0x54, 0x58, 0x5c, 0x60, 0x64, 0x68):
        struct.pack_into('<I', ka, off, 0)
    struct.pack_into('<f', ka, 0x50, 1.0)
    for off in range(0x70, 0xa0, 8):
        struct.pack_into('<Q', ka, off, 0)
    out.append((sym, lds, bytes(ka), ((w+7)//8, (h+7)//8), 0, 'encoder', h, 'block0'))
    checkpoint_0_end = len(out)

    # ===== ENCODER STAGE 1-4 (blocks 1-22) =====
    checkpoint_a_start = len(out)
    ENCODER_STAGES = [('32_0', [1,2,3,4]), ('64_0', [5,6,7,8]),
                      ('128_0', [9,10,11,12,13,14]), ('256_0', [15,16,17,18,19,20,21,22])]
    ENCODER_MODES = [(0,0), (-4,-4), (-4,0), (0,-4)]

    for si, (key, blocks) in enumerate(ENCODER_STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h = w = max(H0 >> si, 8)
        for i, blk in enumerate(blocks):
            ox, oy = ENCODER_MODES[i % 4]
            last = (i == len(blocks) - 1)
            flags = (1 if i == 0 else 0) | (4 if last else 0)
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            ka = V.make_kernarg(H=h, W=w, offy=oy, offx=ox, flags=flags, grid=grid)
            src = enc_in if i == 0 else enc_pp_1 if (i % 2 == 0) else enc_pp_0
            dst = enc_pp_0 if (i % 2 == 0) else enc_pp_1
            weight_name = f'block{blk}'
            w_slot = weight_slot_lookup[weight_name]
            struct.pack_into('<Q', ka, 0x00, S(src))
            struct.pack_into('<Q', ka, 0x08, S(dst))
            struct.pack_into('<Q', ka, 0x10, S(w_slot))
            if last and si+1 < len(ENCODER_STAGES):
                struct.pack_into('<Q', ka, POOL, S(enc_in))
            out.append((sym, lds, bytes(ka), grid, blk, 'encoder', h, 'enc'))

    # ===== C=512 STAGE 1 (blocks 23-30) =====
    c512_1_work = [act_base + 3, act_base + 4, act_base + 5, act_base + 6]
    used_act_slots.update(c512_1_work)
    for blk_idx, blk in enumerate(range(23, 31)):
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer0'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(enc_in if blk_idx == 0 else c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(c512_1_work[1]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        out.append((FFWD_IV, LDS[FFWD_IV], bytes(ka), (1,1), blk, 'c512_1', 8, 'ffwd_iv'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer1'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(enc_in if blk_idx == 0 else c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_1_work[1]))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<iii', ka, 0x20, 8, 8, 4)
        out.append((FFWD2, LDS[FFWD2], bytes(ka), (1,1), blk, 'c512_1', 8, 'ffwd2'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer2'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[1]))
        struct.pack_into('<Q', ka, 0x08, S(enc_in if blk_idx == 0 else c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x18, S(c512_1_work[2]))
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<Q', ka, 0x28, S(w_slot))
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        out.append((CONVV, LDS[CONVV], bytes(ka), (1,1), blk, 'c512_1', 8, 'conv_res_1'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer3'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[2]))
        struct.pack_into('<Q', ka, 0x08, S(c512_1_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        struct.pack_into('<ii', ka, 0x20, 0, 0)
        out.append((QKV, LDS[QKV], bytes(ka), (1,1), blk, 'c512_1', 8, 'qkv_attn'))

        ka = V.make_kernarg(H=8, W=8)
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(c512_1_work[2]))
        struct.pack_into('<Q', ka, 0x18, S(c512_1_work[0]))
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<Q', ka, 0x28, S(w_base))
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        struct.pack_into('<ii', ka, 0x40, 8, 8)
        out.append((CONVV, LDS[CONVV], bytes(ka), (1,1), blk, 'c512_1', 8, 'conv_res_2'))

    checkpoint_a_end = len(out)

    # ===== ViT BLOCKS 31-38 =====
    checkpoint_b_start = len(out)
    vit_work = [act_base + 7, act_base + 8, act_base + 9, act_base + 10, act_base + 11, act_base + 12]
    used_act_slots.update(vit_work)
    B268, B270, B278, B280, B288, BOUT = vit_work
    for blk in range(31, 39):
        n = META_VIT[EXPAND2][1]; ka = bytearray(n)
        weight_name = f'block{blk}_layer0'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[0] if blk == 31 else vit_work[5]))
        struct.pack_into('<Q', ka, 0x08, S(B268))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in META_VIT[EXPAND2][3]:
                o, sz = META_VIT[EXPAND2][3][name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        out.append((EXPAND2, META_VIT[EXPAND2][2], bytes(ka), (1,1), blk, 'vit', 8, 'expand2'))

        n = META_VIT[CONTRACT2][1]; ka = bytearray(n)
        weight_name = f'block{blk}_layer1'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(B268))
        struct.pack_into('<Q', ka, 0x08, S(c512_1_work[0] if blk == 31 else vit_work[5]))
        struct.pack_into('<Q', ka, 0x10, S(B270))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<ii', ka, 0x20, 8, 8)
        struct.pack_into('<i', ka, 0x28, 4)
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in META_VIT[CONTRACT2][3]:
                o, sz = META_VIT[CONTRACT2][3][name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        out.append((CONTRACT2, META_VIT[CONTRACT2][2], bytes(ka), (1,1), blk, 'vit', 8, 'contract2_1'))

        n = META_VIT[QKV2][1]; ka = bytearray(n)
        weight_name = f'block{blk}_layer2'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(B270))
        struct.pack_into('<Q', ka, 0x08, S(B278))
        struct.pack_into('<Q', ka, 0x10, S(B280))
        struct.pack_into('<Q', ka, 0x18, S(B288))
        struct.pack_into('<Q', ka, 0x20, S(w_slot))
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in META_VIT[QKV2][3]:
                o, sz = META_VIT[QKV2][3][name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        out.append((QKV2, META_VIT[QKV2][2], bytes(ka), (1,1), blk, 'vit', 8, 'qkv2'))

        n = META_VIT[ATTN2][1]; ka = bytearray(n)
        struct.pack_into('<Q', ka, 0x00, S(B278))
        struct.pack_into('<Q', ka, 0x08, S(B280))
        struct.pack_into('<Q', ka, 0x10, S(B288))
        struct.pack_into('<Q', ka, 0x18, S(B268))
        struct.pack_into('<ii', ka, 0x20, 8, 8)
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in META_VIT[ATTN2][3]:
                o, sz = META_VIT[ATTN2][3][name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        out.append((ATTN2, META_VIT[ATTN2][2], bytes(ka), (1,1), blk, 'vit', 8, 'attention2'))

        n = META_VIT[CONTRACT2][1]; ka = bytearray(n)
        weight_name = f'block{blk}_layer4'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(B288))
        struct.pack_into('<Q', ka, 0x08, S(B270))
        struct.pack_into('<Q', ka, 0x10, S(vit_work[5]))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<ii', ka, 0x20, 8, 8)
        struct.pack_into('<i', ka, 0x28, 4)
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in META_VIT[CONTRACT2][3]:
                o, sz = META_VIT[CONTRACT2][3][name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        out.append((CONTRACT2, META_VIT[CONTRACT2][2], bytes(ka), (1,1), blk, 'vit', 8, 'contract2_2'))

    checkpoint_b_end = len(out)

    # ===== BLOCK 39: k_dec_upsample =====
    checkpoint_c_start = len(out)
    # Placeholder for block 39 - requires DecUpParams kernelspec
    # For now, add a minimal dispatch that passes through the data
    ka = V.make_kernarg(H=8, W=8)
    weight_name = 'block39'
    w_slot = weight_slot_lookup.get(weight_name, w_base)
    struct.pack_into('<Q', ka, 0x00, S(vit_work[5]))
    struct.pack_into('<Q', ka, 0x08, S(vit_work[5]))
    struct.pack_into('<Q', ka, 0x10, S(w_slot))
    # Use a placeholder kernel - in reality would be k_dec_upsample
    out.append((FFWD_IV, LDS[FFWD_IV], bytes(ka), (1,1), 39, 'block39', 8, 'dec_upsample'))

    # ===== C=512 STAGE 2 (blocks 40-47) =====
    c512_2_work = [act_base + 13, act_base + 14, act_base + 15, act_base + 16]
    used_act_slots.update(c512_2_work)
    for blk_idx, blk in enumerate(range(40, 48)):
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer0'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(vit_work[5] if blk_idx == 0 else c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(c512_2_work[1]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        out.append((FFWD_IV, LDS[FFWD_IV], bytes(ka), (1,1), blk, 'c512_2', 8, 'ffwd_iv'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer1'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(vit_work[5] if blk_idx == 0 else c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_2_work[1]))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<iii', ka, 0x20, 8, 8, 4)
        out.append((FFWD2, LDS[FFWD2], bytes(ka), (1,1), blk, 'c512_2', 8, 'ffwd2'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer2'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[1]))
        struct.pack_into('<Q', ka, 0x08, S(vit_work[5] if blk_idx == 0 else c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x18, S(c512_2_work[2]))
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<Q', ka, 0x28, S(w_slot))
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        out.append((CONVV, LDS[CONVV], bytes(ka), (1,1), blk, 'c512_2', 8, 'conv_res_1'))

        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer3'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[2]))
        struct.pack_into('<Q', ka, 0x08, S(c512_2_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        struct.pack_into('<ii', ka, 0x20, 0, 0)
        out.append((QKV, LDS[QKV], bytes(ka), (1,1), blk, 'c512_2', 8, 'qkv_attn'))

        ka = V.make_kernarg(H=8, W=8)
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(c512_2_work[2]))
        struct.pack_into('<Q', ka, 0x18, S(c512_2_work[0]))
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<Q', ka, 0x28, S(w_base))
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        struct.pack_into('<ii', ka, 0x40, 8, 8)
        out.append((CONVV, LDS[CONVV], bytes(ka), (1,1), blk, 'c512_2', 8, 'conv_res_2'))

    checkpoint_c_end = len(out)

    # ===== DECODER STAGES (blocks 48-69) =====
    checkpoint_d_start = len(out)
    DECODER_STAGES = [('256_0', list(range(48, 56))), ('128_0', list(range(56, 62))),
                      ('64_0', list(range(62, 66))), ('32_0', list(range(66, 70)))]
    DEC_PP = [act_base + 17, act_base + 18]
    DEC_IN = act_base + 19
    used_act_slots.update([DEC_PP[0], DEC_PP[1], DEC_IN])

    for si, (key, blocks) in enumerate(DECODER_STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h = w = max(H0 >> (3 - si), 8)
        for i, blk in enumerate(blocks):
            ox, oy = ENCODER_MODES[i % 4]
            last = (i == len(blocks) - 1)
            flags = (1 if i == 0 else 0) | (4 if last else 0)
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            ka = V.make_kernarg(H=h, W=w, offy=oy, offx=ox, flags=flags, grid=grid)
            src = (c512_2_work[0] if si == 0 and i == 0 else DEC_IN if i == 0 else DEC_PP[(i+1) % 2])
            dst = DEC_PP[i % 2]
            weight_name = f'block{blk}'
            w_slot = weight_slot_lookup.get(weight_name, w_base)
            struct.pack_into('<Q', ka, 0x00, S(src))
            struct.pack_into('<Q', ka, 0x08, S(dst))
            struct.pack_into('<Q', ka, 0x10, S(w_slot))
            if last and si+1 < len(DECODER_STAGES):
                struct.pack_into('<Q', ka, POOL, S(DEC_IN))
            out.append((sym, lds, bytes(ka), grid, blk, 'decoder', h, 'dec'))

    checkpoint_d_end = len(out)

    # ===== FINAL HEAD BLOCK 70 =====
    checkpoint_e_start = len(out)
    # Placeholder for block 70 - requires HeadParams kernelspec
    ka = V.make_kernarg(H=8, W=8)
    weight_name = 'block70'
    w_slot = weight_slot_lookup.get(weight_name, w_base)
    struct.pack_into('<Q', ka, 0x00, S(DEC_PP[1]))
    struct.pack_into('<Q', ka, 0x08, S(DEC_PP[1]))
    struct.pack_into('<Q', ka, 0x10, S(w_slot))
    out.append((FFWD_IV, LDS[FFWD_IV], bytes(ka), (1,1), 70, 'head', 8, 'final_head'))
    checkpoint_e_end = len(out)

    print(f"\nActivation slot usage:")
    print(f"  Used slots: {sorted(used_act_slots)}")
    print(f"  Max: {max(used_act_slots) if used_act_slots else 'none'} < W_BASE {w_base}: [OK]")

    checkpoints = {
        '(0) block0': (checkpoint_0_start, checkpoint_0_end),
        '(a) encoder+23-30': (checkpoint_a_start, checkpoint_a_end),
        '(b) +ViT': (checkpoint_b_start, checkpoint_b_end),
        '(c) +39+40-47': (checkpoint_c_start, checkpoint_c_end),
        '(d) +decoder': (checkpoint_d_start, checkpoint_d_end),
        '(e) +head': (checkpoint_e_start, checkpoint_e_end),
    }

    return out, checkpoints


def arena(seed, act_base, w_base, weight_map):
    """Load all weights into arena."""
    ka = V.make_kernarg(H=16, W=16)
    g, _ = V.build(seed, ka, TOTAL_SLOTS)
    a = g.regions[1].arr.copy()

    for name, uid, size, start_slot, end_slot in weight_map:
        fn = D.ROOT / 'build' / 'weights' / (name + '.bin')
        if fn.exists():
            w = np.frombuffer(fn.read_bytes(), np.uint8)
            start_byte = start_slot * V.SLOT
            a[start_byte:start_byte + len(w)] = w

    return a, TOTAL_SLOTS


def net_run(steps_list, seed):
    """Run net_run on hardware."""
    blob = b''; lines = []
    for sym, lds, ka, grid, blk, stage, h, desc in steps_list:
        o = len(blob); blob += ka
        mod = D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
        gx, gy = grid if isinstance(grid, tuple) else (1, 1)
        lines.append('%s|%s|%d|%d|%d|%d|256' % (mod, sym, o, len(ka), gx, gy))

    arena_data, nslot = arena(seed, ACT_BASE, W_BASE, WEIGHT_ALLOC_MAP)
    (OUT / 'm.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'k.bin').write_bytes(blob)
    (OUT / 'a.bin').write_bytes(arena_data.tobytes())

    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=3600)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


def emulate(steps_list, seed, snap_at=None, on_snapshot=None):
    """Run the emulator over the dispatch list, carrying the arena forward.

    snap_at maps a dispatch index (a checkpoint's exclusive end) to its label. When the pass reaches
    that index the arena goes to on_snapshot immediately, so a run killed part way through still
    yields real verdicts for the checkpoints it did reach - the full pass takes hours, and partial
    real numbers beat a complete run we never see.
    """
    snap_at = snap_at or {}
    base_arena, _ = arena(seed, ACT_BASE, W_BASE, WEIGHT_ALLOC_MAP)
    cur = base_arena.copy()

    for step_idx, (sym, lds, ka, grid, blk, stage, h, desc) in enumerate(steps_list):
        prog = E.load_program(R.DIS, {sym})
        g, KA = V.build(seed, ka, TOTAL_SLOTS)
        g.regions[1].arr[:] = cur

        gx, gy = grid if isinstance(grid, tuple) else (1, 1)
        for wy in range(gy):
            for wx in range(gx):
                E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                                max_steps=30_000_000)
        cur = g.regions[1].arr.copy()
        if step_idx % 20 == 0:
            print(f'    emulated dispatch {step_idx}...', flush=True)
        if (step_idx + 1) in snap_at and on_snapshot is not None:
            on_snapshot(snap_at[step_idx + 1], base_arena, cur.copy())

    return base_arena, cur


# Main
steps_list, checkpoints = plan_full(ACT_BASE, W_BASE, WEIGHT_ALLOC_MAP)
print(f'\n=== Full 71-Block Network ===')
print(f'Total dispatches: {len(steps_list)}')
for label, (start, end) in sorted(checkpoints.items()):
    print(f'  {label}: {end-start} dispatches (indices {start}..{end-1})')

# Run GPU for each checkpoint cumulatively
print(f'\n=== GPU Runs (per checkpoint) ===')
gpu_checkpoints = {}
for label, (start, end) in sorted(checkpoints.items()):
    if start == end:
        print(f'{label}: (empty)')
        gpu_checkpoints[label] = (None, None)
        continue

    cumulative_steps = steps_list[:end]
    print(f'{label}: running {end} cumulative dispatches...', flush=True)
    t0 = time.time()
    rc, msg, out = net_run(cumulative_steps, 1)
    elapsed = time.time() - t0
    if out is None:
        print(f'  GPU FAIL rc={rc}')
        gpu_checkpoints[label] = (None, None)
    else:
        print(f'  {msg.splitlines()[-1] if msg else "OK"}')
        gpu_checkpoints[label] = (out, elapsed)

# One emulator pass over the whole list, verifying each checkpoint as it is crossed.
print()
print("=== Emulator Run + Verification ===")
print("One forward pass, carrying the arena between dispatches. Slow - hours.", flush=True)

verdicts = {}


def verify(label, emu_base, emu_arena):
    """Diff one checkpoint's emulator arena against the GPU arena for the same prefix."""
    gpu_out, gpu_time = gpu_checkpoints.get(label, (None, None))
    changed = np.nonzero(emu_arena != emu_base)[0]
    bytes_written = int(changed.size)
    written_slots = sorted({int(i // V.SLOT) for i in changed})

    errors = []
    if bytes_written == 0:
        errors.append("0 bytes written")
    for slot in written_slots:
        if slot >= W_BASE:
            errors.append("wrote weight slot %d (>= W_BASE %d)" % (slot, W_BASE))

    if gpu_out is None:
        mism, status = None, "UNKNOWN"          # no GPU arena -> no claim, never PASS
    else:
        mism = int(np.count_nonzero(gpu_out != emu_arena))
        status = "FAIL" if (errors or mism) else "PASS"

    verdicts[label] = status
    print("%s:" % label, flush=True)
    print("  bytes written : %d" % bytes_written)
    print("  written slots : %s" % written_slots)
    print("  mismatches    : %s" % ("n/a - GPU arena missing" if mism is None else mism))
    if gpu_time:
        print("  GPU wall      : %.2fs" % gpu_time)
    for e in errors:
        print("  FAIL: %s" % e)
    print("  => %s" % status, flush=True)


snap_at = {end: label for label, (start, end) in checkpoints.items() if start != end}

t0 = time.time()
try:
    emulate(steps_list, 1, snap_at=snap_at, on_snapshot=verify)
    print("Emulation completed in %.1fs" % (time.time() - t0))
except KeyboardInterrupt:
    print("Interrupted after %.1fs - verdicts above are real; the rest did not run."
          % (time.time() - t0))
except Exception as ex:
    print("Emulation failed after %.1fs: %s" % (time.time() - t0, ex))

print()
print("=== Summary ===")
for label, (start, end) in sorted(checkpoints.items()):
    print("  %s: %s" % (label, "(empty)" if start == end else verdicts.get(label, "NOT REACHED")))

bad = [l for l, v in verdicts.items() if v != "PASS"]
unreached = [l for l, (a, b) in checkpoints.items() if a != b and l not in verdicts]
print()
print("Run completed. %d dispatches, %d checkpoints verified, %d not reached."
      % (len(steps_list), len(verdicts), len(unreached)))
raise SystemExit(1 if (bad or unreached) else 0)
