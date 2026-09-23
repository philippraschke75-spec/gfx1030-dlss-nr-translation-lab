"""Full 71-block DLSS network as a single net_run sequence.

Order: block 0 -> 1-22 encoder -> 23-30 C=512 -> 31-38 ViT -> 39 k_dec_upsample
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

S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_full'; OUT.mkdir(parents=True, exist_ok=True)


# ===== BUILD ALLOCATION MAP =====
def build_allocation_map():
    """Build and verify the arena allocation map for all weights and activation buffers."""

    # Collect all weight file sizes
    weight_files = []

    # Encoder blocks 0-22 (23 blocks, each 1 weight)
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

    # ViT blocks 31-38 (8 blocks, 5 layers each = 40 weights, but skip layer 3)
    for blk in range(31, 39):
        for layer in [0, 1, 2, 4]:  # Skip layer 3
            fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer{layer}.bin'
            if fn.exists():
                sz = fn.stat().st_size
                weight_files.append((f'block{blk}_layer{layer}', 200+blk*10+layer, sz))

    # Block 39 (transition)
    fn = D.ROOT / 'build' / 'weights' / 'block39.bin'
    if fn.exists():
        sz = fn.stat().st_size
        weight_files.append((f'block39', 39, sz))

    fn = D.ROOT / 'build' / 'weights' / 'block39_layer0.bin'
    if fn.exists():
        sz = fn.stat().st_size
        weight_files.append((f'block39_layer0', 300+39*10+0, sz))

    # C=512 stage 2 blocks 40-47 (8 blocks, 4 layers each = 32 weights)
    for blk in range(40, 48):
        for layer in range(4):
            fn = D.ROOT / 'build' / 'weights' / f'block{blk}_layer{layer}.bin'
            if fn.exists():
                sz = fn.stat().st_size
                weight_files.append((f'block{blk}_layer{layer}', 300+blk*10+layer, sz))

    # Decoder blocks 48-69 (22 blocks, 1-2 weights each)
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
    fn = D.ROOT / 'build' / 'weights' / 'block70.bin'
    if fn.exists():
        sz = fn.stat().st_size
        weight_files.append((f'block70', 70, sz))

    fn = D.ROOT / 'build' / 'weights' / 'block70_layer0.bin'
    if fn.exists():
        sz = fn.stat().st_size
        weight_files.append((f'block70_layer0', 400+70*10+0, sz))

    fn = D.ROOT / 'build' / 'weights' / 'block70_layer0_blend_scale.bin'
    if fn.exists():
        sz = fn.stat().st_size
        weight_files.append((f'block70_layer0_blend_scale', 400+70*10+1, sz))

    # Fixed activation buffer allocation
    # Encoder: ENC_PP (2 slots for pingpong) + ENC_IN (1) = 3 activation slots
    # C512_1: C512_1_WORK (4) = 4 activation slots
    # VIT: VIT_WORK (6) = 6 activation slots
    # C512_2: C512_2_WORK (4) = 4 activation slots
    # Decoder: DEC_PP (2 for pingpong) + DEC_IN (1) = 3 activation slots
    n_activation_slots = 3 + 4 + 6 + 4 + 3  # = 20

    ACT_BASE = BASE  # 18
    W_BASE = ACT_BASE + n_activation_slots  # 18 + 20 = 38

    # Allocate weights
    weight_allocations = []
    w_slot = W_BASE
    for name, uid, size in sorted(weight_files, key=lambda x: (-x[2], x[0])):  # sort by size descending
        num_slots = slots_for(size)
        weight_allocations.append((name, uid, size, w_slot, w_slot + num_slots))
        w_slot += num_slots

    total_slots = w_slot

    # ===== VERIFICATION =====
    print(f"=== Arena Allocation Map ===")
    print(f"Slots 0..17:        VarParams pointer fields (BASE={BASE})")
    print(f"ACT_BASE = {ACT_BASE}:  Activation/working buffers ({n_activation_slots} slots)")
    print(f"  - ENC_PP (2), ENC_IN (1)")
    print(f"  - C512_1_WORK (4)")
    print(f"  - VIT_WORK (6)")
    print(f"  - C512_2_WORK (4)")
    print(f"  - DEC_PP (2), DEC_IN (1)")
    print(f"W_BASE = {W_BASE}:  Weights (allocated per size)")
    print(f"Total slots needed: {total_slots}")
    print(f"Total arena size: {total_slots * V.SLOT / (1024*1024):.1f} MiB")

    # Find 5 largest weights
    sorted_alloc = sorted(weight_allocations, key=lambda x: -x[2])
    print(f"\n5 largest weights:")
    for i, (name, uid, size, start, end) in enumerate(sorted_alloc[:5]):
        print(f"  {i+1}. {name:30} {size:>10} B ({slots_for(size)} slots) [slot {start}..{end-1}]")

    # Verify no overlaps
    print(f"\n=== Verification ===")
    errors = []

    # Check activation buffer range
    act_end = ACT_BASE + n_activation_slots
    if act_end != W_BASE:
        errors.append(f"MISMATCH: ACT_BASE+n_activation_slots={act_end} != W_BASE={W_BASE}")

    # Check no weight overlaps
    for i, (n1, u1, sz1, s1, e1) in enumerate(weight_allocations):
        for j, (n2, u2, sz2, s2, e2) in enumerate(weight_allocations):
            if i < j and not (e1 <= s2 or e2 <= s1):
                errors.append(f"OVERLAP: {n1} [slot {s1}..{e1-1}] overlaps {n2} [slot {s2}..{e2-1}]")

    # Check no weight intersects activation range
    for name, uid, size, start, end in weight_allocations:
        if not (end <= ACT_BASE or start >= W_BASE):
            errors.append(f"VIOLATION: {name} [slot {start}..{end-1}] intersects activation range [slot {ACT_BASE}..{W_BASE-1}]")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    else:
        print(f"✓ All {len(weight_allocations)} weight allocations verified, no overlaps")
        print(f"✓ Activation buffers [{ACT_BASE}..{W_BASE-1}] do not intersect weights")
        print(f"✓ Total weight slots: {sum(slots_for(x[2]) for x in weight_files)}")

    return ACT_BASE, W_BASE, total_slots, weight_allocations


# Build and verify the map first
ACT_BASE, W_BASE, TOTAL_SLOTS, WEIGHT_ALLOC_MAP = build_allocation_map()
print(f"\nMap validated. Ready to proceed.\n")


def plan_full(act_base, w_base, weight_map):
    """Generate all dispatches for 71-block network."""
    # Build a lookup for weight slot indices
    weight_slot_lookup = {}
    for name, uid, size, start, end in weight_map:
        weight_slot_lookup[name] = start

    out = []
    wi = 0

    # Local allocation offsets relative to ACT_BASE
    enc_pp_0 = act_base + 0
    enc_pp_1 = act_base + 1
    enc_in = act_base + 2

    # ===== ENCODER STAGE 1-4 (blocks 0-22) =====
    ENCODER_STAGES = [('32_0', [1,2,3,4]), ('64_0', [5,6,7,8]),
                      ('128_0', [9,10,11,12,13,14]), ('256_0', [15,16,17,18,19,20,21,22])]
    ENCODER_MODES = [(0,0), (-4,-4), (-4,0), (0,-4)]
    POOL = 0x38

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
            wi += 1

    encoder_wi = wi

    # ===== C=512 STAGE 1 (blocks 23-30) =====
    c512_1_work = [act_base + 3, act_base + 4, act_base + 5, act_base + 6]
    for blk_idx, blk in enumerate(range(23, 31)):
        # 1. k_ffwd_inpview
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer0'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(enc_in if blk_idx == 0 else c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(c512_1_work[1]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        out.append((FFWD_IV, 0, bytes(ka), (1,1), blk, 'c512_1', 8, 'ffwd_iv'))
        wi += 1

        # 2. k_ffwd2
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer1'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(enc_in if blk_idx == 0 else c512_1_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_1_work[1]))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<iii', ka, 0x20, 8, 8, 4)
        out.append((FFWD2, 0, bytes(ka), (1,1), blk, 'c512_1', 8, 'ffwd2'))
        wi += 1

        # 3. k_conv_res_views
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
        out.append((CONVV, 0, bytes(ka), (1,1), blk, 'c512_1', 8, 'conv_res_1'))
        wi += 1

        # 4. k_qkv_attn
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer3'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[2]))
        struct.pack_into('<Q', ka, 0x08, S(c512_1_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        struct.pack_into('<ii', ka, 0x20, 0, 0)
        out.append((QKV, 0, bytes(ka), (1,1), blk, 'c512_1', 8, 'qkv_attn'))
        wi += 1

        # 5. k_conv_res_views
        ka = V.make_kernarg(H=8, W=8)
        struct.pack_into('<Q', ka, 0x00, S(c512_1_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(c512_1_work[2]))
        struct.pack_into('<Q', ka, 0x18, S(c512_1_work[0]))
        struct.pack_into('<i', ka, 0x20, 0)
        # This dispatch has no weight (residual add)
        struct.pack_into('<Q', ka, 0x28, S(w_base))  # Dummy slot beyond weights
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        struct.pack_into('<ii', ka, 0x40, 8, 8)
        out.append((CONVV, 0, bytes(ka), (1,1), blk, 'c512_1', 8, 'conv_res_2'))
        wi += 1

    c512_1_wi = wi - encoder_wi

    # ===== ViT BLOCKS 31-38 =====
    vit_work = [act_base + 7, act_base + 8, act_base + 9, act_base + 10, act_base + 11, act_base + 12]
    B268, B270, B278, B280, B288, BOUT = vit_work
    for blk in range(31, 39):
        # 1. k_expand2
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
        wi += 1

        # 2. k_contract2
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
        wi += 1

        # 3. k_qkv2
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
        wi += 1

        # 4. k_attention2
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
        wi += 1

        # 5. k_contract2
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
        wi += 1

    vit_wi = wi - encoder_wi - c512_1_wi

    # ===== BLOCK 39: k_dec_upsample transition =====
    # TODO: Placeholder - requires kernelspec for DecUpParams
    # For now, skip and note it

    # ===== C=512 STAGE 2 (blocks 40-47) - same as stage 1 =====
    c512_2_work = [act_base + 13, act_base + 14, act_base + 15, act_base + 16]
    for blk_idx, blk in enumerate(range(40, 48)):
        # 1. k_ffwd_inpview
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer0'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(vit_work[5] if blk_idx == 0 else c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(c512_2_work[1]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        out.append((FFWD_IV, 0, bytes(ka), (1,1), blk, 'c512_2', 8, 'ffwd_iv'))
        wi += 1

        # 2. k_ffwd2
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer1'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x08, S(vit_work[5] if blk_idx == 0 else c512_2_work[0]))
        struct.pack_into('<Q', ka, 0x10, S(c512_2_work[1]))
        struct.pack_into('<Q', ka, 0x18, S(w_slot))
        struct.pack_into('<iii', ka, 0x20, 8, 8, 4)
        out.append((FFWD2, 0, bytes(ka), (1,1), blk, 'c512_2', 8, 'ffwd2'))
        wi += 1

        # 3. k_conv_res_views
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
        out.append((CONVV, 0, bytes(ka), (1,1), blk, 'c512_2', 8, 'conv_res_1'))
        wi += 1

        # 4. k_qkv_attn
        ka = V.make_kernarg(H=8, W=8)
        weight_name = f'block{blk}_layer3'
        w_slot = weight_slot_lookup[weight_name]
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[2]))
        struct.pack_into('<Q', ka, 0x08, S(c512_2_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(w_slot))
        struct.pack_into('<ii', ka, 0x18, 8, 8)
        struct.pack_into('<ii', ka, 0x20, 0, 0)
        out.append((QKV, 0, bytes(ka), (1,1), blk, 'c512_2', 8, 'qkv_attn'))
        wi += 1

        # 5. k_conv_res_views
        ka = V.make_kernarg(H=8, W=8)
        struct.pack_into('<Q', ka, 0x00, S(c512_2_work[3]))
        struct.pack_into('<Q', ka, 0x10, S(c512_2_work[2]))
        struct.pack_into('<Q', ka, 0x18, S(c512_2_work[0]))
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<Q', ka, 0x28, S(w_base))  # Dummy slot
        struct.pack_into('<ii', ka, 0x30, 8, 8)
        struct.pack_into('<ii', ka, 0x40, 8, 8)
        out.append((CONVV, 0, bytes(ka), (1,1), blk, 'c512_2', 8, 'conv_res_2'))
        wi += 1

    c512_2_wi = 8 * 5

    # ===== DECODER STAGES (blocks 48-69) - mirror of encoder =====
    DECODER_STAGES = [('256_0', list(range(48, 56))), ('128_0', list(range(56, 62))),
                      ('64_0', list(range(62, 66))), ('32_0', list(range(66, 70)))]
    DEC_PP = [act_base + 17, act_base + 18]
    DEC_IN = act_base + 19

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
            wi += 1

    decoder_wi = wi - encoder_wi - c512_1_wi - vit_wi - c512_2_wi

    # ===== FINAL HEAD BLOCK 70 =====
    # TODO: Placeholder - requires HeadParams kernelspec
    # Single dispatch to final head kernel

    return out, wi


def arena(seed, act_base, w_base, weight_map):
    """Load all weights into arena."""
    steps_list, total_wi = plan_full(act_base, w_base, weight_map)
    g, _ = V.build(seed, steps_list[0][2], TOTAL_SLOTS)
    a = g.regions[1].arr.copy()

    # Load weights according to the allocation map
    for name, uid, size, start_slot, end_slot in weight_map:
        # Reconstruct the filename
        fn = D.ROOT / 'build' / 'weights' / (name + '.bin')
        if fn.exists():
            w = np.frombuffer(fn.read_bytes(), np.uint8)
            start_byte = start_slot * V.SLOT
            a[start_byte:start_byte + len(w)] = w

    return a, TOTAL_SLOTS


def net_run(seed):
    """Run net_run on hardware."""
    steps_list, _ = plan_full(ACT_BASE, W_BASE, WEIGHT_ALLOC_MAP)
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


# Main
steps_list, total_wi = plan_full(ACT_BASE, W_BASE, WEIGHT_ALLOC_MAP)
print('=== Full 71-Block Network ===')
print('Total dispatches:', len(steps_list))

rc, msg, out = net_run(1)
print('\n%s' % (msg.splitlines()[-1] if msg else 'no output'))
if out is None:
    print('GPU FAIL rc=%d' % rc)
    raise SystemExit(1)

print('\nRun completed. Dispatches: %d' % len(steps_list))
raise SystemExit(0)
