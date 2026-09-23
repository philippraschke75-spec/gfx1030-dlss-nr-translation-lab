"""Show the allocation map only, without running the network."""
import sys, re, struct, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, difftest_var as D

def slots_for(size):
    """Calculate number of 1 MiB slots needed for size bytes."""
    return -(-size // V.SLOT)

BASE = len(V.PTR_FIELDS)

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

print(f"Collected {len(weight_files)} weight files")

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

# ===== DISPLAY MAP =====
print(f"\n=== Arena Allocation Map ===")
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
    print("ERRORS FOUND:")
    for err in errors:
        print(f"  ERROR: {err}")
    raise SystemExit(1)
else:
    print(f"[OK] All {len(weight_allocations)} weight allocations verified, no overlaps")
    print(f"[OK] Activation buffers [{ACT_BASE}..{W_BASE-1}] do not intersect weights")
    print(f"[OK] Total weight slots: {sum(slots_for(x[2]) for x in weight_files)}")

print(f"\nMap validated successfully!")
print(f"\nLayout Summary:")
print(f"  Pointer fields:        slots 0..17   (18 slots, BASE={BASE})")
print(f"  Activation buffers:    slots {ACT_BASE}..{W_BASE-1}   ({n_activation_slots} slots)")
print(f"  Weights:               slots {W_BASE}..{total_slots-1}   ({total_slots - W_BASE} slots)")
print(f"  =======================================")
print(f"  TOTAL:                 {total_slots} slots = {total_slots * V.SLOT / (1024*1024):.1f} MiB")
