"""Registry-driven difftests. Each entry records one kernel's explicit-kernarg field layout; the
shared scaffolding in kernelspec.py derives everything else (hidden-args offsets, LDS, arena size).

usage: difftest_spec.py <name> [seed]     (inside sandbox.py)
       difftest_spec.py --list
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernelspec as K

SPECS = {
    # Ported from the hand-written difftests, which these reproduce byte-for-byte.
    'ffwd2': lambda: K.Spec(
        '_Z7k_ffwd211Ffwd2Params',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3},
        # +0x28 counts 16-token groups; 4*16 == 8*8 so the whole tile is live (0 makes it a no-op)
        scalars={0x20: ('<i', 8), 0x24: ('<i', 8), 0x28: ('<i', 4)},
        weights={3: 'block23_layer0.bin'}),
    'expand': lambda: K.Spec(
        '_Z8k_expand12ExpandParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2},
        weights={2: 'block31_layer0.bin'}),
    # Same 24-byte/3-pointer shape as k_expand: s_load_b128 at +0x00 for the first two pointers,
    # s_load_b64 at +0x10 for the third, and hidden_group_size_x at +0x24. No scalar fields at all.
    'final_head': lambda: K.Spec(
        '_Z12k_final_head10HeadParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2},
        # block70 is the last of the 71 weight blocks and the only one carrying a layer0.blend_scale,
        # which is what an output blend would use - the strongest candidate for the head's weights.
        weights={2: 'block70_layer0.bin'}),
    'expand2': lambda: K.Spec(
        '_Z9k_expand212ExpandParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2},
        weights={2: 'block31_layer1.bin'}),
    'conv_res_views': lambda: K.Spec(
        '_Z16k_conv_res_views12ConvPlParams',
        pointers={0x00: 0, 0x10: 1, 0x18: 2, 0x28: 3},
        scalars={0x08: ('<Q', 0), 0x20: ('<i', 0), 0x30: ('<i', 4), 0x34: ('<i', 4),
                 0x38: ('<i', 0), 0x40: ('<i', 4), 0x44: ('<i', 4)},
        weights={3: 'block23_layer3.bin'}),

    # --- 40-byte family: s_load_b256 at +0x00 then s_load_b64 at +0x20, hidden args at +0x28 ---
    # That is five pointers and nothing else: 5*8 == 40 exactly. Treating +0x20 as an H/W scalar pair
    # instead made k_qkv fault dereferencing 0x800000088 (the packed 8,8) and made k_attention write
    # into the weight slot, so the fifth-pointer reading is what the hardware actually agrees with.
    'dec_upsample': lambda: K.Spec(              # decoder upsampling; blocks 48-69 are the decoder path
        '_Z14k_dec_upsample11DecUpParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        weights={3: 'block48_layer0.bin'}),
    'qkv': lambda: K.Spec(                       # ViT QKV; block31.layer2 is 1024*1024*3 + 128
        '_Z5k_qkv9QkvParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        weights={4: 'block31_layer2.bin'}),
    'attention': lambda: K.Spec(                 # ViT 1-D attention
        '_Z11k_attention12AttnParams1d',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        # This one strides well past its base pointers - it faulted one slot past a 6-slot arena -
        # so it needs a larger working buffer than the pointer count alone implies.
        nslot=24,
        weights={4: 'block31_layer2.bin'}),
    'contract2': lambda: K.Spec(                 # ViT FFN contract; block31.layer4 is 1024*1024 + 2048
        '_Z11k_contract212ConvParams1d',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        scalars={0x28: ('<i', 4)},
        weights={4: 'block31_layer4.bin'}),

    'qkv2': lambda: K.Spec(                      # same 5-pointer shape as k_qkv
        '_Z6k_qkv29QkvParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        weights={4: 'block31_layer2.bin'}),
    'attention2': lambda: K.Spec(                # same shape as k_attention, same large-stride need
        '_Z12k_attention212AttnParams1d',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        nslot=24,
        weights={4: 'block31_layer2.bin'}),
    'conv_splitk': lambda: K.Spec(               # same 48-byte shape as k_contract2
        '_Z13k_conv_splitk12ConvParams1d',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3, 0x20: 4},
        scalars={0x28: ('<i', 4)},
        nslot=24,                                # strides past a 6-slot arena, like k_attention
        weights={4: 'block31_layer4.bin'}),
    'ffwd_inpview': lambda: K.Spec(              # 32 B: s_load_b256 at +0x00, hidden_group_size_x at +0x2c
        '_Z14k_ffwd_inpview12FfwdPlParams',
        pointers={0x00: 0, 0x08: 1, 0x10: 2, 0x18: 3},
        weights={3: 'block23_layer1.bin'}),
    'align_probe': lambda: K.Spec(               # a single pointer
        '_Z13k_align_probePh',
        pointers={0x00: 0}),

    'repack': lambda: K.Spec(                    # 2 pointers then four i32 (+0x10/+0x14/+0x18/+0x1c)
        '_Z8k_repack12RepackParams',
        pointers={0x00: 0, 0x08: 1},
        scalars={0x10: ('<i', 8), 0x14: ('<i', 8), 0x18: ('<i', 8), 0x1c: ('<i', 8)}),
    'mean': lambda: K.Spec(                      # ptr, four i32, then a second pointer at +0x18
        '_Z6k_mean10MeanParams',
        pointers={0x00: 0, 0x18: 1},
        # A sweep shows +0x0c and +0x10 scale the step count identically (an H/W pair) while +0x08
        # and +0x14 do not affect control flow at all.
        scalars={0x08: ('<i', 8), 0x0c: ('<i', 8), 0x10: ('<i', 8), 0x14: ('<i', 8)},
        # Two fixture requirements, both learned the hard way. The input must be well-conditioned
        # f32: random bytes read as f32 span ~60 orders of magnitude and include NaNs, making the
        # summation order-dependent so emulator and hardware disagree for non-translation reasons.
        # The output must start zeroed because this kernel CAS-accumulates into it - over random
        # bytes it writes nothing at all. With both, the result scales exactly linearly with the
        # workgroup count (0.6227 at 1x1, 1.2455 at 2x1, 2.4910 at 4x1), as a cross-workgroup
        # accumulating reduction should.
        fill={0: 'f32', 1: 'zero'}),
}

if len(sys.argv) > 1 and sys.argv[1] == '--list':
    print('\n'.join(sorted(SPECS)))
    raise SystemExit(0)
name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
row = K.run_difftest(SPECS[name](), seed, tag='%s_s%d' % (name, seed))
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
