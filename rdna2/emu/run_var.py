"""Host-only emulator run of a k_swin_var kernel with a guessed VarParams block.

Purpose: decide, without touching the GPU, whether a candidate parameter block terminates, stays
inside its buffers and produces non-degenerate output. A block that does not pass here must not be
sent to the GPU.
"""
import sys, struct, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E
import run_emu as R

SLOT = 1 << 20                      # each pointer field gets its own 1 MiB slot in the arena
ARENA = 0x7500_0000_0000
PTR_FIELDS = [0x00, 0x08, 0x10, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x78, 0x80, 0x88, 0x90, 0x98, 0xA0]


def make_kernarg(H=16, W=16, offy=-4, offx=-4, flags=0, grid=(1, 1), threads=256, ptr_fields=PTR_FIELDS):
    ka = bytearray(424)
    for k, off in enumerate(ptr_fields):
        struct.pack_into('<Q', ka, off, ARENA + k * SLOT)
    # Legacy argument names are reversed: source s26 (+0x20) contributes to
    # the X origin, s27 (+0x24) to Y (PC 0xb0360..0xb036c). Keep compatibility;
    # difftest_var exposes correctly named --offset-x/--offset-y controls.
    struct.pack_into('<iiii', ka, 0x18, H, W, offy, offx)
    struct.pack_into('<I', ka, 0x28, flags)
    # 0xA8.. are HIP hidden arguments (explicit params end at 0xA8): the runtime fills them from the launch
    struct.pack_into('<III', ka, 0xA8, grid[0], grid[1], 1)     # hidden_block_count_x/y/z
    struct.pack_into('<HHH', ka, 0xB4, threads, 1, 1)           # hidden_group_size_x/y/z
    return ka


def build(seed, ka, nslots, mode='small'):
    rng = np.random.default_rng(seed)
    g = E.GMem()
    KA = 0x7000_0000_0000
    g.add('kernarg', KA, np.frombuffer(bytes(ka), np.uint8).copy())
    e = rng.integers(2, 8, nslots * SLOT, dtype=np.uint8); m = rng.integers(0, 8, nslots * SLOT, dtype=np.uint8); s = rng.integers(0, 2, nslots * SLOT, dtype=np.uint8)
    arena = ((s << 7) | (e << 3) | m).astype(np.uint8)
    g.add('arena', ARENA, arena)
    # image .rodata (E4M3 lookup table initialised as the translator does) at its original address
    rdir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(rdir))
    import translate_final_head as base
    _, sections, _ = base.kd.parse_elf(str(base.INPUT))
    ro = next(x for x in sections if x['name'] == '.rodata')
    rodata = (rdir / 'build' / 'kernels-hw-scratch' / 'initialized-rodata.bin').read_bytes()
    assert len(rodata) == ro['size'], (len(rodata), ro['size'])
    g.add('rodata', ro['addr'], np.frombuffer(bytearray(rodata), np.uint8).copy())
    return g, KA


def run(symbol, ka, seed=1, wg=(0, 0), lds=15616, max_steps=8_000_000, trace=None):
    prog = E.load_program(R.DIS, {symbol})
    g, KA = build(seed, ka, len(PTR_FIELDS))
    init = g.regions[1].arr.copy()
    sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: wg[0], 15: wg[1]}
    t = time.time()
    r = E.run_workgroup(prog, g, lds, int(struct.unpack_from('<I', ka, 0xB4)[0]) & 0xffff, sg, max_steps=max_steps, trace=trace)
    return g, init, r, time.time() - t


if __name__ == '__main__':
    sym = sys.argv[1] if len(sys.argv) > 1 else '_Z10k_swin_varILi32ELb1EEv9VarParams'
    flags = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
    ka = make_kernarg(flags=flags)
    try:
        g, init, r, secs = run(sym, ka)
    except Exception as ex:
        print('FAILED:', type(ex).__name__, str(ex)[:300])
        raise SystemExit(2)
    a = g.regions[1]
    changed = np.nonzero(a.arr != init)[0]
    print('terminated: steps=%d in %.1fs; arena touched [%s, %s); bytes changed=%d' % (r['steps'], secs, hex(a.lo), hex(a.hi), len(changed)))
    if len(changed):
        slots = np.unique(changed // SLOT)
        print('written slots (pointer field index -> kernarg offset):', {int(k): hex(PTR_FIELDS[k]) for k in slots})
