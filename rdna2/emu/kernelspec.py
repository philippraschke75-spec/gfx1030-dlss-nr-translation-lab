"""Shared kernarg/dispatch scaffolding for the per-kernel difftests.

Every difftest_<kernel>.py had the same ~70 lines of boilerplate: build a kernarg buffer, place
pointers at arena slots, fill the HSA hidden-args, run the emulator over the grid, dispatch the
translated module on the GPU, diff. The only genuinely per-kernel knowledge is the field layout,
and that is what a Spec captures.

Two things this centralizes that were repeatedly gotten wrong by hand:

* **Hidden-args offsets are read from the kernel's own metadata, not guessed.** They sit after the
  explicit struct, whose size varies per kernel, and not every kernel declares the full set.
  Guessing this is what made `k_qkv_attn` mismatch on real data for a whole session: a field
  believed to be a kernel parameter at +0x34 was really `hidden_group_size_x`.
* **A kernel that writes nothing is a failed test, not a passing one.** `k_ffwd2` dispatched
  cleanly on both the emulator and real hardware while writing zero bytes, because a count field
  was left at 0. `run_difftest` treats an empty write set as FAIL.
"""
import json, re, struct, hashlib
from pathlib import Path
import numpy as np
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D


def kernel_meta(sym):
    """Read explicit-struct size, total kernarg size and LDS bytes from the kernel's own .s file."""
    s = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
    kd = s.split('.amdhsa_kernel ' + sym, 1)[1]
    kernarg = int(re.search(r'\.amdhsa_kernarg_size (\d+)', kd)[1])
    lds = int(re.search(r'\.amdhsa_group_segment_fixed_size (\d+)', kd)[1])
    md = s.split('amdhsa.kernels:', 1)[1]
    # The explicit struct is everything before the compiler's hidden args, which is not always a
    # single by_value blob - a kernel taking a raw pointer (k_align_probe) declares a global_buffer
    # arg instead. Take the extent of every non-hidden arg.
    args = re.findall(r'\.offset:\s*(\d+)\s*\n\s*\.size:\s*(\d+)\s*\n\s*\.value_kind:\s*(\w+)', md)
    explicit = max((int(o) + int(sz) for o, sz, kind in args if not kind.startswith('hidden_')), default=0)
    # Take the hidden args exactly as declared rather than deriving them: not every kernel has the
    # full set (k_align_probe, which takes a bare pointer, declares none at all and has an 8-byte
    # kernarg), and the declaration is authoritative anyway.
    hidden = {kind[len('hidden_'):]: (int(o), int(sz)) for o, sz, kind in args if kind.startswith('hidden_')}
    return explicit, kernarg, lds, hidden


class Spec:
    """Field layout of one kernel's explicit kernarg struct.

    pointers: {offset: slot_index}  - each becomes ARENA + slot*SLOT
    scalars:  {offset: (fmt, value)} - e.g. {0x20: ('<i', 8)}
    weights:  {slot_index: filename} - real weight bytes loaded into that slot
    """

    def __init__(self, sym, pointers, scalars=None, weights=None, grid=(1, 1), threads=256, nslot=None,
                 fill=None):
        self.sym, self.pointers, self.scalars = sym, pointers, dict(scalars or {})
        self.weights, self.grid, self.threads = dict(weights or {}), grid, threads
        # fill={slot: 'f32'} puts well-conditioned floats in a slot instead of random bytes. Random
        # bytes read as f32 span ~60 orders of magnitude and include NaNs, which makes any summation
        # order-dependent - emulator and hardware then disagree for reasons that are not translation
        # bugs. Reduction kernels (k_mean) need this; byte-oriented kernels do not care.
        self.fill = dict(fill or {})
        self.explicit, self.kernarg_size, self.lds, self.hidden = kernel_meta(sym)
        need = max(list(pointers.values()) + list(self.weights)) + 1
        for slot, fn in self.weights.items():                 # a weight may span several 1 MiB slots
            n = (D.ROOT / 'build' / 'weights' / fn).stat().st_size
            need = max(need, slot + -(-n // V.SLOT))
        self.nslot = nslot or need + 1

    def kernarg(self):
        ka = bytearray(self.kernarg_size)
        for off, slot in self.pointers.items():
            struct.pack_into('<Q', ka, off, V.ARENA + slot * V.SLOT)
        for off, (fmt, val) in self.scalars.items():
            struct.pack_into(fmt, ka, off, val)
        gx, gy = self.grid
        vals = {'block_count_x': gx, 'block_count_y': gy, 'block_count_z': 1,
                'group_size_x': self.threads, 'group_size_y': 1, 'group_size_z': 1,
                'remainder_x': 0, 'remainder_y': 0, 'remainder_z': 0,
                'global_offset_x': 0, 'global_offset_y': 0, 'global_offset_z': 0, 'grid_dims': 2}
        for name, (off, size) in self.hidden.items():
            if name in vals:
                struct.pack_into('<' + {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[size], ka, off, vals[name])
        return ka

    def arena(self, seed):
        rng = np.random.default_rng(seed + 977)
        a = rng.integers(0, 256, V.SLOT * self.nslot, dtype=np.uint8)
        for slot, kind in self.fill.items():
            if kind == 'f32':
                f = rng.uniform(-1, 1, V.SLOT // 4).astype(np.float32)
                a[slot * V.SLOT:(slot + 1) * V.SLOT] = f.view(np.uint8)
            elif kind == 'zero':
                # For kernels that read indices or offsets out of a buffer: random bytes there become
                # huge offsets and fault. Zeros keep every derived address in range.
                a[slot * V.SLOT:(slot + 1) * V.SLOT] = 0
            else:
                raise ValueError(kind)
        for slot, fn in self.weights.items():
            w = np.frombuffer((D.ROOT / 'build' / 'weights' / fn).read_bytes(), np.uint8)
            a[slot * V.SLOT:slot * V.SLOT + len(w)] = w
        return a


def emulate(spec, seed, max_steps=30_000_000):
    ka = spec.kernarg()
    prog = E.load_program(R.DIS, {spec.sym})
    g, KA = V.build(seed, bytes(ka), spec.nslot)
    a = g.regions[1].arr
    a[:] = spec.arena(seed)[:len(a)]
    init = a.copy()
    steps = 0
    gx, gy = spec.grid
    for wy in range(gy):
        for wx in range(gx):
            steps += E.run_workgroup(prog, g, spec.lds, spec.threads,
                                     {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                                     max_steps=max_steps)['steps']
    return bytes(ka), init, a.copy(), steps


def run_difftest(spec, seed, tag=None):
    ka, init, ref, steps = emulate(spec, seed)
    module = D.ROOT / 'build' / 'kernels-hw-scratch' / (spec.sym + '.co')
    rc, msg, out = D.gpu(module, spec.sym, tag or (spec.sym + '_s%d' % seed), ka, init, spec.grid)
    if out is not None:                       # rebase onto the real device arena and re-reference
        V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
        ka, init, ref, steps = emulate(spec, seed)
    changed = np.nonzero(ref != init)[0]
    row = dict(kernel=spec.sym, seed=seed, grid=list(spec.grid), emu_steps=steps,
               emu_bytes_written=int(len(changed)), gpu_rc=rc, gpu=msg,
               written_slots=sorted({int(i // V.SLOT) for i in changed}))
    if out is None:
        row['status'] = 'GPU_FAIL'
    else:
        diff = np.nonzero(out != ref)[0]
        row['mismatches'] = int(len(diff))
        # An all-quiet kernel is a vacuous pass: k_ffwd2 once "passed" while writing nothing at all.
        row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
        if len(diff):
            row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
        elif not len(changed):
            row['note'] = 'kernel wrote nothing - degenerate config, test is vacuous'
        row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
    return row
