"""Host-only regression tests for the pre-block launch fixture and the bisect harness (never dispatches).

Root cause these guard (BLOCK0_FINDINGS.md): +0x68 of the pre-block kernarg is the RNG seed (i32, ctx+0x38;
s22 -> s_mul_i32 s17, s22, 0x9e3779b9 at 0xb0550). run_var.PTR_FIELDS lists it as a pointer, so make_kernarg
writes a full 8-byte arena pointer there. Zeroing only the low dword leaves arena bits in +0x6c, the GPU
runner (var_gpu_test.cpp: any qword inside [arena_base, arena_base+size) is rebased) shifts that qword into
device space, and the kernel is seeded with a device address while the emulator's reference is seeded with 0.
"""
import os
import struct
import unittest
from unittest.mock import patch
import numpy as np
import run_var as V
import statebisect as B

REAL_POINTERS = {0x08, 0x10, 0x38, 0x40, 0x48, 0xa0}      # pre-block launch: the only qwords that are arena pointers


def arena_qwords(ka, base=None, nslots=None):
    """Offsets of the 8-byte aligned qwords the GPU runner would rebase (same rule as var_gpu_test.cpp)."""
    base = V.ARENA if base is None else base
    size = (len(V.PTR_FIELDS) if nslots is None else nslots) * V.SLOT
    out = []
    for o in range(0, len(ka) - 7, 8):
        v = struct.unpack_from('<Q', ka, o)[0]
        if base <= v < base + size:
            out.append(o)
    return out


def preblock_ctx(legacy=False):
    ctx = B.Ctx.__new__(B.Ctx)
    ctx.H = ctx.W = 8; ctx.flags = 0x14; ctx.grid = (1, 1)
    with patch.dict(os.environ, {'BISECT_LEGACY_KERNARG': '1' if legacy else '0'}):
        ctx.rebuild_kernarg()
    return ctx


class PreblockKernargTests(unittest.TestCase):
    def test_only_real_pointers_are_in_arena_range(self):
        ctx = preblock_ctx()
        self.assertEqual(set(arena_qwords(ctx.ka)), REAL_POINTERS)

    def test_seed_and_padding_are_zero(self):
        ka = preblock_ctx().ka
        self.assertEqual(struct.unpack_from('<Q', ka, 0x68)[0], 0)        # seed (+0x68) and padding (+0x6c)
        self.assertEqual(struct.unpack_from('<Q', ka, 0x00)[0], 0)
        self.assertEqual(struct.unpack_from('<Q', ka, 0x30)[0], 0)
        self.assertEqual(struct.unpack_from('<f', ka, 0x50)[0], 1.0)
        self.assertEqual(struct.unpack_from('<Q', ka, 0x58)[0], 0)
        self.assertEqual(struct.unpack_from('<Q', ka, 0x60)[0], 0)

    def test_legacy_fixture_is_detected_as_having_a_stray_pointer(self):
        """difftest_preblock.py's fixture: the stray qword is +0x68, which is what seeds the kernel on the GPU."""
        ctx = preblock_ctx(legacy=True)
        self.assertEqual(set(arena_qwords(ctx.ka)) - REAL_POINTERS, {0x68})
        self.assertEqual(struct.unpack_from('<I', ctx.ka, 0x68)[0], 0)        # low dword zero ...
        self.assertNotEqual(struct.unpack_from('<I', ctx.ka, 0x6c)[0], 0)     # ... but the high dword is not

    def test_rebasing_the_stray_qword_changes_the_seed_the_kernel_reads(self):
        ctx = preblock_ctx(legacy=True)
        dev = 0x404010000                                                   # an allocation the runner might return
        seed_emulator = struct.unpack_from('<I', ctx.ka, 0x68)[0]
        v = struct.unpack_from('<Q', ctx.ka, 0x68)[0]
        seed_gpu = (dev + (v - V.ARENA)) & 0xffffffff                       # what var_gpu_test.cpp would write there
        self.assertEqual(seed_emulator, 0)
        self.assertEqual(seed_gpu, 0x04010000)


class VariantSourceTests(unittest.TestCase):
    SRC = 'kern:\n.Lpc_10:\ns_nop 0\n.amdhsa_next_free_vgpr 5\n.amdhsa_next_free_sgpr 6\n'

    def test_first_arrival_has_no_counter(self):
        t = B.variant_source(self.SRC, 4, 0x10, 'kern')
        self.assertNotIn('s97', t)
        self.assertNotIn('.Lresume_10', t)

    def test_nth_arrival_counts_per_wave_and_restores_scc(self):
        t = B.variant_source(self.SRC, 4, 0x10, 'kern', occ=3).split('\n')
        self.assertIn('s_mov_b32 s97, 0', t)                                # zeroed at entry
        i = t.index('.Lpc_10:')
        self.assertEqual(t[i + 1], 's_cselect_b32 s101, 1, 0')              # SCC saved before anything clobbers it
        self.assertEqual(t[i + 2:i + 5], ['s_add_u32 s97, s97, 1', 's_cmp_lg_u32 s97, 3', 's_cbranch_scc1 .Lresume_10'])
        j = t.index('.Lresume_10:')
        self.assertEqual(t[j + 1], 's_cmp_lg_u32 s101, 0')                  # SCC restored ...
        self.assertEqual(t[j + 2], 's_nop 0')                               # ... then the original instruction

    def test_kernel_using_a_reserved_sgpr_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'reserves'):
            B.variant_source(self.SRC + 's_mov_b32 s97, 1\n', 4, 0x10, 'kern')
        with self.assertRaisesRegex(AssertionError, 'reserves'):
            B.variant_source(self.SRC + 's_load_dwordx2 s[96:97], s[0:1], 0x0\n', 4, 0x10, 'kern')

    def test_dump_base_can_come_from_the_output_pointer(self):
        d = B.DUMP_OFF - V.SLOT * V.PTR_FIELDS.index(0x08)                  # slot 0 == out_ptr - SLOT
        t = B.variant_source(self.SRC, 4, 0x10, 'kern', dump_src_off=0x08, dump_delta=d)
        self.assertIn('s_load_dwordx2 s[104:105], s[0:1], 0x8', t)
        self.assertIn('s_sub_u32 s104, s104, 0x80000', t)
        self.assertIn('s_subb_u32 s105, s105, 0', t)


class GetpcExplanationTests(unittest.TestCase):
    """Strict mode explains a PC-valued SGPR pair only when it is exactly a getpc return address."""

    def setUp(self):
        self.ctx = B.Ctx.__new__(B.Ctx)
        self.ctx.nreg = 1; self.ctx.hi_seen = set()
        self.ctx.prog = [(0x100, 's_getpc_b64', 's[38:39]', 4)]
        self.v = np.full((1, 32), 0x3f800000, dtype=np.uint32)
        self.s = np.full(256, B.POISON, dtype=np.uint32)
        self.s[B.E.EXEC] = 0xffffffff; self.s[B.E.VCC] = 0
        self.vec = np.zeros((8, 2, 32), dtype=np.uint32)
        self.vec[0, 0] = self.v[0]; self.vec[0, 1, 0] = B.MARK
        self.sca = np.full((8, len(B.SG_LIST)), B.POISON, dtype=np.uint32)
        self.sca[0, -3:] = [0xffffffff, 0, 0]
        self.ctx.emu_snap = lambda addr: {0: (self.v, self.s, 0)}
        self.ctx.gpu_dump = lambda addr, tag: (0, '', self.vec, self.sca)
        env = patch.dict(os.environ, {'ULP': '0', 'BISECT_HEURISTIC_FILTERS': '0', 'BISECT_PREDICATE': 'regs'})
        env.start(); self.addCleanup(env.stop)

    def set_pair(self, emu_lo, emu_hi, gpu_lo, gpu_hi, reg=38):
        self.s[reg], self.s[reg + 1] = emu_lo, emu_hi
        self.sca[0, reg], self.sca[0, reg + 1] = gpu_lo, gpu_hi

    def test_return_address_pair_is_explained_and_counted(self):
        self.set_pair(0x104, 0, 0x22104, 4)
        bad, msg = self.ctx.compare(1, 't')
        self.assertIs(bad, False)
        self.assertIn('1 getpc-return SGPR pairs explained', msg)

    def test_other_values_in_the_same_registers_are_not_hidden(self):
        self.set_pair(0xaa80, 0, 0x22080, 4)                                # a PC-derived POINTER: not a return address
        bad, msg = self.ctx.compare(1, 't')
        self.assertIs(bad, True)
        self.assertIn('s38', msg)

    def test_nonzero_high_dword_is_not_a_pc(self):
        self.set_pair(0x104, 7, 0x22104, 4)
        self.assertIs(self.ctx.compare(1, 't')[0], True)

    def test_unrelated_sgpr_difference_is_not_hidden(self):
        self.set_pair(5, 0, 6, 0, reg=2)                                    # s2 is a getpc destination elsewhere in some kernels
        self.assertIs(self.ctx.compare(1, 't')[0], True)


if __name__ == '__main__':
    unittest.main()
