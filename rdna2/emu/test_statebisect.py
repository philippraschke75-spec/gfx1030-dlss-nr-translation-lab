"""Host-only negative controls for the checkpoint diagnostic; never dispatches."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import statebisect as B


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.ctx = B.Ctx.__new__(B.Ctx)
        self.ctx.nreg = 1
        self.ctx.hi_seen = set()
        self.events = []
        self.v = np.full((1, 32), 0x3f800000, dtype=np.uint32)
        self.s = np.full(256, B.POISON, dtype=np.uint32)
        self.s[B.E.EXEC] = 0xffffffff
        self.s[B.E.VCC] = 0
        self.vec = np.zeros((8, 2, 32), dtype=np.uint32)
        self.vec[0, 0] = self.v[0]
        self.vec[0, 1, 0] = B.MARK
        self.sca = np.full((8, len(B.SG_LIST)), B.POISON, dtype=np.uint32)
        self.sca[0, -3:] = [0xffffffff, 0, 0]
        def snap(addr):
            self.events.append('emu')
            return {0: (self.v, self.s, 0)}
        def gpu(addr, tag):
            self.events.append('gpu')
            return 0, '', self.vec, self.sca
        self.ctx.emu_snap = snap
        self.ctx.gpu_dump = gpu
        self.env = patch.dict(os.environ, {'ULP': '0', 'BISECT_HEURISTIC_FILTERS': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_allocated_address_validated_before_checkpoint(self):
        ctx = self.ctx
        ctx.ka = b'actual'
        ctx.adopt = lambda dev: self.events.append(('adopt', dev))
        arena = np.zeros(8, np.uint8)
        ctx.arena = lambda: (SimpleNamespace(regions=[None, SimpleNamespace(arr=arena)]), 0)
        ctx.validate_allocation(1, arena.copy(), 0x1404010000, b'actual')
        self.assertEqual(self.events, [('adopt', 0x1404010000), 'emu'])
        with self.assertRaisesRegex(RuntimeError, 'kernarg'):
            ctx.validate_allocation(1, arena, 4, b'wrong')
        with self.assertRaisesRegex(RuntimeError, 'input arena'):
            ctx.validate_allocation(1, np.ones(8, np.uint8), 4, b'actual')
        ctx.emu_snap = lambda addr: {}
        with self.assertRaisesRegex(RuntimeError, 'unreachable'):
            ctx.validate_allocation(1, arena, 4, b'actual')

    def test_preflight_before_dispatch_and_pointer_refresh(self):
        self.assertIs(self.ctx.compare(1, 'test')[0], False)
        self.assertEqual(self.events, ['emu', 'gpu', 'emu'])

    def test_fault_prevents_dispatch(self):
        def fault(addr):
            raise RuntimeError('memory fault')
        self.ctx.emu_snap = fault
        with self.assertRaises(RuntimeError):
            self.ctx.compare(1, 'test')
        self.assertEqual(self.events, [])

    def test_unreachable_prevents_dispatch(self):
        self.ctx.emu_snap = lambda addr: {}
        self.assertIsNone(self.ctx.compare(1, 'test')[0])
        self.assertEqual(self.events, [])

    def test_upper_half_difference_is_not_hidden(self):
        self.vec[0, 0, 0] ^= np.uint32(0x10000)
        self.assertIs(self.ctx.compare(1, 'test')[0], True)

    def test_pointer_like_integer_is_not_hidden(self):
        self.v[0, 0] = 0x9000
        self.vec[0, 0, 0] = 0x9001
        self.assertIs(self.ctx.compare(1, 'test')[0], True)

    def test_exec_mask_two_is_compared(self):
        self.s[B.E.EXEC] = 2
        self.assertIs(self.ctx.compare(1, 'test')[0], True)

    def test_ulp_requires_explicit_exploratory_mode_before_dispatch(self):
        with patch.dict(os.environ, {'ULP': '4'}):
            with self.assertRaises(ValueError):
                self.ctx.compare(1, 'test')
        self.assertEqual(self.events, [])

    def test_gpu_failure_is_not_match(self):
        self.ctx.gpu_dump = lambda addr, tag: (124, 'timeout', None, None)
        self.assertIsNone(self.ctx.compare(1, 'test')[0])


if __name__ == '__main__':
    unittest.main()
