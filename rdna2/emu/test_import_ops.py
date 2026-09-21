"""Host-only checks of the emulator ops added for k_import; expected values from the ISA definitions, not from the emulator."""
import unittest
import numpy as np
import gfx11emu as E


def run(op, args, v=None, s=None, n=1):
    w = E.Wave(E.GMem(), np.zeros(256, np.uint8), 0, 0)
    for r, x in (v or {}).items(): w.V[r] = np.uint32(x & 0xffffffff)
    for r, x in (s or {}).items(): w.S[r] = x & 0xffffffff
    E.Exec([(0, op, args, 8)]).compile(0)(w)
    return w


class ImportOps(unittest.TestCase):
    def test_minmax_is_max_of_min(self):           # clamp(v0, 0, s6): max(min(v0, s6), 0)
        for x, hi, want in ((-5, 9, 0), (4, 9, 4), (50, 9, 9)):
            w = run('v_minmax_i32_e64', 'v4, v0, s6, 0', {0: x}, {6: hi})
            self.assertEqual(int(w.V[4][0].view(np.int32)), want)

    def test_s_min_max_i32_and_scc(self):
        w = run('s_min_i32', 's3, s3, s6', s={3: -1, 6: 4}); self.assertEqual((w.S[3], w.scc), (0xffffffff, 1))
        w = run('s_min_i32', 's3, s3, s6', s={3: 7, 6: 4}); self.assertEqual((w.S[3], w.scc), (4, 0))
        w = run('s_max_i32', 's3, s3, 0', s={3: -1}); self.assertEqual((w.S[3], w.scc), (0, 0))
        w = run('s_max_i32', 's3, s3, 0', s={3: 9}); self.assertEqual((w.S[3], w.scc), (9, 1))

    def test_cvt_f32_i32_and_ubyte1(self):
        w = run('v_cvt_f32_i32_e32', 'v5, v5', {5: -3}); self.assertEqual(w.V[5][0].view(np.float32), -3.0)
        w = run('v_cvt_f32_ubyte1_e32', 'v6, v0', {0: 0x00007f00}); self.assertEqual(w.V[6][0].view(np.float32), 127.0)

    def test_frexp(self):
        w = run('v_frexp_mant_f32_e32', 'v0, v4', {4: int(np.float32(12.0).view(np.uint32))})
        self.assertEqual(w.V[0][0].view(np.float32), 0.75)            # 12 = 0.75 * 2^4
        w = run('v_frexp_mant_f32_e32', 'v0, v4', {4: int(np.float32(-0.375).view(np.uint32))})
        self.assertEqual(w.V[0][0].view(np.float32), -0.75)           # -0.375 = -0.75 * 2^-1

    def test_cvt_f64_and_frexp_exp(self):
        w = run('v_cvt_f64_f32_e32', 'v[5:6], v4', {4: int(np.float32(12.0).view(np.uint32))})
        lo, hi = int(w.V[5][0]), int(w.V[6][0]); self.assertEqual(np.uint64(lo | hi << 32).view(np.float64), 12.0)
        w = run('v_frexp_exp_i32_f64_e32', 'v5, v[5:6]', {5: lo, 6: hi}); self.assertEqual(int(w.V[5][0].view(np.int32)), 4)

    def test_subrev_borrow(self):
        # D = S1 - S0 - cin ; borrow out when S1 < S0 + cin
        w = run('v_subrev_co_ci_u32_e64', 'v5, s8, 3, v5, vcc_lo', {5: 2}, {E.VCC: 0})
        self.assertEqual((int(w.V[5][0]), w.S[8] & 1), (0xffffffff, 1))
        w = run('v_subrev_co_ci_u32_e64', 'v5, s8, 1, v5, vcc_lo', {5: 5}, {E.VCC: 1})
        self.assertEqual((int(w.V[5][0]), w.S[8] & 1), (3, 0))


if __name__ == '__main__':
    unittest.main()


class MoreOps(unittest.TestCase):
    def test_s_abs_bcnt(self):
        w = run('s_abs_i32', 's1, s2', s={2: -7}); self.assertEqual((w.S[1], w.scc), (7, 1))
        w = run('s_abs_i32', 's1, s2', s={2: 0}); self.assertEqual((w.S[1], w.scc), (0, 0))
        w = run('s_bcnt1_i32_b32', 's1, s2', s={2: 0xf0f0}); self.assertEqual((w.S[1], w.scc), (8, 1))

    def test_mul24_mad24(self):
        w = run('v_mul_i32_i24_e32', 'v1, v2, v3', {2: 0xffffff, 3: 5})      # 0xffffff = -1 as signed 24-bit
        self.assertEqual(int(w.V[1][0].view(np.int32)), -5)
        w = run('v_mad_i32_i24', 'v1, v2, v3, v4', {2: 100, 3: 100, 4: 7}); self.assertEqual(int(w.V[1][0]), 10007)

    def test_max3_med3_frexp32(self):
        f = lambda x: int(np.float32(x).view(np.uint32))
        w = run('v_max3_f32', 'v1, v2, v3, v4', {2: f(1), 3: f(5), 4: f(3)}); self.assertEqual(w.V[1][0].view(np.float32), 5.0)
        w = run('v_med3_i32', 'v1, v2, v3, v4', {2: 9, 3: 0xffffffff, 4: 4}); self.assertEqual(int(w.V[1][0]), 4)
        w = run('v_frexp_exp_i32_f32_e32', 'v1, v2', {2: f(12.0)}); self.assertEqual(int(w.V[1][0].view(np.int32)), 4)

    def test_lane_ops(self):
        w = E.Wave(E.GMem(), np.zeros(256, np.uint8), 0, 0)
        w.V[2] = np.arange(32, dtype=np.uint32) * 3
        w.S[5] = 9
        E.Exec([(0, 'v_readlane_b32', 's4, v2, s5', 8)]).compile(0)(w); self.assertEqual(w.S[4], 27)
        w.S[6] = 777; w.S[7] = 3
        E.Exec([(0, 'v_writelane_b32', 'v2, s6, s7', 8)]).compile(0)(w); self.assertEqual(int(w.V[2][3]), 777)

    def test_64bit(self):
        w = E.Wave(E.GMem(), np.zeros(256, np.uint8), 0, 0)
        x = np.uint64(0x0000_00ff_0000_0000); w.V[4] = np.uint32(x & np.uint64(0xffffffff)); w.V[5] = np.uint32(x >> np.uint64(32))
        E.Exec([(0, 'v_lshrrev_b64', 'v[6:7], 8, v[4:5]', 8)]).compile(0)(w)
        self.assertEqual(int(w.V[6][0]) | int(w.V[7][0]) << 32, 0x0000_0000_ff00_0000 >> 0 if False else (0x000000ff00000000 >> 8))
        d = lambda v: np.float64(v).view(np.uint64)
        for r, v in ((4, 1.5), (6, 2.25)):
            w.V[r] = np.uint32(d(v) & np.uint64(0xffffffff)); w.V[r + 1] = np.uint32(d(v) >> np.uint64(32))
        E.Exec([(0, 'v_add_f64', 'v[8:9], v[4:5], v[6:7]', 8)]).compile(0)(w)
        self.assertEqual(np.uint64(int(w.V[8][0]) | int(w.V[9][0]) << 32).view(np.float64), 3.75)
        E.Exec([(0, 'v_cvt_f32_f64_e32', 'v10, v[8:9]', 8)]).compile(0)(w); self.assertEqual(w.V[10][0].view(np.float32), 3.75)
        w.S[E.VCC] = 0
        E.Exec([(0, 'v_cmp_le_u64_e32', 'vcc_lo, v[4:5], v[6:7]', 8)]).compile(0)(w); self.assertEqual(w.S[E.VCC] & 1, 1)   # 1.5 < 2.25 as bit patterns too

    def test_fmac_f16(self):
        h = lambda x: int(np.float16(x).view(np.uint16))
        w = run('v_fmac_f16_e32', 'v1, v2, v3', {1: h(1.0), 2: h(2.0), 3: h(3.0)}); self.assertEqual(int(w.V[1][0]), h(7.0))
