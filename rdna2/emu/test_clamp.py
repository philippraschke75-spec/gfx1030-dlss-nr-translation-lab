"""Finite-value clamp regression controls; host only."""
import unittest
import numpy as np
import gfx11emu as E


class ClampTests(unittest.TestCase):
    def evaluate(self, op, args, inputs):
        w = E.Wave(E.GMem(), np.zeros(256, np.uint8), 0, 0)
        for r, value in inputs.items():
            w.V[r] = np.float32(value).view(np.uint32)
        E.Exec([(0, op, args, 8)]).compile(0)(w)
        return w.V[3].view(np.float32)[0]

    def test_fma_negative_saturates(self):
        self.assertEqual(self.evaluate('v_fma_f32', 'v3, 8.0, v3, 0.5 clamp', {3: -0.06250004}), 0)

    def test_fma_unclamped_remains_negative(self):
        self.assertLess(self.evaluate('v_fma_f32', 'v3, 8.0, v3, 0.5', {3: -0.06250004}), 0)

    def test_fma_upper_bound(self):
        self.assertEqual(self.evaluate('v_fma_f32', 'v3, 2.0, v3, 0.5 clamp', {3: 1}), 1)

    def test_mul_interior(self):
        self.assertEqual(self.evaluate('v_mul_f32_e64', 'v3, v1, v2 clamp', {1: 0.5, 2: 0.5}), 0.25)

    def test_mul_negative(self):
        self.assertEqual(self.evaluate('v_mul_f32_e64', 'v3, v1, v2 clamp', {1: -2, 2: 1}), 0)

    def test_clamped_zero_has_positive_sign(self):
        value = self.evaluate('v_mul_f32_e64', 'v3, v1, v2 clamp', {1: -0.0, 2: 1})
        self.assertEqual(int(value.view(np.uint32)), 0)


if __name__ == '__main__':
    unittest.main()
