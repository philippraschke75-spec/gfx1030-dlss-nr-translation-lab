"""Execute emitted integer instructions against explicit packed-bit answers."""
import sys
from pathlib import Path
import unittest
import numpy as np
import gfx11emu as E
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from translate_final_head import lower_pack_f16


class PackLoweringTests(unittest.TestCase):
    def check_pack(self, args, expected):
        w = E.Wave(E.GMem(), np.zeros(256, np.uint8), 0, 0)
        w.V[1] = 0xabcd302b
        w.V[2] = 0x9876b9a2
        for line in lower_pack_f16(args, 100):
            op, operands = line.split(None, 1)
            E.Exec([(0, op, operands, 4)]).compile(0)(w)
        dest = int(args.split(',')[0][1:])
        self.assertTrue(np.all(w.V[dest] == expected))

    def test_observed_bias_constant(self):
        self.check_pack('v3, v1, 1.0', 0x3c00302b)

    def test_low_half_constant(self):
        self.check_pack('v3, -0.5, v2', 0xb9a2b800)

    def test_both_constants(self):
        self.check_pack('v3, 1.0, -2.0', 0xc0003c00)

    def test_source_destination_aliases(self):
        for args in ('v1, v1, v2', 'v2, v1, v2'):
            self.check_pack(args, 0xb9a2302b)

    def test_integer_literal_stays_raw(self):
        self.check_pack('v3, v1, 1', 0x0001302b)


if __name__ == '__main__':
    unittest.main()
