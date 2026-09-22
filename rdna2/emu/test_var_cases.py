"""Shape configuration and evidence identity; no GPU dispatch."""
import contextlib
import io
import unittest
import difftest_var as D


class VarCaseTests(unittest.TestCase):
    def test_existing_cli_defaults(self):
        a = D.parse_case(['32_1', '16,0x3f'])
        self.assertEqual((a.seed, a.gx, a.gy, a.height, a.width), (1, 2, 2, 16, 16))
        self.assertEqual(a.flag_list, [16, 63])

    def test_asymmetric_case(self):
        a = D.parse_case(['32_1', '16', '3', '1', '2', '--height', '17', '--width', '19'])
        self.assertEqual((a.gx, a.gy, a.height, a.width), (1, 2, 17, 19))

    def test_invalid_dimensions_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            D.parse_case(['32_1', '16', '--height', '0'])

    def test_evidence_identity_includes_grid_and_shape(self):
        cases = [((2, 2), 16, 16), ((1, 2), 16, 16),
                 ((2, 1), 16, 16), ((2, 2), 17, 16), ((2, 2), 16, 19)]
        self.assertEqual(len({D.case_tag('32_1', 16, 1, g, h, w) for g, h, w in cases}), 5)


if __name__ == '__main__':
    unittest.main()
