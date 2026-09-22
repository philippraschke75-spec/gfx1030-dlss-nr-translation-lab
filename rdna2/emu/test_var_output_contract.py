import unittest
from var_output_contract import audit, trunc_half


class OutputContractTests(unittest.TestCase):
    def test_observed_collision(self):
        r = audit(7, 9)
        c = next(x for x in r['collisions'] if x['offset'] == 0x30)
        self.assertEqual([(x['wave'], x['lane'], x['iteration']) for x in c['writers']],
                         [(1, 16, 0), (5, 0, 1)])
        self.assertGreater(r['negative_coordinates'], 0)
        self.assertGreater(r['before_base_bytes'], 0)

    def test_zero_origin_ownership(self):
        for h, w, grid in [(7, 9, (1, 1)), (17, 19, (2, 1)), (17, 19, (1, 2))]:
            r = audit(h, w, grid, 0, 0)
            self.assertEqual((r['collision_bytes'], r['negative_coordinates'], r['before_base_bytes']), (0, 0, 0))

    def test_signed_division_truncates_toward_zero(self):
        self.assertEqual([trunc_half(n) for n in (-5, -4, -1, 0, 1, 5)], [-2, -2, 0, 0, 0, 2])
