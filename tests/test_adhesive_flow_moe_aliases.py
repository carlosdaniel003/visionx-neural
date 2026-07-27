import unittest

import numpy as np

from src.core.experts.shift_expert import ShiftExpert


class AdhesiveFlowMoEAliasTests(unittest.TestCase):
    def test_adhesive_aliases_survive_later_expert_field_collisions(self):
        reference = np.full((50, 50, 3), (30, 60, 90), dtype=np.uint8)
        reference[28:36, 20:30] = (25, 35, 85)
        test = reference.copy()
        test[30:46, 20:40] = (25, 35, 85)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={"category": "Much Adhesive"},
            aoi_epicenters=[(0, 0, 50, 50)],
        )
        merged = dict(result)
        merged.update(
            {
                "comparison_mode": "structural_xor",
                "reference_view": np.zeros((3, 3, 3), dtype=np.uint8),
                "test_view": np.zeros((3, 3, 3), dtype=np.uint8),
                "dx": 999.0,
                "dy": 999.0,
            }
        )

        self.assertEqual(
            merged["adhesive_comparison_mode"],
            "adhesive_flow",
        )
        self.assertEqual(
            merged["adhesive_reference_view"].shape,
            (50, 50, 3),
        )
        self.assertEqual(
            merged["adhesive_test_view"].shape,
            (50, 50, 3),
        )
        self.assertNotEqual(merged["adhesive_dx"], 999.0)
        self.assertNotEqual(merged["adhesive_dy"], 999.0)


if __name__ == "__main__":
    unittest.main()
