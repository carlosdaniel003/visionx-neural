import unittest

import numpy as np

from src.core.experts.ssim_expert import SSIMExpert


class SSIMFocusConsistencyTests(unittest.TestCase):
    def test_canonical_focus_preserves_source_crops_and_heatmap_resolution(self):
        gab = np.zeros((23, 41, 3), dtype=np.uint8)
        gab[:, :, 1] = 80
        test = gab.copy()
        test[6:15, 12:28] = (210, 20, 20)

        result = SSIMExpert().analyze(
            gab,
            test,
            canonical_focus=True,
            focus_box=(11, 7, 41, 23),
        )

        self.assertTrue(np.array_equal(result["crop_gab"], gab))
        self.assertTrue(np.array_equal(result["crop_test"], test))
        self.assertEqual(result["heat_map_raw"].shape, gab.shape[:2])
        self.assertEqual(result["focus_width"], 41)
        self.assertEqual(result["focus_height"], 23)
        self.assertEqual(result["focus_source"], "epicenter_extractor")
        self.assertEqual(result["focus_box"], (11, 7, 41, 23))
        self.assertTrue(result["is_epicenter"])

    def test_tiny_focus_restores_heatmap_to_original_resolution(self):
        gab = np.zeros((5, 6, 3), dtype=np.uint8)
        test = gab.copy()
        test[2:4, 2:5] = 255

        result = SSIMExpert().analyze(
            gab,
            test,
            canonical_focus=True,
            focus_box=(0, 0, 6, 5),
        )

        self.assertEqual(result["heat_map_raw"].shape, (5, 6))
        self.assertTrue(np.array_equal(result["crop_gab"], gab))
        self.assertTrue(np.array_equal(result["crop_test"], test))


if __name__ == "__main__":
    unittest.main()
