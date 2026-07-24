import unittest

import cv2
import numpy as np

from src.core.experts.ssim_expert import SSIMExpert


class SSIMAdhesiveHeatmapTests(unittest.TestCase):
    @staticmethod
    def _make_scene() -> np.ndarray:
        """ROI sintética com cobre, lateral metálica e corpo do componente."""
        image = np.zeros((120, 100, 3), dtype=np.uint8)
        image[:] = (35, 70, 100)

        # Padding de cobre laranja.
        image[60:110, 15:85] = (20, 105, 205)

        # Lateral prata e corpo escuro do componente.
        image[20:85, 38:62] = (190, 190, 195)
        image[30:75, 42:58] = (60, 60, 70)
        return image

    def test_adhesive_map_preserves_roi_and_ignores_stable_materials(self):
        gab = self._make_scene()
        test = gab.copy()
        adhesive_region = (slice(72, 105), slice(42, 72))
        test[adhesive_region] = (25, 35, 85)

        result = SSIMExpert().analyze(
            gab,
            test,
            canonical_focus=True,
            focus_box=(0, 0, 100, 120),
        )
        heat = result["heat_map_raw"]

        stable_copper = heat[70:105, 15:35]
        stable_component = heat[25:65, 40:60]
        adhesive = heat[adhesive_region]

        self.assertEqual(heat.shape, gab.shape[:2])
        self.assertEqual(result["heat_map_mode"], "adhesive_excess")
        self.assertGreater(float(adhesive.mean()), float(stable_copper.mean()) + 50.0)
        self.assertGreater(float(adhesive.mean()), float(stable_component.mean()) + 50.0)
        self.assertGreater(result["adhesive_coverage"], 0.001)
        self.assertIsNotNone(result["adhesive_centroid"])

    def test_alignment_suppresses_shifted_component_and_keeps_adhesive(self):
        gab = self._make_scene()
        transform = np.float32([[1, 0, 4], [0, 1, 2]])
        test = cv2.warpAffine(
            gab,
            transform,
            (gab.shape[1], gab.shape[0]),
            borderMode=cv2.BORDER_REFLECT101,
        )
        adhesive_region = (slice(76, 108), slice(46, 76))
        test[adhesive_region] = (25, 35, 85)

        result = SSIMExpert().analyze(
            gab,
            test,
            canonical_focus=True,
            focus_box=(0, 0, 100, 120),
        )
        heat = result["heat_map_raw"]

        shifted_component = heat[20:70, 35:70]
        adhesive = heat[adhesive_region]

        self.assertEqual(result["heat_map_mode"], "adhesive_excess")
        self.assertGreater(float(adhesive.mean()), float(shifted_component.mean()) + 40.0)
        self.assertNotEqual(result["alignment_shift"], (0.0, 0.0))

    def test_identical_roi_produces_no_adhesive_heat(self):
        gab = self._make_scene()
        result = SSIMExpert().analyze(
            gab,
            gab.copy(),
            canonical_focus=True,
            focus_box=(0, 0, 100, 120),
        )

        self.assertEqual(int(result["heat_map_raw"].max()), 0)
        self.assertEqual(result["adhesive_coverage"], 0.0)
        self.assertEqual(result["heat_map_mode"], "generic_ssim")

    def test_non_warm_difference_keeps_generic_ssim_map(self):
        gab = np.full((60, 60, 3), (50, 80, 120), dtype=np.uint8)
        test = gab.copy()
        test[20:40, 20:40] = (200, 20, 20)

        result = SSIMExpert().analyze(
            gab,
            test,
            canonical_focus=True,
            focus_box=(0, 0, 60, 60),
        )

        self.assertEqual(result["heat_map_mode"], "generic_ssim")
        self.assertEqual(result["heat_map_raw"].shape, (60, 60))


if __name__ == "__main__":
    unittest.main()
