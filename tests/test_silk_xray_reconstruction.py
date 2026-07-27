import unittest

import cv2
import numpy as np

from src.core.experts.silk_expert import SilkExpert


class SilkStructuralReconstructionTests(unittest.TestCase):
    @staticmethod
    def _make_scene() -> np.ndarray:
        image = np.full((120, 160, 3), (45, 55, 65), dtype=np.uint8)
        cv2.rectangle(image, (25, 35), (135, 85), (90, 100, 110), -1)
        cv2.rectangle(image, (35, 45), (125, 75), (30, 30, 35), -1)
        cv2.line(image, (20, 100), (140, 100), (230, 230, 230), 3)
        cv2.putText(
            image,
            "R10",
            (55, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            2,
        )
        return image

    def test_identical_images_have_empty_difference_and_matching_views(self):
        gab = self._make_scene()
        result = SilkExpert().analyze(
            gab,
            gab.copy(),
            aoi_epicenters=[(0, 0, 160, 120)],
        )

        self.assertFalse(result["is_defect"])
        self.assertLess(result["silk_error_pct"], 0.01)
        self.assertEqual(int(result["diff_mask"].max()), 0)
        self.assertEqual(result["reference_view"].shape, gab.shape)
        self.assertEqual(result["test_view"].shape, gab.shape)
        self.assertEqual(result["difference_view"].shape, gab.shape)

    def test_small_translation_is_aligned_instead_of_marking_entire_roi(self):
        gab = self._make_scene()
        transform = np.float32([[1, 0, 3], [0, 1, 2]])
        test = cv2.warpAffine(
            gab,
            transform,
            (gab.shape[1], gab.shape[0]),
            borderMode=cv2.BORDER_REFLECT101,
        )

        result = SilkExpert().analyze(
            gab,
            test,
            aoi_epicenters=[(0, 0, 160, 120)],
        )

        self.assertLess(result["silk_error_pct"], 0.03)
        self.assertNotEqual((result["dx"], result["dy"]), (0.0, 0.0))
        self.assertLess(float(np.mean(result["diff_mask"] > 0)), 0.05)

    def test_new_structure_is_classified_as_extra(self):
        gab = self._make_scene()
        test = gab.copy()
        cv2.line(test, (10, 15), (60, 15), (255, 255, 255), 3)

        result = SilkExpert().analyze(
            gab,
            test,
            aoi_epicenters=[(0, 0, 160, 120)],
        )

        self.assertGreater(result["extra_pct"], result["missing_pct"])
        self.assertGreater(cv2.countNonZero(result["extra_mask"]), 0)
        self.assertLess(float(np.mean(result["diff_mask"] > 0)), 0.25)

    def test_removed_structure_is_classified_as_missing(self):
        gab = self._make_scene()
        test = gab.copy()
        cv2.line(test, (20, 100), (140, 100), (45, 55, 65), 5)

        result = SilkExpert().analyze(
            gab,
            test,
            aoi_epicenters=[(0, 0, 160, 120)],
        )

        self.assertGreater(result["missing_pct"], result["extra_pct"])
        self.assertGreater(cv2.countNonZero(result["missing_mask"]), 0)

    def test_canonical_roi_dimensions_are_preserved(self):
        gab = self._make_scene()
        test = gab.copy()
        result = SilkExpert().analyze(
            gab,
            test,
            aoi_epicenters=[(30, 20, 78, 85)],
        )

        self.assertEqual(result["roi_box"], (30, 20, 78, 85))
        self.assertEqual(result["roi_width"], 78)
        self.assertEqual(result["roi_height"], 85)
        self.assertEqual(result["difference_view"].shape[:2], (85, 78))


if __name__ == "__main__":
    unittest.main()
