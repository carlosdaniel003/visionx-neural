import unittest

import cv2
import numpy as np

from src.core.experts.missing_component_expert import MissingComponentExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.roi_visual_alignment import (
    best_translation,
    install_roi_visual_alignment,
)


def asymmetric_patch(width=150, height=110):
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (18, 17), (72, 83), (205, 205, 205), -1)
    cv2.circle(image, (103, 34), 16, (40, 190, 235), -1)
    cv2.line(image, (85, 87), (132, 63), (220, 80, 60), 7)
    cv2.putText(
        image,
        "R5",
        (26, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return image


def translate(image, dx, dy):
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def mean_error(first, second):
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


class ROIVisualAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_roi_visual_alignment(SilkExpert, MissingComponentExpert)

    def test_best_translation_moves_downshifted_test_up(self):
        reference = asymmetric_patch()
        test = translate(reference, 3, 11)

        aligned, shift, score, gain = best_translation(reference, test)

        self.assertLess(shift[1], -6.0)
        self.assertGreater(shift[1], -16.0)
        self.assertLess(shift[0], 0.0)
        self.assertGreater(gain, 0.018)
        self.assertGreater(score, 0.70)
        self.assertLess(mean_error(reference, aligned), mean_error(reference, test) * 0.55)

    def test_structural_expert_uses_correct_alignment_direction(self):
        reference = asymmetric_patch()
        test = translate(reference, 0, 10)
        expert = SilkExpert()

        result = expert.analyze(
            reference,
            test,
            {},
            {"category": "FALTANDO"},
            [(0, 0, reference.shape[1], reference.shape[0])],
        )

        self.assertLess(float(result["dy"]), -5.0)
        self.assertLess(
            mean_error(reference, result["roi_test_aligned"]),
            mean_error(reference, test) * 0.60,
        )
        self.assertEqual(result["roi_box"], (0, 0, 150, 110))

    def test_missing_debug_aligns_view_but_preserves_raw_input(self):
        reference = asymmetric_patch()
        test = translate(reference, 0, 9)
        expert = MissingComponentExpert()

        result = expert.analyze(
            reference,
            test,
            {},
            {"category": "FALTANDO"},
            [(0, 0, reference.shape[1], reference.shape[0])],
        )

        self.assertTrue(result["missing_visual_alignment_applied"])
        self.assertLess(float(result["missing_visual_alignment_dy"]), -4.0)
        self.assertTrue(np.array_equal(result["missing_test_input_raw"], test))
        self.assertLess(
            mean_error(reference, result["missing_test_aligned_raw"]),
            mean_error(reference, test) * 0.65,
        )
        self.assertIn("missing_test_view_raw", result)
        self.assertEqual(result["missing_display_mode"], "aligned_debug")

    def test_identical_roi_is_not_shifted(self):
        reference = asymmetric_patch()
        aligned, shift, _, gain = best_translation(reference, reference.copy())

        self.assertEqual(shift, (0.0, 0.0))
        self.assertEqual(gain, 0.0)
        self.assertTrue(np.array_equal(aligned, reference))


if __name__ == "__main__":
    unittest.main()
