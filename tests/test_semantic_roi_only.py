import unittest

import cv2
import numpy as np

from src.core.experts.semantic_calibration import install_semantic_calibration
from src.core.experts.semantic_expert import SemanticExpert
from src.core.semantic_roi_extension import install_semantic_roi_extension


class SemanticROIOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_semantic_calibration(SemanticExpert)
        install_semantic_roi_extension(SemanticExpert)

    def setUp(self):
        self.expert = SemanticExpert()
        self.roi = [(42, 31, 28, 24)]
        self.reference = np.full((100, 130, 3), (42, 78, 110), dtype=np.uint8)
        cv2.rectangle(self.reference, (42, 31), (69, 54), (28, 31, 36), -1)
        cv2.line(self.reference, (47, 47), (64, 37), (210, 210, 205), 3)

    def test_difference_outside_roi_does_not_change_semantic_embedding(self):
        test = self.reference.copy()
        test[:28, :] = (255, 255, 255)
        test[60:, :] = (0, 0, 0)
        result = self.expert.analyze(
            self.reference,
            test,
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["semantic_active"])
        self.assertEqual(result["semantic_scope"], "epicenter_roi")
        self.assertEqual(tuple(result["semantic_focus_box"]), (42, 31, 28, 24))
        self.assertEqual(result["semantic_focus_test"].shape[:2], (24, 28))
        self.assertLess(result["semantic_loss"], 0.05)
        self.assertFalse(result["is_defect"])
        self.assertFalse(result["semantic_debug"]["full_image_fallback"])

    def test_difference_inside_roi_changes_semantic_signature(self):
        test = self.reference.copy()
        test[31:55, 42:70] = (215, 215, 215)
        cv2.circle(test, (56, 43), 8, (15, 15, 15), -1)
        result = self.expert.analyze(
            self.reference,
            test,
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["semantic_active"])
        self.assertEqual(result["semantic_debug"]["analysis_scope"], "epicenter_roi")
        self.assertGreater(result["semantic_loss"], 0.10)
        self.assertEqual(len(result["query_emb"]), 128)
        self.assertEqual(len(result["semantic_delta"]), 128)
        self.assertEqual(result["semantic_reconstruction_map"].shape, (24, 28))

    def test_missing_epicenter_abstains_instead_of_using_full_image(self):
        test = np.full_like(self.reference, 255)
        result = self.expert.analyze(
            self.reference,
            test,
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=[],
        )
        self.assertFalse(result["semantic_active"])
        self.assertFalse(result["is_defect"])
        self.assertIsNone(result["query_emb"])
        self.assertIsNone(result["semantic_focus_box"])
        self.assertIn("abstido", result["reason"].lower())


if __name__ == "__main__":
    unittest.main()
