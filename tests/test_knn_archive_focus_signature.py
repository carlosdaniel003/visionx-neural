"""AOI visual KNN offline: caixas verdes grande/menor e gabarito x captura."""
from __future__ import annotations

import unittest
import cv2
import numpy as np

from src.services.startup_regression.knn_archive_focus_signature import (
    build_focus_signature, compare_focus_signatures, valid_focus_signature,
)


def _pair(*, defect_at=(43, 47), markers=True):
    ref = np.full((128, 128, 3), 45, dtype=np.uint8)
    test = ref.copy()
    # Componente em ambas as imagens, para que a única diferença física
    # seja o defeito simulado, e não o marcador verde da AOI.
    cv2.rectangle(ref, (25, 25), (102, 103), (110, 110, 110), -1)
    cv2.rectangle(test, (25, 25), (102, 103), (110, 110, 110), -1)
    x, y = defect_at
    cv2.rectangle(test, (x, y), (x + 12, y + 9), (255, 255, 255), -1)
    if markers:
        cv2.rectangle(test, (15, 15), (113, 113), (0, 255, 0), 2)
        cv2.rectangle(test, (35, 38), (75, 77), (0, 255, 0), 2)
    return ref, test


class ArchiveAOIEpicenterTests(unittest.TestCase):
    def test_detects_real_macro_and_micro_boxes(self):
        ref, test = _pair()
        sig = build_focus_signature(ref, test)
        self.assertTrue(valid_focus_signature(sig))
        self.assertTrue(sig["focus"]["outer_detected"])
        self.assertTrue(sig["focus"]["inner_detected"])
        self.assertIsNotNone(sig["outer_map"])
        self.assertIsNotNone(sig["inner_map"])
        self.assertLess(sig["focus"]["inner_box"][2], sig["focus"]["outer_box"][2])

    def test_similarity_prefers_same_visual_difference(self):
        a = build_focus_signature(*_pair(defect_at=(43, 47)))
        b = build_focus_signature(*_pair(defect_at=(44, 47)))
        c = build_focus_signature(*_pair(defect_at=(89, 89)))
        exact, _ = compare_focus_signatures(a, a)
        near, detail = compare_focus_signatures(a, b)
        distant, _ = compare_focus_signatures(a, c)
        self.assertAlmostEqual(exact, 1.0, places=5)
        self.assertGreater(near, distant)
        self.assertIn("inner_map", detail["groups"])

    def test_no_green_roi_is_not_fabricated(self):
        signature = build_focus_signature(*_pair(markers=False))
        self.assertFalse(signature["focus"]["outer_detected"])
        self.assertFalse(signature["focus"]["inner_detected"])
        self.assertIsNone(signature["inner_map"])
        self.assertIsNone(signature["outer_map"])
        self.assertTrue(valid_focus_signature(signature))

    def test_two_lookalike_pngs_not_compared_by_dates(self):
        ref, test = _pair()
        sig1 = build_focus_signature(ref, test)
        sig2 = build_focus_signature(ref.copy(), test.copy())
        score, _ = compare_focus_signatures(sig1, sig2)
        self.assertAlmostEqual(score, 1.0, places=5)
        self.assertNotIn("date", sig1)
        self.assertNotIn("day", sig1)


if __name__ == "__main__":
    unittest.main()
