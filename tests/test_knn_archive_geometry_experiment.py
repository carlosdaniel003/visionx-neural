"""Regras da comparação A/B do descritor geométrico no acervo AOI."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.services.startup_regression.knn_archive_focus_signature import (
    build_focus_signature,
)
from src.services.startup_regression.knn_archive_geometry_experiment import (
    build_geometry_signature, compare_geometry_signatures,
    compare_geometry_hybrid,
)


def simulated_pair(offset=(4, 0), *, marks=True):
    ref = np.full((144, 144, 3), 33, dtype=np.uint8)
    test = ref.copy()
    cv2.rectangle(ref, (50, 57), (88, 79), (220, 220, 220), -1)
    dx, dy = offset
    cv2.rectangle(test, (50 + dx, 57 + dy), (88 + dx, 79 + dy),
                  (220, 220, 220), -1)
    if marks:
        cv2.rectangle(test, (15, 20), (130, 128), (0, 255, 0), 2)
        cv2.rectangle(test, (42, 47), (103, 91), (0, 255, 0), 2)
    return ref, test


class KNNArchiveGeometryExperimentalTests(unittest.TestCase):
    def test_geometry_is_symmetric_and_uses_actual_boxes(self):
        ref, test = simulated_pair(offset=(8, 0))
        focus = build_focus_signature(ref, test)
        geom = build_geometry_signature(ref, test, focus["focus"])
        self.assertIsNotNone(geom["scales"]["inner"])
        self.assertIsNotNone(geom["scales"]["outer"])
        self.assertGreater(geom["scales"]["inner"]["added_stats"][0], 0)
        same, _ = compare_geometry_signatures(geom, geom)
        self.assertGreater(same, .99)
        combined, detail = compare_geometry_hybrid(focus, focus, geom, geom)
        self.assertGreater(combined, .99)
        self.assertEqual(detail["geometry_weight"], .35)

    def test_direction_of_component_shift_contributes_to_similarity(self):
        base = simulated_pair(offset=(10, 0))
        similar = simulated_pair(offset=(9, 0))
        opposite = simulated_pair(offset=(-10, 0))
        signatures = []
        for ref, obs in (base, similar, opposite):
            focus = build_focus_signature(ref, obs)
            signatures.append((
                focus, build_geometry_signature(ref, obs, focus["focus"])
            ))
        q, qgeo = signatures[0]
        n, ngeo = signatures[1]
        far, fgeo = signatures[2]
        near_score, _ = compare_geometry_hybrid(q, n, qgeo, ngeo)
        far_score, _ = compare_geometry_hybrid(q, far, qgeo, fgeo)
        self.assertGreater(near_score, far_score)

    def test_missing_boxes_never_fabricates_epicenter(self):
        ref, test = simulated_pair(offset=(3, 0), marks=False)
        focus = build_focus_signature(ref, test)
        self.assertFalse(focus["focus"]["inner_detected"])
        result = build_geometry_signature(ref, test, focus["focus"])
        self.assertIsNone(result["scales"]["outer"])
        self.assertIsNone(result["scales"]["inner"])

    def test_invalid_shape_or_unsupported_signature_rejected(self):
        ref, test = simulated_pair()
        focus = build_focus_signature(ref, test)
        with self.assertRaisesRegex(ValueError, "Par AOI"):
            build_geometry_signature(np.zeros((1, 1, 3), dtype=np.uint8),
                                     test, focus["focus"])
        signature = build_geometry_signature(ref, test, focus["focus"])
        with self.assertRaisesRegex(ValueError, "incompatíveis"):
            compare_geometry_signatures(signature, {"schema": "other"})


if __name__ == "__main__":
    unittest.main()
