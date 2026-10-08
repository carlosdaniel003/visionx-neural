"""v1.1 de diagnóstico DESLOCADO sem KNN, editor ou treinamento.

Testa imagens sintéticas em CPU e invariantes do acervo derivado.
Não substitui a validação real da fábrica.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.services.deslocado_ok_geometry_v11 import (
    SCHEMA_V11, _candidate, analyze_pair_v11, diagnose_ok_geometry_v11,
)
from src.services.deslocado_ok_geometry import _zones
from tests.test_deslocado_cnn import DeslocadoFixtures


def _features(seed=4, *, height=300, width=420):
    rng = np.random.default_rng(seed)
    base = rng.integers(35, 220, (height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), .8)
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # No textual feature in the exterior; the center is free to change.
    cv2.rectangle(image, (int(.35*width), int(.32*height)),
                  (int(.65*width), int(.75*height)), (25, 25, 25), -1)
    return image


class FeatureGeometryTests(unittest.TestCase):
    def test_registered_background_shift_can_be_measured_without_marking(self):
        reference = _features()
        h, w = reference.shape[:2]
        test = cv2.warpAffine(
            reference, np.float32([[1, 0, 4], [0, 1, -3]]),
            (w, h), borderMode=cv2.BORDER_REFLECT_101,
        )
        output = analyze_pair_v11(reference, test)
        self.assertEqual(output["status"], "METRICAS_DESCRITIVAS_DISPONIVEIS",
                         output)
        self.assertIn(output["selected_method"], ("ORB", "AKAZE"))
        dx, dy = output["global_background_shift_xy_px"]
        self.assertAlmostEqual(dx, -4.0, delta=1.4)
        self.assertAlmostEqual(dy, 3.0, delta=1.4)
        self.assertFalse(output["fixed_pads_verified"])
        self.assertIsNone(output["physical_component_shift_px"])
        self.assertIsNone(output["classification_ok_ng"])

    def test_changed_text_does_not_manufacture_ng(self):
        reference = _features(seed=10)
        test = reference.copy()
        cv2.putText(test, "22", (165, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (255, 255, 255), 4, cv2.LINE_AA)
        out = analyze_pair_v11(reference, test)
        self.assertIsNone(out["classification_ok_ng"])
        self.assertIsNone(out["physical_component_shift_px"])
        self.assertTrue(out["requires_review_for_operational_decision"])
        if out["status"] == "METRICAS_DESCRITIVAS_DISPONIVEIS":
            self.assertGreater(out["central_gray_difference"], 0)
            self.assertLess(out["peripheral_edge_difference"], .15)

    def test_only_printing_in_middle_is_not_enough_for_registration(self):
        reference = np.full((260, 320, 3), 44, dtype=np.uint8)
        test = reference.copy()
        cv2.putText(reference, "104", (113, 139), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255, 255, 255), 3)
        cv2.putText(test, "0", (135, 139), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255, 255, 255), 3)
        res = analyze_pair_v11(reference, test)
        self.assertEqual(res["status"], "EVIDENCIA_INSUFICIENTE")
        self.assertEqual(res["reason_codes"],
                         ["NO_GEOMETRICALLY_VALID_FEATURE_REGISTRATION"])
        self.assertIsNone(res["classification_ok_ng"])

    def test_large_mismatch_in_crops_rejected(self):
        a, b = _features(), _features(height=180)
        out = analyze_pair_v11(a, b)
        self.assertEqual(out["status"], "EVIDENCIA_INSUFICIENTE")
        self.assertIn("CROP_DIMENSIONS_INCOMPATIBLE", out["reason_codes"])

    def test_detector_prevents_spatially_insufficient_match(self):
        reference = _features(seed=8)
        zone = _zones(*reference.shape[:2])["outer_context"]
        blank = np.zeros_like(zone)
        result = _candidate(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
            blank, "ORB"
        )
        self.assertEqual(result["status"], "EVIDENCIA_INSUFICIENTE")
        self.assertIn("INSUFFICIENT_DISTRIBUTED_KEYPOINTS", result["reasons"])

    def test_unrelated_images_do_not_get_an_ok_class(self):
        first, second = _features(seed=11), _features(seed=12)
        res = analyze_pair_v11(first, second)
        self.assertIsNone(res["classification_ok_ng"])
        self.assertIsNone(res["physical_component_shift_px"])
        self.assertFalse(res["fixed_pads_verified"])


class OfflineV11EndToEndTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        _, run = self.prepared()
        self.manifest = run / "manifest.json"

    def test_v11_processes_complete_archive_and_writes_comparison(self):
        old_hashes = {
            f: sha256(f.read_bytes()).hexdigest()
            for f in (self.root/"public"/"ok_archive").glob("*.png")
        }
        report, out = diagnose_ok_geometry_v11(self.root, self.manifest)
        self.assertEqual(report["schema"], SCHEMA_V11)
        self.assertEqual(report["total_ok_images"], 6)
        self.assertEqual(report["total_events_candidate"], 4)
        self.assertEqual(report["by_lighting"]["SIDE"]["total"], 4)
        self.assertEqual(report["by_lighting"]["TOP"]["total"], 1)
        self.assertEqual(report["by_lighting"]["MID"]["total"], 1)
        self.assertEqual(len(report["images"]), 6)
        self.assertEqual(
            report["comparison"]["v11_metrics_available"], 0
        )
        self.assertEqual(report["comparison"]["baseline_v1_metrics_available"], 0)
        self.assertEqual(len(report["historical_false_ng_cases"]), 1)
        self.assertTrue((out/"deslocado_ok_geometry_v11.json").is_file())
        self.assertTrue((out/"deslocado_ok_geometry_v11.txt").is_file())
        self.assertTrue(Path(report["baseline_v1_report"]).is_file())
        self.assertEqual(
            old_hashes,
            {f: sha256(f.read_bytes()).hexdigest() for f in old_hashes},
        )
        self.assertIsNone(report["real_ng_recall"])
        for name in ("knn_used", "trained_model", "production_approved",
                     "production_modified", "automated_ok_decision_enabled",
                     "physical_component_shift_measured"):
            self.assertFalse(report[name])

    def test_new_ok_or_ng_fails_closed(self):
        new_ok = self.root/"public"/"ok_archive"/"2026-10-09_1111_DESLOCADO.png"
        new_ok.write_bytes(b"unprepared")
        with self.assertRaisesRegex(ValueError, "Inventário DESLOCADO mudou"):
            diagnose_ok_geometry_v11(self.root, self.manifest)
        new_ok.unlink()
        new_ng = self.root/"public"/"ng_archive"/"2026-10-09_1111_DESLOCADO.png"
        new_ng.write_bytes(b"unverified")
        with self.assertRaisesRegex(ValueError, "NG real encontrado"):
            diagnose_ok_geometry_v11(self.root, self.manifest)

    def test_original_tampering_fails_closed(self):
        file = self.root/"public"/"ok_archive"/"2026-10-02_1349_DESLOCADO.png"
        file.write_bytes(file.read_bytes()+b"tamper")
        with self.assertRaisesRegex(ValueError, "original alterado"):
            diagnose_ok_geometry_v11(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
