"""Regressão do diagnóstico automático DESLOCADO OK, sem máscaras manuais."""
from __future__ import annotations

from hashlib import sha256
import json

import cv2
import numpy as np

from src.services.deslocado_ok_geometry import (
    SCHEMA, analyze_pair, diagnose_ok_geometry,
)
from tests.test_deslocado_cnn import DeslocadoFixtures


class DeslocadoOKGeometryTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        _, self.run = self.prepared()
        self.manifest = self.run / "manifest.json"

    def test_all_ok_images_and_triplet_without_drawing(self):
        hashes = {
            file: sha256(file.read_bytes()).hexdigest()
            for file in (self.root / "public" / "ok_archive").glob("*.png")
        }
        report, output = diagnose_ok_geometry(self.root, self.manifest)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["total_ok_images"], 6)
        self.assertEqual(report["total_candidate_events"], 4)
        self.assertEqual(report["source_lights"], {"SIDE": 4, "TOP": 1, "MID": 1})
        self.assertEqual(len(report["images"]), 6)
        self.assertEqual(len({row["source_path"] for row in report["images"]}), 6)
        self.assertEqual(report["candidate_event_links_unverified"], 1)
        self.assertEqual(report["historical_false_ng_cases_present"], 1)
        self.assertTrue((output / "deslocado_ok_geometry.json").is_file())
        self.assertTrue((output / "deslocado_ok_geometry.txt").is_file())
        self.assertEqual(hashes, {
            file: sha256(file.read_bytes()).hexdigest() for file in hashes
        })

    def test_every_result_stays_descriptive_without_ng_guess(self):
        report, _ = diagnose_ok_geometry(self.root, self.manifest)
        for item in report["images"]:
            self.assertIn(item["evidence_status"], (
                "EVIDENCIA_INSUFICIENTE",
                "METRICAS_DESCRITIVAS_DISPONIVEIS",
            ))
            self.assertIsNone(item["detected_real_ng"])
            self.assertFalse(item["fixed_pads_verified"])
            self.assertFalse(item["component_segmented"])
            self.assertTrue(item["requires_review_for_operational_decision"])
        self.assertFalse(report["trained_model"])
        self.assertFalse(report["knn_used"])
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["production_modified"])
        self.assertFalse(report["automated_ok_decision_enabled"])
        self.assertIsNone(report["real_ng_recall"])

    def test_center_text_change_does_not_generate_ng_label(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, (145, 171, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (7, 7), 0)
        different_text = image.copy()
        cv2.putText(different_text, "101", (61, 81),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 2)
        result = analyze_pair(image, different_text)
        self.assertIsNone(result["detected_real_ng"])
        self.assertFalse(result["fixed_pads_verified"])
        self.assertIn(result["evidence_status"], (
            "EVIDENCIA_INSUFICIENTE",
            "METRICAS_DESCRITIVAS_DISPONIVEIS",
        ))
        if result["central_gray_difference"] is not None:
            self.assertGreater(result["central_gray_difference"], 0)

    def test_low_context_has_insufficient_evidence(self):
        blank = np.zeros((130, 140, 3), np.uint8)
        cv2.rectangle(blank, (55, 43), (85, 88), (255, 255, 255), 2)
        result = analyze_pair(blank, blank.copy())
        self.assertEqual(result["evidence_status"], "EVIDENCIA_INSUFICIENTE")
        self.assertIn("WEAK_OR_LOCALIZED_BACKGROUND_ANCHORS", result["reasons"])

    def test_unequal_crops_do_not_silently_scale_large_changes(self):
        a = np.zeros((100, 140, 3), np.uint8)
        b = np.zeros((140, 140, 3), np.uint8)
        report = analyze_pair(a, b)
        self.assertEqual(report["evidence_status"], "EVIDENCIA_INSUFICIENTE")
        self.assertIn("CROP_DIMENSIONS_INCOMPATIBLE", report["reasons"])
        self.assertFalse(report["resized_test_for_comparison"])

    def test_new_ok_requires_new_inventory_run(self):
        (self.root / "public" / "ok_archive" /
         "2026-10-08_0801_DESLOCADO_SIDE.png").write_bytes(b"\x89PNG")
        with self.assertRaisesRegex(ValueError, "Inventário DESLOCADO mudou"):
            diagnose_ok_geometry(self.root, self.manifest)

    def test_real_ng_blocks_ok_only_protocol(self):
        (self.root / "public" / "ng_archive" /
         "2026-10-08_0801_DESLOCADO_SIDE.png").write_bytes(b"\x89PNG")
        with self.assertRaisesRegex(ValueError, "NG real encontrado"):
            diagnose_ok_geometry(self.root, self.manifest)

    def test_source_tampering_blocks_diagnostics(self):
        file = self.root / "public" / "ok_archive" / "2026-10-02_1349_DESLOCADO.png"
        file.write_bytes(file.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "original alterado"):
            diagnose_ok_geometry(self.root, self.manifest)

    def test_manifest_extracted_pair_missing_blocks_diagnostics(self):
        info = json.loads(self.manifest.read_text(encoding="utf-8"))
        first = info["samples"][0]
        extracted = self.run / first["test_path"]
        extracted.unlink()
        with self.assertRaisesRegex(ValueError, "Par extraído"):
            diagnose_ok_geometry(self.root, self.manifest)

    def test_split_group_key_is_not_treated_as_verified_event(self):
        report, _ = diagnose_ok_geometry(self.root, self.manifest)
        triples = [event for event in report["events"] if len(event["lights"]) == 3]
        self.assertEqual(len(triples), 1)
        self.assertEqual(
            triples[0]["association"], "UNVERIFIED_NAME_OCR_CANDIDATE"
        )
        self.assertEqual(set(triples[0]["lights"]), {"SIDE", "TOP", "MID"})


if __name__ == "__main__":
    import unittest
    unittest.main()
