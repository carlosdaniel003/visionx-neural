"""Cross-category ODIN audit: explicit proof that no 0/1 or live routing is enabled."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.faltando_cross_category_audit import (
    SCHEMA, audit_cross_category,
)
from src.services.startup_regression.archive_inventory import inventory_archives
from src.ui.production_confidence_gate import production_decision_policy


class FakeExtractor:
    def __init__(self, *, observed_category="EMBORCADO"):
        self.calls = 0
        self.category = observed_category

    def __call__(self, image):
        self.calls += 1
        return image.copy(), image.copy(), {
            "category": self.category, "board": "PLACA_TESTE", "parts": "R1",
            "value": "104",
        }


class FakeCNN:
    def __init__(self, *, review=False):
        self.calls = 0
        self.review = review

    def inspect(self, reference, test, light):
        self.calls += 1
        if self.review:
            return {
                "verdict": "REVISÃO OBRIGATÓRIA",
                "production_review_required": True,
                "detail": {"cnn_v2_active": True, "cnn_v2_experimental": True,
                           "cnn_v2_checkpoint_verified": False}
            }
        ng = float(np.mean(test)) > 100
        return {
            "verdict": "DEFEITO REAL" if ng else "FALHA FALSA",
            "is_defect": ng,
            "detail": {
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_ng_score_uncalibrated": .95 if ng else .05,
            },
        }


class CrossCategoryAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ok = self.root / "public" / "ok_archive"
        self.ng = self.root / "public" / "ng_archive"
        self.ok.mkdir(parents=True)
        self.ng.mkdir(parents=True)

    def save(self, folder, filename, value):
        frame = np.full((96, 112, 3), value, dtype=np.uint8)
        path = folder / filename
        self.assertTrue(cv2.imwrite(str(path), frame))
        return path

    def test_all_categories_audited_no_live_change(self):
        self.save(self.ok, "2026-10-09_0730_EMBORCADO_SIDE.png", 65)
        self.save(self.ng, "2026-10-09_0731_INVERTIDO_SIDE.png", 205)
        self.save(self.ok, "2026-10-09_0732_FALTANDO_TOP.png", 45)
        self.save(self.ng, "2026-10-09_0733_MUITO_ADESIVO_MID.png", 220)
        original = {
            p: sha256(p.read_bytes()).hexdigest()
            for p in (*self.ok.glob("*.png"), *self.ng.glob("*.png"))
        }
        model, extractor = FakeCNN(), FakeExtractor()
        report, out = audit_cross_category(
            self.root, extractor=extractor, predictor=model
        )
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["total_images"], 4)
        self.assertEqual(report["summary"]["cnn_decisions"], 4)
        self.assertEqual(report["summary"]["correct_against_archive_label"], 4)
        self.assertEqual(report["summary"]["archived_ng_called_ok"], 0)
        self.assertTrue({"FALTANDO", "EMBORCADO", "INVERTIDO"}.issubset(
            report["by_category"]
        ))
        self.assertEqual(model.calls, 4)
        self.assertEqual(extractor.calls, 4)
        self.assertFalse(report["knn_used"])
        self.assertFalse(report["production_modified"])
        self.assertFalse(report["automatic_zero_or_one_enabled"])
        self.assertFalse(report["cnn_cross_category_approved"])
        self.assertFalse(report["ng_generalization_proven"])
        self.assertTrue((out/"cross_category_audit.json").is_file())
        self.assertTrue((out/"cross_category_audit.txt").is_file())
        self.assertEqual(original, {
            p: sha256(p.read_bytes()).hexdigest() for p in original
        })

    def test_model_uncertainty_cannot_be_counted_as_ok(self):
        self.save(self.ng, "2026-10-09_0730_INVERTIDO.png", 192)
        report, _ = audit_cross_category(
            self.root, extractor=FakeExtractor(), predictor=FakeCNN(review=True),
        )
        self.assertEqual(report["summary"]["cnn_review"], 1)
        self.assertEqual(report["summary"]["cnn_decisions"], 0)
        self.assertIsNone(report["images"][0]["label_agrees_with_cnn_decision"])
        self.assertIsNone(report["images"][0]["raw_score_at_half_only"])

    def test_png_corruption_does_not_disappear_silently(self):
        file = self.ok / "2026-10-09_0730_EMBORCADO.png"
        file.write_bytes(b"bad PNG")
        model = FakeCNN()
        report, _ = audit_cross_category(
            self.root, extractor=FakeExtractor(), predictor=model
        )
        self.assertEqual(report["total_images"], 1)
        self.assertEqual(report["images"][0]["evaluation_status"], "INVALID_PNG")
        self.assertEqual(model.calls, 0)

    def test_identical_pixels_with_opposite_labels_not_scored(self):
        ok = self.save(self.ok, "2026-10-09_0730_EMBORCADO.png", 88)
        ng = self.ng / "2026-10-09_0730_INVERTIDO.png"
        ng.write_bytes(ok.read_bytes())
        model = FakeCNN()
        report, _ = audit_cross_category(
            self.root, extractor=FakeExtractor(), predictor=model
        )
        self.assertEqual(report["total_images"], 2)
        self.assertEqual(report["cross_label_conflict_groups"], 1)
        self.assertEqual(report["summary"]["status_counts"]["CONTRADICTORY_LABEL"], 2)
        self.assertEqual(model.calls, 0)

    def test_archive_mutation_after_inventory_is_blocked(self):
        path = self.save(self.ok, "2026-10-09_0730_EMBORCADO.png", 80)
        inventory = inventory_archives(self.root)
        original = path.read_bytes()
        self.save(self.ok, path.name, 105)
        report, _ = audit_cross_category(
            self.root, inventory=inventory,
            extractor=FakeExtractor(), predictor=FakeCNN()
        )
        self.assertEqual(report["summary"]["cnn_decisions"], 0)
        self.assertEqual(report["images"][0]["evaluation_status"], "INVALID_OR_FAILED")
        self.assertIn("mudou", report["images"][0]["issues"][0])
        self.assertNotEqual(path.read_bytes(), original)

    def test_an_archived_ng_called_ok_is_explicitly_counted(self):
        self.save(self.ng, "2026-10-09_0730_INVERTIDO.png", 50)
        report, _ = audit_cross_category(
            self.root, extractor=FakeExtractor(), predictor=FakeCNN()
        )
        self.assertEqual(report["summary"]["archived_ng_called_ok"], 1)
        self.assertEqual(report["summary"]["correct_against_archive_label"], 0)
        self.assertFalse(report["cnn_cross_category_approved"])

    def test_existing_production_gate_blocks_new_cnn_ok(self):
        result = {
            "is_defect": False, "verdict": "FALHA FALSA",
            "detail": {"cnn_v2_active": True, "cnn_v2_experimental": True}
        }
        policy = production_decision_policy(result)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")

    def test_invalid_inventory_count_aborts_audit(self):
        self.save(self.ok, "2026-10-09_0730_FALTANDO.png", 50)
        inventory = inventory_archives(self.root)
        inventory["summary"]["png_count"] = 0
        with self.assertRaisesRegex(ValueError, "Cobertura"):
            audit_cross_category(
                self.root, inventory=inventory,
                extractor=FakeExtractor(), predictor=FakeCNN()
            )


if __name__ == "__main__":
    unittest.main()
