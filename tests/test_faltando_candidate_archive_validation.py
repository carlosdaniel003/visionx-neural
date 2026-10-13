"""Validação do candidato 0,50 sem pesos ativos/XP, replay sem holdout."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.services import faltando_cnn_candidate_archive_validation as v
from src.services.startup_regression.interactive_training_validation import (
    summarize,
)


class CandidateArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / v.PINNED_RELATIVE_CHECKPOINT
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"known original model")
        self.candidate = self.root / "reports/neural_online/recovery_candidates/mock.pt"
        self.candidate.parent.mkdir(parents=True)
        self.candidate.write_bytes(b"new offline weights")
        self.base_sha = sha256(self.checkpoint.read_bytes()).hexdigest()
        self.cand_sha = sha256(self.candidate.read_bytes()).hexdigest()
        self.fake_events = [self.root / f"event_{i}.json" for i in range(3)]
        self.mocks = [
            patch.object(v, "PINNED_CHECKPOINT_SHA256", self.base_sha),
            patch.object(v, "CANDIDATE_SHA", self.cand_sha),
            patch.object(v, "validate_recovery_inputs",
                         return_value={"events": self.fake_events}),
        ]
        for m in self.mocks:
            m.start()
            self.addCleanup(m.stop)
        self.comparison = (
            self.root / "reports/neural_online/recovery_comparisons" /
            v.COMPARISON_FILE
        )
        self.comparison.parent.mkdir(parents=True)
        self.report = self.root / "reports/neural_online/training_reports" / "fake.json"
        self.report.parent.mkdir(parents=True)
        self.report.write_text(json.dumps({
            "candidate_sha256": self.cand_sha, "candidate_only": True,
            "promoted": False, "recovery_correction_id": v.CORRECTION_ID,
            "light_loss_weight": 0.50,
            "base_checkpoint_sha256": self.base_sha,
        }), encoding="utf-8")
        self.comparison.write_text(json.dumps({
            "schema": "visionx.faltando_recovery_light_weight_comparison.v1",
            "correction_id": v.CORRECTION_ID,
            "epochs_per_experiment": 7,
            "independent_from_original_checkpoint": True,
            "candidate_only": True, "active_model_changed": False,
            "production_approved": False, "automatic_selection": False,
            "original_checkpoint_sha256": self.base_sha,
            "verified_event_ids": [e.stem for e in self.fake_events],
            "experiments": {"0.50": {
                "epochs": 7, "light_loss_weight": 0.50,
                "passed_internal_gates": True, "promoted": False,
                "candidate_sha256": self.cand_sha,
                "candidate_path": self.candidate.relative_to(self.root).as_posix(),
                "report_path": self.report.relative_to(self.root).as_posix(),
                "correction_only": {
                    "images_correct": 9, "images_total": 9,
                    "correct_by_light": {"SIDE": 3, "TOP": 3, "MID": 3},
                },
                "historical": {
                    "correct_images": 117, "total_images": 117,
                    "ng_false_ok": 0,
                },
                "online_all_events": {
                    "correct_images": 9, "total_images": 9, "ng_false_ok": 0,
                },
                "gates": {
                    "historical_unchanged": True,
                    "all_confirmed_online_correct": True,
                    "no_new_historical_false_ok": True,
                },
            }},
        }), encoding="utf-8")

    def test_candidate_preflight_sha_and_exact_identity(self):
        data = v.candidate_preflight(self.root)
        self.assertEqual(data["candidate_path"], self.candidate)
        self.assertEqual(data["candidate_sha256"], self.cand_sha)
        self.assertEqual(data["baseline_sha256"], self.base_sha)

    def test_tampered_candidate_rejected(self):
        self.candidate.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            v.candidate_preflight(self.root)

    def test_pointer_present_fails_closed(self):
        pointer = self.root / "reports/neural_online/live_active.json"
        pointer.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "online ativo"):
            v.candidate_preflight(self.root)

    @staticmethod
    def inventory(_root):
        return {"images": [
            {"path": "public/ng_archive/ng_SIDE.png", "file_sha256": "012",
             "status": "VALID_PNG", "expected_label": "NG",
             "category_hint": "FALTANDO", "lighting_mode": "SIDE"},
            {"path": "public/ok_archive/ok_MID.png", "file_sha256": "345",
             "status": "VALID_PNG", "expected_label": "OK",
             "category_hint": "INVERTIDO", "lighting_mode": "MID"},
        ]}

    def mock_validate(self, root, *, inventory, cnn, progress):
        is_candidate = cnn.checkpoint_path == self.candidate
        rows = []
        for item in inventory["images"]:
            rows.append({
                "model": "KNN_MEMORIA", "source_path": item["path"],
                "category": item["category_hint"],
                "lighting_mode": item["lighting_mode"],
                "expected_label": item["expected_label"],
                "predicted_label": item["expected_label"], "status": "PASSOU",
                "cnn_score_ng": None, "checkpoint_sha256": None,
                "error": None,
            })
            defect = item["expected_label"] == "NG"
            rows.append({
                "model": v.CNN, "source_path": item["path"],
                "category": item["category_hint"],
                "lighting_mode": item["lighting_mode"],
                "expected_label": item["expected_label"],
                "predicted_label": (
                    None if is_candidate and defect else
                    ("OK" if defect else "OK")
                ),
                "status": (
                    "REVISAO" if is_candidate and defect else
                    ("ERRO_CLASSIFICACAO" if defect else "PASSOU")
                ),
                "cnn_score_ng": 0.66 if is_candidate and defect else 0.01,
                "checkpoint_sha256": cnn.expected_sha256,
                "error": None,
            })
        return {
            "mode": "READ_ONLY_ARCHIVE_REPLAY", "complete": True,
            "canceled": False, "archive_png_count": 2,
            "processed_model_checks": 4, "planned_model_checks": 4,
            "cases": rows, "models": summarize(rows, inventory["images"]),
            "root": str(root), "duration_seconds": 0.1,
            "notes": ["mock replay"], "generated_at_utc": "mock",
        }

    def test_comparison_measures_review_not_false_ok_and_writes_no_pointer(self):
        class FakeCNN:
            def __init__(self, *, checkpoint_path, expected_sha256, online_root):
                self.checkpoint_path = checkpoint_path
                self.expected_sha256 = expected_sha256

        result = v.validate_candidate_on_archive(
            self.root, inventory_fn=self.inventory,
            validator=self.mock_validate, model_factory=FakeCNN,
        )
        self.assertTrue(Path(result["report_path"]).is_file())
        self.assertFalse(result["holdout_independent"])
        self.assertFalse(result["active_model_changed"])
        comparison = result["comparison"]
        self.assertEqual(comparison["ng_false_ok_baseline"], 1)
        self.assertEqual(comparison["ng_false_ok_candidate"], 0)
        self.assertEqual(comparison["reviews_candidate"], 1)
        self.assertEqual(len(comparison["case_changes"]), 1)
        self.assertEqual(comparison["case_changes"][0]["candidate_status"], "REVISAO")
        self.assertEqual(sha256(self.checkpoint.read_bytes()).hexdigest(), self.base_sha)
        self.assertFalse(
            (self.root / "reports/neural_online/live_active.json").exists()
        )

    def test_comparison_detects_archive_label_swap(self):
        rows = self.mock_validate(
            self.root, inventory=self.inventory(self.root),
            cnn=type("C", (), {"checkpoint_path": self.checkpoint,
                               "expected_sha256": self.base_sha})(),
            progress=None,
        )
        revised = json.loads(json.dumps(rows))
        revised["cases"][1]["expected_label"] = "OK"
        with self.assertRaisesRegex(ValueError, "Identidade"):
            v.compare_archive_results(rows, revised, self.base_sha, self.base_sha)


if __name__ == "__main__":
    unittest.main()
