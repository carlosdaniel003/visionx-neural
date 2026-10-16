"""Reproducible, read-only benchmark of the original 97.70% candidate."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts import benchmark_original_97p70_offline as bench
from src.scripts.train_aoi_universal_quality_offline import (
    SCHEMA as UNIVERSAL_SCHEMA, REPORT_SCHEMA, _metrics,
)
from src.scripts.train_faltando_cnn_v2_online import _assess
from src.services.aoi_universal_quality_archive import (
    universal_archive_events, split_universal_events,
)
from tests.test_aoi_universal_quality_offline import UniversalQualityOfflineTests


class Benchmark97p70Tests(unittest.TestCase):
    setUp = UniversalQualityOfflineTests.setUp
    _prepare = UniversalQualityOfflineTests._prepare

    def _reports(self):
        records, provenance = universal_archive_events(self.manifest)
        train, val, test, split = split_universal_events(records)
        torch.manual_seed(126)
        universal_model = FaltandoCNNV2()
        u_predictions = _assess(universal_model, test, 64, .7)
        u_metrics = _metrics(test, u_predictions)
        u_folder = (
            self.root/"reports/aoi_universal_quality/experiments/synthetic"
        )
        u_folder.mkdir(parents=True)
        u_path = u_folder/"aoi_general_quality_offline_candidate.pt"
        torch.save({
            "schema": UNIVERSAL_SCHEMA, "candidate_only": True,
            "production_approved": False, "automatic_ok_authorized": False,
            "source_manifest_sha256": provenance["manifest_sha256"],
            "selected_epoch": 1, "state_dict": universal_model.state_dict(),
        }, u_path)
        u_report = {
            "schema": REPORT_SCHEMA, "candidate_only": True,
            "promoted": False, "production_approval_eligible": False,
            "aoi_category_used_as_model_input": False,
            "epoch_selection_uses_test": False,
            "provenance": provenance,
            "source_candidate_relative_path": u_path.relative_to(self.root).as_posix(),
            "candidate_sha256": sha256(u_path.read_bytes()).hexdigest(),
            "selected_epoch": 1,
            "seed": 42,
            "split": split,
            "test": u_metrics,
            "test_cases": u_predictions["cases"],
        }
        u_report_path = u_folder/"training_report.json"
        u_report_path.write_text(json.dumps(u_report), encoding="utf-8")
        torch.manual_seed(1)
        original_model = FaltandoCNNV2()
        legacy_folder = (
            self.root/"reports/faltando_neural/archive_experiments/synthetic"
        )
        legacy_folder.mkdir(parents=True)
        original_candidate = (
            legacy_folder/"faltando_v2_archive_experimental_candidate.pt"
        )
        base_sha = "a" * 64
        torch.save({
            "schema": MODEL_SCHEMA_V2,
            "state_dict": original_model.state_dict(),
            "experimental": True, "production_approved": False,
            "individual_pair_review_complete": False,
            "archive_labels_experimental": True,
            "source_champion_sha256": base_sha,
            "provisional_archive_manifest_sha256": "b" * 64,
            "image_size": 64, "focus_fraction": .7,
        }, original_candidate)
        c_sha = sha256(original_candidate.read_bytes()).hexdigest()
        legacy_report = {
            "qualification_mode": "ARCHIVE_FOLDER_BULK_EXPERIMENT",
            "candidate_only": True, "promoted": False,
            "production_approval_eligible": False,
            "base_checkpoint_sha256": base_sha,
            "candidate_relative_path":
                original_candidate.relative_to(self.root).as_posix(),
            "candidate_sha256": c_sha,
            "provenance": {"manifest_sha256": "b" * 64},
            "new_holdout": {
                "images": 87, "correct_images": 85, "ng_false_ok": 0,
            },
            "new_historical": {"images": 117, "correct_images": 98},
            "holdout_event_ids": [test[0]["id"]],
        }
        old_report_path = legacy_folder/"training_report.json"
        old_report_path.write_text(json.dumps(legacy_report), encoding="utf-8")
        return old_report_path, u_report_path, c_sha

    def test_exact_shared_partition_and_no_production_changes(self):
        old_report, u_report, digest = self._reports()
        pngs = {str(x): sha256(x.read_bytes()).hexdigest()
                for x in self.root.rglob("*.png")}
        with patch.object(bench, "PINNED_97_70_SHA256", digest):
            r = bench.benchmark_97p70(
                self.root, self.manifest, old_report, u_report,
            )
        self.assertTrue(r["comparison_only"])
        self.assertFalse(r["trained"])
        self.assertFalse(r["production_approval_eligible"])
        self.assertFalse(r["promoted"])
        self.assertFalse(r["active_pointer_changed"])
        self.assertEqual(r["same_universal_test_total"],
                         r["original_97p70_on_same_test"]["total"])
        self.assertTrue(r["test_split_exactly_matched_original"])
        self.assertEqual(r["identities_overlap_with_old_87_holdout"], 1)
        self.assertEqual(sum(r["pairwise_correctness"].values()),
                         r["same_universal_test_total"])
        self.assertEqual(
            r["first_experiment_original_holdout"]["historical_false_ng_count"], 19
        )
        self.assertEqual(pngs, {
            path: sha256(Path(path).read_bytes()).hexdigest()
            for path in pngs
        })
        self.assertFalse(
            (self.root/"reports/neural_online/live_active.json").exists()
        )
        self.assertFalse((self.run/"qualification.json").exists())

    def test_wrong_candidate_or_universal_sha_fails_before_output(self):
        old_report, u_report, digest = self._reports()
        with self.assertRaisesRegex(ValueError, "primeiro checkpoint"):
            bench.benchmark_97p70(
                self.root, self.manifest, old_report, u_report,
            )
        self.assertFalse(
            (self.root/"reports/aoi_universal_quality/benchmarks").exists()
        )
        data = json.loads(u_report.read_text(encoding="utf-8"))
        data["candidate_sha256"] = "0" * 64
        u_report.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(bench, "PINNED_97_70_SHA256", digest):
            with self.assertRaisesRegex(ValueError, "não corresponde"):
                bench.benchmark_97p70(
                    self.root, self.manifest, old_report, u_report,
                )
        self.assertFalse(
            (self.root/"reports/aoi_universal_quality/benchmarks").exists()
        )

    def test_universal_score_or_test_split_tamper_is_rejected(self):
        old_report, u_report, digest = self._reports()
        data = json.loads(u_report.read_text(encoding="utf-8"))
        data["test_cases"][0]["ng_score"] = 3.7
        u_report.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(bench, "PINNED_97_70_SHA256", digest):
            with self.assertRaisesRegex(ValueError, "não finito"):
                bench.benchmark_97p70(self.root, self.manifest, old_report, u_report)

    def test_operational_diagnostics_do_not_auto_approve(self):
        diagnostics = bench._operational_at_existing_thresholds([
            {"label": "NG", "ng_score": .02},
            {"label": "NG", "ng_score": .65},
            {"label": "NG", "ng_score": .93},
            {"label": "OK", "ng_score": .91},
            {"label": "OK", "ng_score": .05},
        ])
        self.assertEqual(diagnostics["NG_missed_as_OK"], 1)
        self.assertEqual(diagnostics["NG_review"], 1)
        self.assertEqual(diagnostics["NG_conclusive"], 1)
        self.assertEqual(diagnostics["OK_false_NG"], 1)
        self.assertTrue(diagnostics["no_threshold_changed"])
