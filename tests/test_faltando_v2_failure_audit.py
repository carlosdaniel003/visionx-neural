"""Regression tests for read-only failure visualizations without training."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import cv2

from src.scripts.faltando_v2_audit_failed_pairs import (
    audit_failed_pairs, _difference, _read_report,
)
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


class FailureAuditTests(CNNV2Fixture):
    def _report(self):
        from src.services.faltando_neural_archive_experiment import (
            archive_labeled_experimental_events,
        )
        events, provenance = archive_labeled_experimental_events(self.manifest_path)
        ng = next(e for e in events if e["label"] == "NG")
        ok = next(e for e in events if e["label"] == "OK")
        def case(event, pred, score):
            light = next(iter(event["views"]))
            return {
                "event_id_or_group_hint":
                    "archive_provisional:" + event["sources"][0],
                "label": event["label"],
                "predicted": pred,
                "correct": pred == event["label"],
                "ng_score": score,
                "per_light": {
                    light: {"score": score, "correct": pred == event["label"]}
                },
            }
        ngcase = case(ng, "OK", .012)
        okcase = case(ok, "NG", .990)
        history_case = dict(okcase)
        history_case["event_id_or_group_hint"] = (
            "single:" + ok["sources"][0]
        )
        data = {
            "schema": "visionx.faltando_v2_guarded_light_repair_experiment.v1",
            "candidate_only": True, "promoted": False,
            "production_approval_eligible": False,
            "manifest_sha256": provenance["manifest_sha256"],
            "candidate_sha256": "synthetic_sha",
            "new_historical": {"cases": [history_case]},
            "new_train": {"cases": [ngcase, okcase]},
            "new_holdout": {"cases": [ngcase]},
        }
        folder = self.root / "reports/faltando_neural/archive_repairs/test"
        folder.mkdir(parents=True)
        path = folder / "repair_report.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, data

    def test_contact_sheets_show_real_pairs_and_keep_sources_unchanged(self):
        report, data = self._report()
        originals = {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*.png")
        }
        result = audit_failed_pairs(
            self.root, self.manifest_path, report, per_page=1,
        )
        self.assertEqual(result["counts"], {
            "historico": 1, "treino": 2, "holdout": 1
        })
        self.assertEqual(result["unique_source_count"], 2)
        self.assertEqual(len(result["pages"]), 4)
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["checkpoint_promoted"])
        self.assertFalse(result["labels_changed"])
        self.assertEqual(
            result["cases"]["historico"][0]["source"],
            result["cases"]["treino"][0]["source"],
        )
        for filename in result["pages"]:
            img = cv2.imread(str(self.root / filename))
            self.assertIsNotNone(img)
            self.assertGreater(img.shape[1], 800)
            self.assertGreater(img.shape[0], 250)
        self.assertEqual(originals, {
            name: sha256(Path(name).read_bytes()).hexdigest()
            for name in originals
        })
        self.assertFalse((self.run/"qualification.json").exists())
        self.assertFalse((self.root/"reports/neural_online/live_active.json").exists())

    def test_changed_manifest_rejected_before_writing(self):
        report, data = self._report()
        data["manifest_sha256"] = "0" * 64
        report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifesto mudou"):
            audit_failed_pairs(self.root, self.manifest_path, report)
        self.assertFalse(
            (self.root/"reports/faltando_neural/failed_pair_audits").exists()
        )

    def test_paths_outside_experiment_rejected(self):
        with self.assertRaisesRegex(ValueError, "Relatório fora"):
            _read_report(self.root, self.manifest_path)
