"""Regressões: comparar 5/7 sem warm-start nem ativar checkpoint."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.core.neural import faltando_live
from src.scripts.compare_faltando_recovery_epochs import (
    _relative, run_comparison, summarize_report, EPOCHS,
)
from tests.test_faltando_cnn_recovery_preflight import (
    RecoveryPreflightTests, ID,
)


class RecoveryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RecoveryPreflightTests(
            "test_accept_three_true_ng_events_from_complete_correction"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.original = self.root / faltando_live.PINNED_RELATIVE_CHECKPOINT
        self.original.parent.mkdir(parents=True, exist_ok=True)
        self.original.write_bytes(b"checkpoint pinado sintetico, imutavel")
        self.expected_sha = sha256(self.original.read_bytes()).hexdigest()
        self.guard = patch.object(
            faltando_live, "PINNED_CHECKPOINT_SHA256", self.expected_sha
        )
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.calls = []

    def trainer(self, root, first_event, *, epochs, candidate_only,
                recovery_correction_id):
        self.calls.append((epochs, candidate_only, recovery_correction_id))
        self.assertEqual(root, self.root)
        self.assertTrue(candidate_only)
        self.assertEqual(recovery_correction_id, ID)
        self.assertEqual(epochs, EPOCHS[len(self.calls) - 1])
        self.assertEqual(first_event, self.fixture.journals[0])

        stem = f"fake_{epochs}_epochs"
        candidate = self.root / "reports/neural_online/recovery_candidates" / (
            stem + ".pt"
        )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(f"independent training {epochs}".encode())
        event_rows = []
        correct_images = 0
        for index, path in enumerate(self.fixture.journals):
            scores = {
                "SIDE": .999,
                "TOP": .94,
                "MID": .10 if epochs == 5 and index == 2 else .80,
            }
            light_data = {
                mode: {"score": score, "correct": score >= .5}
                for mode, score in scores.items()
            }
            correct_images += sum(a["correct"] for a in light_data.values())
            event_rows.append({
                "event_id_or_group_hint": path.stem,
                "label": "NG",
                "predicted": "NG",
                "correct": True,
                "ng_score": .999,
                "per_light": light_data,
            })
        complete = correct_images == 9
        report = {
            "schema": "visionx.specialist_online_training_report.v1",
            "train_epochs": epochs,
            "base_checkpoint_sha256": self.expected_sha,
            "correction_id": ID,
            "recovery_correction_id": ID,
            "candidate_only": True,
            "promoted": False,
            "forced_events_per_epoch": [
                path.stem for path in self.fixture.journals
            ],
            "training_event_count": 70,
            "historical_baseline": {
                "correct_images": 117, "images": 117,
            },
            "historical_candidate": {
                "correct_images": 117, "images": 117,
                "correct_events": 67, "events": 67, "ng_false_ok": 0,
            },
            "online_baseline": {"correct_images": 4},
            "online_candidate": {
                "correct_images": correct_images,
                "images": 9, "correct_events": 3, "events": 3,
                "ng_false_ok": 0, "cases": event_rows,
            },
            "gates": {"historical_unchanged": True,
                      "all_confirmed_online_correct": complete},
            "passed_internal_gates": complete,
            "candidate": candidate.relative_to(self.root).as_posix(),
            "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        }
        saved = self.root / "reports/neural_online/training_reports" / (
            stem + ".json"
        )
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(report), encoding="utf-8")
        return report

    def test_independent_five_and_seven_epochs_comparison(self):
        from src.scripts.compare_faltando_recovery_epochs import SUMMARY_SCHEMA
        result = run_comparison(self.root, ID, trainer=self.trainer)
        self.assertEqual(self.calls, [(5, True, ID), (7, True, ID)])
        self.assertEqual(result["schema"], SUMMARY_SCHEMA)
        self.assertEqual(result["epochs_compared"], [5, 7])
        self.assertEqual(
            result["experiments"]["5"]["correction_only"]["images_correct"], 8
        )
        self.assertEqual(
            result["experiments"]["7"]["correction_only"]["images_correct"], 9
        )
        self.assertEqual(
            result["experiments"]["5"]["correction_only"]["correct_by_light"]["MID"], 2
        )
        self.assertEqual(
            result["experiments"]["7"]["correction_only"]["correct_by_light"]["MID"], 3
        )
        self.assertEqual(result["experiments"]["7"]["historical"]["correct_images"], 117)
        self.assertTrue(Path(result["report_path"]).is_file())
        self.assertEqual(
            json.loads(Path(result["report_path"]).read_text())["schema"],
            SUMMARY_SCHEMA,
        )
        self.assertEqual(sha256(self.original.read_bytes()).hexdigest(), self.expected_sha)
        self.assertFalse(
            (self.root / "reports/neural_online/live_active.json").exists()
        )
        self.assertFalse(
            (self.root / "reports/neural_online/outcomes").exists()
        )

    def test_prior_live_active_blocks_training(self):
        pointer = self.root / "reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"active":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Ponteiro"):
            run_comparison(self.root, ID, trainer=self.trainer)
        self.assertEqual(self.calls, [])

    def test_missing_mid_score_blocks_consolidation(self):
        def defective_trainer(*args, **kwargs):
            row = self.trainer(*args, **kwargs)
            if kwargs["epochs"] == 7:
                row["online_candidate"]["cases"][0]["per_light"].pop("MID")
                file = self.root / "reports/neural_online/training_reports/fake_7_epochs.json"
                file.write_text(json.dumps(row), encoding="utf-8")
            return row

        with self.assertRaisesRegex(ValueError, "SIDE/TOP/MID"):
            run_comparison(self.root, ID, trainer=defective_trainer)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(list(
            (self.root / "reports/neural_online/recovery_comparisons").glob("*.json")
        ) if (self.root / "reports/neural_online/recovery_comparisons").exists() else [])

    def test_rejects_candidate_outside_experimental_folder(self):
        with self.assertRaisesRegex(ValueError, "inseguro"):
            _relative(self.root, "../live_active.json",
                      self.root / "reports/neural_online/recovery_candidates")


if __name__ == "__main__":
    unittest.main()
