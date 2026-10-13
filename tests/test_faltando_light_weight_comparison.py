"""Experimento 7 épocas: peso auxiliar por luz, somente candidatos."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.core.neural import faltando_live
from src.scripts.compare_faltando_recovery_light_weights import (
    SCHEMA, WEIGHTS, EPOCHS, compare_light_weights,
)
from tests.test_faltando_cnn_recovery_preflight import (
    RecoveryPreflightTests, ID,
)


class LightWeightComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RecoveryPreflightTests(
            "test_accept_three_true_ng_events_from_complete_correction"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root.resolve()
        self.base = self.root / faltando_live.PINNED_RELATIVE_CHECKPOINT
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.base.write_bytes(b"mock-fixed-pinned-checkpoint")
        self.base_sha = sha256(self.base.read_bytes()).hexdigest()
        self.pin = patch.object(
            faltando_live, "PINNED_CHECKPOINT_SHA256", self.base_sha,
        )
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.calls = []

    def trainer(self, root, first_event, *, epochs, candidate_only,
                recovery_correction_id, light_loss_weight):
        self.assertEqual(root, self.root)
        self.assertEqual(first_event.resolve(), self.fixture.journals[0].resolve())
        self.assertTrue(candidate_only)
        self.assertEqual(recovery_correction_id, ID)
        self.assertEqual(epochs, EPOCHS)
        self.assertEqual(
            light_loss_weight, WEIGHTS[len(self.calls)]
        )
        self.calls.append(light_loss_weight)
        stem = f"experiment_{len(self.calls)}"
        candidate = self.root / "reports/neural_online/recovery_candidates" / (
            stem + ".pt"
        )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(f"weights_{light_loss_weight}".encode())
        records = []
        scores_last_mid = {0.15: 0.489674, 0.5: 0.70, 1.0: 0.88}
        n_correct = 0
        for i, item in enumerate(self.fixture.journals):
            per_light = {}
            for light in ("SIDE", "TOP", "MID"):
                value = (
                    scores_last_mid[light_loss_weight]
                    if i == 2 and light == "MID" else 0.97
                )
                per_light[light] = {
                    "score": value, "correct": value >= 0.5,
                }
                n_correct += value >= 0.5
            records.append({
                "event_id_or_group_hint": item.stem,
                "label": "NG",
                "predicted": "NG",
                "correct": True,
                "ng_score": 0.97,
                "per_light": per_light,
            })
        passed = n_correct == 9
        result = {
            "schema": "visionx.specialist_online_training_report.v1",
            "candidate_only": True, "promoted": False,
            "train_epochs": epochs,
            "light_loss_weight": light_loss_weight,
            "recovery_correction_id": ID,
            "base_checkpoint_sha256": self.base_sha,
            "forced_events_per_epoch": [x.stem for x in self.fixture.journals],
            "training_event_count": 70,
            "historical_baseline": {
                "images": 117, "correct_images": 117,
            },
            "historical_candidate": {
                "images": 117, "correct_images": 117,
                "correct_events": 67, "events": 67, "ng_false_ok": 0,
            },
            "online_baseline": {"correct_images": 4},
            "online_candidate": {
                "images": 9, "correct_images": n_correct,
                "events": 3, "correct_events": 3,
                "ng_false_ok": 0, "cases": records,
            },
            "gates": {"all_confirmed_online_correct": passed,
                      "historical_unchanged": True},
            "passed_internal_gates": passed,
            "candidate": candidate.relative_to(self.root).as_posix(),
            "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        }
        report_path = self.root / "reports/neural_online/training_reports" / (
            stem + ".json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result), encoding="utf-8")
        return result

    def test_compares_three_independent_candidates_with_same_sha(self):
        result = compare_light_weights(self.root, ID, trainer=self.trainer)
        self.assertEqual(self.calls, list(WEIGHTS))
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["epochs_per_experiment"], 7)
        self.assertTrue(result["independent_from_original_checkpoint"])
        self.assertFalse(result["automatic_selection"])
        self.assertFalse(result["production_approved"])
        for key in ("0.15", "0.50", "1.00"):
            entry = result["experiments"][key]
            self.assertFalse(entry["promoted"])
            self.assertEqual(entry["historical"]["correct_images"], 117)
            self.assertEqual(entry["correction_only"]["correct_by_light"]["SIDE"], 3)
            self.assertEqual(entry["correction_only"]["correct_by_light"]["TOP"], 3)
        self.assertEqual(
            result["experiments"]["0.15"]["correction_only"]["correct_by_light"]["MID"], 2
        )
        self.assertEqual(
            result["experiments"]["1.00"]["correction_only"]["correct_by_light"]["MID"], 3
        )
        self.assertTrue(Path(result["report_path"]).is_file())
        self.assertEqual(sha256(self.base.read_bytes()).hexdigest(), self.base_sha)
        self.assertFalse(
            (self.root / "reports/neural_online/live_active.json").exists()
        )
        self.assertFalse(
            (self.root / "reports/neural_online/outcomes").exists()
        )

    def test_rejects_wrong_weight_report_before_next_round(self):
        def tampered(*args, **kwargs):
            data = self.trainer(*args, **kwargs)
            data["light_loss_weight"] = 42
            return data

        with self.assertRaisesRegex(ValueError, "não respeitou peso"):
            compare_light_weights(self.root, ID, trainer=tampered)
        self.assertEqual(self.calls, [0.15])

    def test_rejects_pointer_activation_mid_comparison(self):
        def intruder(*args, **kwargs):
            data = self.trainer(*args, **kwargs)
            pointer = self.root / "reports/neural_online/live_active.json"
            pointer.write_text('{"BAD":"activated"}', encoding="utf-8")
            return data

        with self.assertRaisesRegex(ValueError, "modelo ativo/base mudou"):
            compare_light_weights(self.root, ID, trainer=intruder)
        self.assertEqual(self.calls, [0.15])
        self.assertFalse(
            (self.root / "reports/neural_online/recovery_comparisons").exists()
        )


if __name__ == "__main__":
    unittest.main()
