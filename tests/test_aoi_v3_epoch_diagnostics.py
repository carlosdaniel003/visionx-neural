"""CNN V3: aprender as 14 imagens sem perder as épocas reprovadas no gate."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import torch

from src.services.aoi_v3_epoch_diagnostics import (
    EPOCH_SCHEMA, operator_epoch_replay, epoch_review_comparison,
    best_exploratory_epoch, persist_epoch_diagnostics,
)


def _human_rows():
    return [
        {
            "id": f"human-{index:02d}",
            "label": "NG" if index < 9 else "OK",
            "label_provenance": "HUMAN_V3_IMAGE_SHA256",
            "views": {"SIDE": {}},
            "sources": [f"public/archive/image_{index:02d}.png"],
        }
        for index in range(14)
    ]


def _predictions(rows, correct_count):
    cases = []
    for index, row in enumerate(rows):
        good = index < correct_count
        score = (
            (.97 if good else .03)
            if row["label"] == "NG" else (.03 if good else .97)
        )
        cases.append({
            "event_id_or_group_hint": row["id"],
            "label": row["label"],
            "predicted": row["label"] if good
                         else ("OK" if row["label"] == "NG" else "NG"),
            "ng_score": score,
            "correct": good,
            "per_light": {"SIDE": {"score": score, "correct": good}},
        })
    return {"cases": cases}


def _validation(ng_false_ok, ok_false_ng, balanced):
    return {
        "NG_false_OK": ng_false_ok,
        "OK_false_NG": ok_false_ng,
        "balanced_accuracy": balanced,
    }


class EpochDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.rows = _human_rows()
        self.base = operator_epoch_replay(
            self.rows, _predictions(self.rows, 0),
        )

    def test_human_training_replay_distinguishes_progress_in_both_classes(self):
        observed = operator_epoch_replay(
            self.rows, _predictions(self.rows, 12),
        )
        self.assertEqual(observed["metrics"]["correct"], 12)
        self.assertEqual(observed["metrics"]["NG_correct"], 9)
        self.assertEqual(observed["metrics"]["OK_correct"], 3)
        self.assertEqual(len(observed["incorrect_sources"]), 2)
        self.assertGreater(
            observed["mean_true_label_score"],
            self.base["mean_true_label_score"],
        )
        self.assertLess(
            observed["mean_true_label_log_loss"],
            self.base["mean_true_label_log_loss"],
        )
        self.assertTrue(observed["in_sample_not_independent_test"])

    def test_replay_rejects_wrong_count_or_unreviewed_labels(self):
        with self.assertRaisesRegex(ValueError, "9 NG e 5 OK"):
            operator_epoch_replay(
                self.rows[:-1], _predictions(self.rows[:-1], 0),
            )
        rows = [dict(row) for row in self.rows]
        rows[0]["label_provenance"] = "ARCHIVE_FOLDER_PROVISIONAL"
        with self.assertRaisesRegex(ValueError, "9 NG e 5 OK"):
            operator_epoch_replay(rows, _predictions(rows, 0))

    def test_epoch_catches_human_learning_but_independent_ok_regression(self):
        candidate = operator_epoch_replay(
            self.rows, _predictions(self.rows, 10),
        )
        baseline_validation = _validation(3, 0, .9903846)
        epoch_validation = _validation(1, 1, .9888584)
        diagnostic = epoch_review_comparison(
            baseline_validation, epoch_validation, self.base, candidate,
        )
        self.assertEqual(diagnostic["human_correct_delta"], 10)
        self.assertEqual(len(diagnostic["human_became_correct"]), 10)
        self.assertEqual(diagnostic["NG_false_OK_change_on_validation"], -2)
        self.assertEqual(diagnostic["OK_false_NG_change_on_validation"], 1)
        self.assertFalse(diagnostic["validation_non_regression_gate_passed"])
        self.assertEqual(diagnostic["validation_errors_worse"], ["OK_TO_NG"])
        history = [
            {
                "epoch": 0, "validation": baseline_validation,
                "operator_training_replay": self.base,
                "comparison_with_epoch_zero": epoch_review_comparison(
                    baseline_validation, baseline_validation, self.base,
                    self.base,
                ),
            },
            {
                "epoch": 1, "validation": epoch_validation,
                "operator_training_replay": candidate,
                "comparison_with_epoch_zero": diagnostic,
            },
        ]
        # O selecionado seguro segue na época zero, mas a época um
        # permanece diagnóstico útil, sem promover nem olhar TESTE.
        self.assertEqual(best_exploratory_epoch(history), 1)

    def test_no_exploratory_progress_if_all_epochs_fail_to_learn(self):
        val = _validation(3, 0, .9903)
        history = [
            {"epoch": number, "validation": val,
             "operator_training_replay": self.base,
             "comparison_with_epoch_zero": {}}
            for number in (0, 1, 2)
        ]
        self.assertIsNone(best_exploratory_epoch(history))

    def test_all_epoch_weights_are_persisted_only_as_incompatible_diagnostics(self):
        val = _validation(3, 0, .9903)
        human_one = operator_epoch_replay(
            self.rows, _predictions(self.rows, 5),
        )
        human_two = operator_epoch_replay(
            self.rows, _predictions(self.rows, 11),
        )
        v0 = epoch_review_comparison(val, val, self.base, self.base)
        v1 = epoch_review_comparison(val, val, self.base, human_one)
        v2 = epoch_review_comparison(val, val, self.base, human_two)
        history = [
            {"epoch": 0, "validation": val,
             "operator_training_replay": self.base,
             "comparison_with_epoch_zero": v0},
            {"epoch": 1, "validation": val,
             "operator_training_replay": human_one,
             "comparison_with_epoch_zero": v1},
            {"epoch": 2, "validation": val,
             "operator_training_replay": human_two,
             "comparison_with_epoch_zero": v2},
        ]
        self.assertEqual(best_exploratory_epoch(history), 2)
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            folder = (
                root/"reports/aoi_quality_cnn_v3/candidates/run"
            )
            folder.mkdir(parents=True)
            snapshots = [(1, {"weight": torch.tensor([1.])}),
                         (2, {"weight": torch.tensor([2.])})]
            info = persist_epoch_diagnostics(
                folder, snapshots, history,
                manifest_sha256="a"*64,
                previous_candidate_sha256="b"*64,
                image_size=96, safe_selected_epoch=0,
            )
            self.assertEqual(len(info), 2)
            self.assertFalse(any(x["safe_gate_selected"] for x in info))
            self.assertEqual([x["operator_14_correct"] for x in info], [5, 11])
            for entry in info:
                filename = root/entry["path"]
                self.assertTrue(filename.is_file())
                self.assertEqual(
                    sha256(filename.read_bytes()).hexdigest(), entry["sha256"],
                )
                doc = torch.load(filename, weights_only=True)
                self.assertEqual(doc["schema"], EPOCH_SCHEMA)
                self.assertFalse(doc["production_approved"])
                self.assertFalse(doc["automatic_xp_authorized"])
                self.assertFalse(doc["test_evaluated_for_this_epoch"])
                self.assertNotIn("state_dict", doc)
                self.assertNotIn("test", doc)
                self.assertIn("model_weights_not_production_state_dict", doc)
            with self.assertRaisesRegex(ValueError, "Snapshots"):
                persist_epoch_diagnostics(
                    root, [(1, {"weight": torch.tensor([1.])})],
                    history, manifest_sha256="a"*64,
                    previous_candidate_sha256=None,
                    image_size=96, safe_selected_epoch=0,
                )


if __name__ == "__main__":
    unittest.main()
