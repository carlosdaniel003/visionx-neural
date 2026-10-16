"""Regression tests for conservative CNN V2 offline replay and the 97% floor."""
from __future__ import annotations

from unittest.mock import patch

from src.scripts.train_faltando_cnn_v2_verified_archive import (
    _balanced_hard_replay, _operational_light_summary, _selection_score,
    candidate_acceptance,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


def _assessment(cases):
    return {
        "images": sum(len(c["per_light"]) for c in cases),
        "correct_images": sum(
            v["correct"] for c in cases for v in c["per_light"].values()
        ),
        "all_correct": all(
            c["correct"] and all(v["correct"] for v in c["per_light"].values())
            for c in cases
        ),
        "ng_false_ok": sum(
            c["label"] == "NG" and not c["correct"] for c in cases
        ),
        "cases": cases,
    }


def _cases(label: str, correct: int, errors: int):
    return [
        {
            "label": label, "correct": i < correct,
            "per_light": {
                "SIDE": {"score": (
                    .95 if (i < correct) == (label == "NG") else .05
                ), "correct": i < correct},
            },
        }
        for i in range(correct + errors)
    ]


class BalancedReplayTests(CNNV2Fixture):
    def test_balanced_replay_pairs_every_ng_with_ok_and_rotates(self):
        events, _ = archive_labeled_experimental_events(self.manifest_path)
        ng = [e for e in events if e["label"] == "NG"]
        ok = [e for e in events if e["label"] == "OK"]
        self.assertGreaterEqual(len(ng), 8)

        def assess(_model, rows, _size, _focus):
            return _assessment([
                {
                    "label": e["label"], "correct": True,
                    "per_light": {"SIDE": {
                        "score": float(i) / 100, "correct": True,
                    }},
                }
                for i, e in enumerate(rows)
            ])

        with patch(
            "src.scripts.train_faltando_cnn_v2_verified_archive._assess",
            side_effect=assess,
        ):
            first, stats = _balanced_hard_replay(
                ng, ng + ok, None, 64, .70, seed=42, epoch=1, epochs=5
            )
            second, _ = _balanced_hard_replay(
                ng, ng + ok, None, 64, .70, seed=42, epoch=2, epochs=5
            )
        self.assertEqual(stats["NG"], 8)
        self.assertEqual(stats["OK"], 8)
        self.assertEqual(stats["paired_steps"], 8)
        self.assertEqual([e["label"] for e in first], ["NG", "OK"] * 8)
        self.assertNotEqual(
            [e["id"] for e in first[::2]],
            [e["id"] for e in second[::2]],
        )
        self.assertEqual(
            {e["id"] for e in first if e["label"] == "OK"},
            {e["id"] for e in ok[-8:]},
        )

    def test_holdout_floor_97_percent_not_96(self):
        history = _assessment(_cases("OK", 8, 0) + _cases("NG", 8, 0))
        online = _assessment(_cases("OK", 2, 0) + _cases("NG", 8, 0))
        before_good = _assessment(_cases("OK", 50, 0) + _cases("NG", 47, 3))
        after_good = _assessment(_cases("OK", 50, 0) + _cases("NG", 47, 3))
        result = candidate_acceptance(
            history, history, online, online, before_good, after_good,
            minimum_holdout_accuracy=.97,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["holdout_accuracy"], .97)
        self.assertFalse(result["holdout_goal_reached"])
        before_bad = _assessment(_cases("OK", 50, 0) + _cases("NG", 46, 4))
        after_bad = _assessment(_cases("OK", 50, 0) + _cases("NG", 46, 4))
        rejected = candidate_acceptance(
            history, history, online, online, before_bad, after_bad,
            minimum_holdout_accuracy=.97,
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["gates"]["holdout_accuracy_minimum"])

    def test_accuracy_alone_never_overrides_new_false_ng(self):
        anchor = _assessment(_cases("OK", 3, 0) + _cases("NG", 3, 0))
        baseline = _assessment(_cases("OK", 50, 0) + _cases("NG", 40, 10))
        candidate = _assessment(_cases("OK", 49, 1) + _cases("NG", 50, 0))
        result = candidate_acceptance(
            anchor, anchor, anchor, anchor, baseline, candidate,
            minimum_holdout_accuracy=.97,
        )
        self.assertEqual(result["holdout_accuracy"], .99)
        self.assertFalse(result["gates"]["holdout_ok_not_worse"])
        self.assertFalse(result["passed"])

    def test_replay_selection_prioritizes_historical_ok(self):
        good_historical = _assessment(
            _cases("OK", 10, 0) + _cases("NG", 10, 0)
        )
        bad_historical = _assessment(
            _cases("OK", 9, 1) + _cases("NG", 10, 0)
        )
        online_worse = _assessment(
            _cases("OK", 2, 0) + _cases("NG", 7, 3)
        )
        online_better = _assessment(
            _cases("OK", 2, 0) + _cases("NG", 10, 0)
        )
        self.assertLess(
            _selection_score(good_historical, online_worse, good_historical),
            _selection_score(bad_historical, online_better, good_historical),
        )

    def test_operational_review_separately_from_binary_gate(self):
        report = _assessment([
            {"label": "OK", "correct": False,
             "per_light": {"TOP": {"score": .65, "correct": False}}},
            {"label": "OK", "correct": False,
             "per_light": {"TOP": {"score": .98, "correct": False}}},
            {"label": "NG", "correct": False,
             "per_light": {"MID": {"score": .01, "correct": False}}},
        ])
        lights = _operational_light_summary(report)
        self.assertEqual(lights["TOP"]["review"], 1)
        self.assertEqual(lights["TOP"]["ok_as_conclusive_ng"], 1)
        self.assertEqual(lights["MID"]["ng_as_conclusive_ok"], 1)
