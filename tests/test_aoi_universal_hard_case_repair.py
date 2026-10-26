"""Safety properties for the operator-supervised CNN hard-case repair."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.scripts.train_aoi_universal_hard_case_repair import (
    _hard_metrics, _protected_split, _safe_key, _operator_events,
)


class HardCaseRepairTests(unittest.TestCase):
    def test_human_replay_counts_8_ng_missed_0_ok_missed(self):
        outcomes = (
            [{"label": "NG", "predicted": "OK"} for _ in range(8)]
            + [{"label": "NG", "predicted": "NG"} for _ in range(4)]
            + [{"label": "OK", "predicted": "OK"} for _ in range(3)]
        )
        for i, x in enumerate(outcomes):
            x.update({
                "event_id_or_group_hint": f"case_{i}",
                "ng_score": .1, "correct": x["label"] == x["predicted"],
            })
        metrics = _hard_metrics({"cases": outcomes})
        self.assertEqual(metrics["NG_false_OK"], 8)
        self.assertEqual(metrics["OK_false_NG"], 0)
        self.assertEqual(len(metrics["errors"]), 8)

    def test_safety_veto_for_unrelated_false_ok_or_false_ng(self):
        baseline = {"NG_false_OK": 2, "OK_false_NG": 1,
                    "macro_category_balanced_accuracy": .9,
                    "balanced_accuracy": .9}
        verified = {"NG_false_OK": 0, "OK_false_NG": 0}
        self.assertIsNone(_safe_key(
            verified, {**baseline, "NG_false_OK": 3}, baseline,
        ))
        self.assertIsNone(_safe_key(
            verified, {**baseline, "OK_false_NG": 4}, baseline,
        ))
        self.assertIsNotNone(_safe_key(verified, baseline, baseline))

    def test_exact_fence_moves_validation_group_and_test_untouched(self):
        def e(name, fold="train"):
            src = "public/ng_archive/" + name
            return {
                "id": "archive:" + src,
                "sources": [src], "label": "NG",
                "split_group": "AOI_MINUTE:" + name[:15],
            }
        first = e("2026-10-14_0852_INVERTIDO_MID.png")
        same_minute_other_category = e("2026-10-14_0852_FALTANDO_SIDE.png")
        original_train = e("2026-10-14_1400_FALTANDO_TOP.png")
        unrelated_validation = e("2026-10-15_1437_FALTANDO_SIDE.png")
        unrelated_validation_ok = dict(
            e("2026-10-15_1438_FALTANDO_SIDE.png"), label="OK",
        )
        original_test = e("2026-10-15_1513_FALTANDO_SIDE.png")
        split = ( [original_train],
                  [first, same_minute_other_category, unrelated_validation, unrelated_validation_ok],
                  [original_test],
                  {"physical_event_association_verified": False})
        with patch(
            "src.scripts.train_aoi_universal_hard_case_repair.split_universal_events",
            return_value=split,
        ):
            train, val, test, report = _protected_split(
                [], {first["sources"][0]: {"quality_label": "NG"}}, seed=42,
            )
        self.assertEqual(len(test), 1)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 2)  # original train + same-minute other category
        self.assertEqual(report["validation_rows_reassigned_to_train"], 2)
        self.assertEqual(report["confirmed_old_rows_removed_for_replacement"], 1)
        self.assertFalse(report["new_independent_holdout"])

    def test_existing_test_group_contaminated_fails_closed(self):
        a = {
            "id": "x",
            "sources": ["public/ng_archive/2026-10-14_0852_INVERTIDO_MID.png"],
            "label": "NG", "split_group": "AOI_MINUTE:2026-10-14_0852",
        }
        with patch(
            "src.scripts.train_aoi_universal_hard_case_repair.split_universal_events",
            return_value=([a], [a], [a], {}),
        ):
            with self.assertRaisesRegex(ValueError, "contamina teste"):
                _protected_split(
                    [a], {a["sources"][0]: {"quality_label": "NG"}}, seed=42,
                )

    def test_operator_quality_preserved_not_derived_from_aoi_type(self):
        rows = [
            {"id": "public/ng_archive/2026-10-21_1414_FALTANDO.png",
             "label": "OK", "views": {"SIDE": {"reference": "r", "test": "t"}}},
            {"id": "public/ng_archive/2026-10-21_1415_EMBORCADO_TOP.png",
             "label": "NG", "views": {"TOP": {"reference": "r", "test": "t"}}},
        ]
        result = _operator_events(rows)
        self.assertEqual([r["label"] for r in result], ["OK", "NG"])
        self.assertEqual([r["category_hint"] for r in result], ["FALTANDO", "EMBORCADO"])
        self.assertTrue(all(not x["event_link_verified"] for x in result))


if __name__ == "__main__":
    unittest.main()
