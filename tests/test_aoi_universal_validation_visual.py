"""Fail-closed diagnosis of mismatched provisional archive labels."""
from __future__ import annotations

import unittest

from src.scripts.audit_aoi_universal_validation_visual import (
    _rank_cases,
)


class VisualQueueTests(unittest.TestCase):
    def setUp(self):
        self.records = [{
            "id": f"event{i:02d}", "label": "NG" if i < 5 else "OK",
        } for i in range(12)]
        self.outcomes = [{
            "event_id_or_group_hint": row["id"],
            "label": row["label"],
            "predicted": (
                "OK" if i < 3 else "NG" if i >= 5 and i < 11
                else row["label"]
            ),
            "ng_score": .2 if i < 3 else .8,
            "correct": (i >= 3 and i < 5) or i == 11,
        } for i, row in enumerate(self.records)]
        self.report = {"validation_selected": {
            "total": 12, "NG_correct": 2, "NG_false_OK": 3,
            "OK_correct": 1, "OK_false_NG": 6,
        }}

    def test_queue_exactly_nine_disagreements(self):
        items = _rank_cases(
            self.records, {"cases": self.outcomes}, self.report,
        )
        self.assertEqual(len(items), 9)
        self.assertEqual(
            sum(x["label"] == "NG" for _, x in items), 3,
        )
        self.assertEqual(
            sum(x["label"] == "OK" for _, x in items), 6,
        )

    def test_blocks_changed_validation_metrics(self):
        report = {"validation_selected": {
            **self.report["validation_selected"], "NG_false_OK": 2,
        }}
        with self.assertRaisesRegex(ValueError, "Divergência"):
            _rank_cases(self.records, {"cases": self.outcomes}, report)

    def test_blocks_missing_case(self):
        with self.assertRaisesRegex(ValueError, "incompleto"):
            _rank_cases(self.records, {"cases": self.outcomes[:-1]}, self.report)

    def test_blocks_order_drift(self):
        cases = [dict(x) for x in self.outcomes]
        cases[0]["event_id_or_group_hint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "não corresponde"):
            _rank_cases(self.records, {"cases": cases}, self.report)


if __name__ == "__main__":
    unittest.main()
