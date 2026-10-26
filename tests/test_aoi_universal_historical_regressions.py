"""Offline historical regression comparison: exact identities and counts."""
from __future__ import annotations

import unittest

from src.scripts.audit_aoi_universal_historical_regressions import (
    _identify_regressions,
)


class HistoricalRegressionsTests(unittest.TestCase):
    def setUp(self):
        self.events, self.previous, self.current = [], [], []
        for index in range(106):
            label = "NG" if index < 63 else "OK"
            event_id = f"archive:{index}"
            self.events.append({
                "id": event_id, "label": label,
            })
            self.previous.append({
                "event_id_or_group_hint": event_id,
                "label": label, "predicted": label,
                "ng_score": .95 if label == "NG" else .05,
            })
            self.current.append(dict(self.previous[-1]))
        for i in (11, 27):
            self.current[i]["predicted"] = "OK"
            self.current[i]["ng_score"] = .21
        self.expected = {
            "historical_test_not_blind": {
                "total": 106, "NG_false_OK": 2, "OK_false_NG": 0,
            },
        }

    def review(self):
        return _identify_regressions(
            self.events, {"cases": self.previous},
            {"cases": self.current}, self.expected,
        )

    def test_exact_two_ng_regressions_are_identified(self):
        result = self.review()
        self.assertEqual([r["id"] for r, _, _ in result],
                         ["archive:11", "archive:27"])
        self.assertTrue(all(r["label"] == "NG" for r, _, _ in result))

    def test_drift_of_record_identity_rejected(self):
        self.current[2]["event_id_or_group_hint"] = "other"
        with self.assertRaisesRegex(ValueError, "ordem"):
            self.review()

    def test_a_new_false_ng_is_rejected(self):
        self.current[80]["predicted"] = "NG"
        with self.assertRaisesRegex(ValueError, "duas regressões NG"):
            self.review()

    def test_previous_model_not_perfect_blocks_assumption(self):
        self.previous[10]["predicted"] = "OK"
        with self.assertRaisesRegex(ValueError, "duas regressões NG"):
            self.review()

    def test_wrong_report_totals_block(self):
        self.expected["historical_test_not_blind"]["NG_false_OK"] = 1
        with self.assertRaisesRegex(ValueError, "duas regressões NG"):
            self.review()

    def test_previous_model_no_new_regressions_blocks(self):
        self.current[11]["predicted"] = "NG"
        with self.assertRaisesRegex(ValueError, "duas regressões NG"):
            self.review()


if __name__ == "__main__":
    unittest.main()
