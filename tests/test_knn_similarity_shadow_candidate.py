"""Regressões de política KNN aproximada: abstenção e shadow sem XP."""
from __future__ import annotations

import unittest


from src.core.knn_similarity_shadow_candidate import (
    select_shadow_candidate, summarize_shadow_candidates,
)


def neigh(label, day, number, sim=.96):
    return {
        "label": label, "similarity": sim,
        "source_path": f"public/{'ng' if label == 'NG' else 'ok'}_archive/"
                       f"{day}_143{number}_FALTANDO_SIDE.png",
    }


class ShadowCandidatePolicyTests(unittest.TestCase):
    def test_all_ok_with_ng_counterexamples_across_days(self):
        nearest = [
            neigh("OK", "2026-10-03", 0),
            neigh("OK", "2026-10-03", 1),
            neigh("OK", "2026-10-07", 2),
            neigh("OK", "2026-10-07", 3),
            neigh("OK", "2026-10-07", 4),
        ]
        result = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_days={"OK": 2, "NG": 2},
        )
        self.assertEqual(result["status"], "SUGERE_OK")
        self.assertFalse(result["auto_decision_authorized"])
        self.assertIsNone(result["xp_command"])
        missing = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_days={"OK": 2, "NG": 0},
        )
        self.assertEqual(missing["status"], "REVISAO")
        scarce = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_days={"OK": 2, "NG": 1},
        )
        self.assertEqual(scarce["reason"], "NG_SEM_COBERTURA_INDEPENDENTE")

    def test_three_ng_across_days_supports_ng_suggestion_only(self):
        nearest = [
            neigh("NG", "2026-10-14", 0),
            neigh("NG", "2026-10-15", 1),
            neigh("NG", "2026-10-15", 2),
            neigh("OK", "2026-10-08", 3),
            neigh("OK", "2026-10-08", 4),
        ]
        out = select_shadow_candidate(
            nearest, category="INVERTIDO", lighting="TOP",
            scope_days={"OK": 1, "NG": 2},
        )
        self.assertEqual(out["status"], "SUGERE_NG")
        single_day = [dict(n, source_path=n["source_path"].replace(
            "2026-10-14", "2026-10-15")) for n in nearest]
        self.assertEqual(select_shadow_candidate(
            single_day, category="INVERTIDO", lighting="TOP",
            scope_days={"OK": 1, "NG": 1},
        )["status"], "REVISAO")

    def test_invalid_coverage_and_weak_similarity_fail_closed(self):
        rows = [neigh("OK", "2026-10-03", x, .93) for x in range(5)]
        for support in (None, {}, {"OK": 3, "NG": 2}):
            out = select_shadow_candidate(
                rows, category="FALTANDO", lighting="SIDE",
                scope_days=support,
            )
            self.assertEqual(out["status"], "REVISAO")
        out = select_shadow_candidate(
            rows, category="FALTANDO", lighting="WRONG",
            scope_days={"OK": 2, "NG": 2},
        )
        self.assertEqual(out["reason"], "ESCOPO_OCR_OU_LUZ_INVALIDO")

    def test_summary_separates_missed_ng_from_reviews(self):
        cases = [
            {"expected_label": "NG",
             "shadow_candidate": {"status": "REVISAO"}},
            {"expected_label": "NG",
             "shadow_candidate": {"status": "SUGERE_OK"}},
            {"expected_label": "OK",
             "shadow_candidate": {"status": "SUGERE_OK"}},
        ]
        out = summarize_shadow_candidates(cases)
        self.assertEqual(out["NG_suggested_OK"], 1)
        self.assertEqual(out["reviews"], 1)
        self.assertEqual(out["by_label"]["NG"]["incorrect_suggestions"], 1)



if __name__ == "__main__":
    unittest.main()
