"""Regressões de política KNN aproximada: abstenção e shadow sem XP."""
from __future__ import annotations

import unittest


from src.core.knn_similarity_shadow_candidate import (
    select_shadow_candidate, summarize_shadow_candidates,
)


def neigh(label, day, number, sim=.96):
    return {
        "label": label, "similarity": sim,
        "source_pixel_sha256": f"distinct-{label}-{day}-{number}",
        "source_path": f"public/{'ng' if label == 'NG' else 'ok'}_archive/"
                       f"{day}_143{number}_FALTANDO_SIDE.png",
    }


class ShadowCandidatePolicyTests(unittest.TestCase):
    def test_all_ok_with_independent_ng_visual_counterexamples(self):
        nearest = [
            neigh("OK", "2026-10-03", 0),
            neigh("OK", "2026-10-03", 1),
            neigh("OK", "2026-10-07", 2),
            neigh("OK", "2026-10-07", 3),
            neigh("OK", "2026-10-07", 4),
        ]
        result = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_examples={"OK": 2, "NG": 2},
        )
        self.assertEqual(result["status"], "SUGERE_OK")
        self.assertFalse(result["auto_decision_authorized"])
        self.assertIsNone(result["xp_command"])
        missing = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_examples={"OK": 2, "NG": 0},
        )
        self.assertEqual(missing["status"], "REVISAO")
        scarce = select_shadow_candidate(
            nearest, category="FALTANDO", lighting="SIDE",
            scope_examples={"OK": 2, "NG": 1},
        )
        self.assertEqual(scarce["reason"], "NG_SEM_CONTRAPROVA_VISUAL")

    def test_three_ng_distinct_images_can_support_ng_same_day(self):
        nearest = [
            neigh("NG", "2026-10-15", 0),
            neigh("NG", "2026-10-15", 1),
            neigh("NG", "2026-10-15", 2),
            neigh("OK", "2026-10-15", 3),
            neigh("OK", "2026-10-15", 4),
        ]
        out = select_shadow_candidate(
            nearest, category="INVERTIDO", lighting="TOP",
            scope_examples={"OK": 2, "NG": 3},
        )
        self.assertEqual(out["status"], "SUGERE_NG")
        # Cinco entradas idênticas não contam como cinco votos.
        duplicated = [dict(n, source_pixel_sha256="same-pixels")
                      for n in nearest]
        self.assertEqual(select_shadow_candidate(
            duplicated, category="INVERTIDO", lighting="TOP",
            scope_examples={"OK": 2, "NG": 3},
        )["status"], "REVISAO")

    def test_invalid_coverage_and_weak_similarity_fail_closed(self):
        rows = [neigh("OK", "2026-10-03", x, .79) for x in range(5)]
        for support in (None, {}, {"OK": 3, "NG": 2}):
            out = select_shadow_candidate(
                rows, category="FALTANDO", lighting="SIDE",
                scope_examples=support,
            )
            self.assertEqual(out["status"], "REVISAO")
        out = select_shadow_candidate(
            rows, category="FALTANDO", lighting="WRONG",
            scope_examples={"OK": 2, "NG": 2},
        )
        self.assertEqual(out["reason"], "ESCOPO_OCR_OU_LUZ_INVALIDO")

    def test_real_archive_false_ng_majority_is_reviewed_by_weight(self):
        # Caso real do relatório 2026-10-20 14:47: OK 1231 FALTANDO
        # com 3 NG distantes e dois OK mais próximos (votoNG ~0.2285).
        near = [
            ("OK", .96814430), ("OK", .79097875),
            ("NG", .72231997), ("NG", .71974018), ("NG", .71794431),
        ]
        neighbors = [
            {"label": label, "similarity": sim,
             "source_pixel_sha256": f"distinct-{i}"}
            for i, (label, sim) in enumerate(near)
        ]
        out = select_shadow_candidate(
            neighbors, category="FALTANDO", lighting="SIDE",
            scope_examples={"OK": 15, "NG": 45},
        )
        self.assertEqual(out["status"], "REVISAO")
        self.assertEqual(out["reason"], "VOTO_PONDERADO_DISCORDA_DA_MAIORIA_NG")
        self.assertLess(out["weighted_score_ng"], .5)
        self.assertFalse(out["weighted_vote_agrees"])
        self.assertFalse(out["auto_decision_authorized"])

    def test_selective_precision_target_97_is_not_total_archive_accuracy(self):
        # 97 decisões corretas de 100 sugestões, mais 100 revisões:
        # precisão 97%, cobertura 50%, não acurácia 97% em 200 imagens.
        cases = [
            {"expected_label": "OK",
             "shadow_candidate": {"status": "SUGERE_OK"}}
            for _ in range(97)
        ] + [
            {"expected_label": "OK",
             "shadow_candidate": {"status": "SUGERE_NG"}}
            for _ in range(3)
        ] + [
            {"expected_label": "NG",
             "shadow_candidate": {"status": "REVISAO"}}
            for _ in range(100)
        ]
        stats = summarize_shadow_candidates(cases)
        self.assertEqual(stats["correct_suggestions"], 97)
        self.assertEqual(stats["suggestions"], 100)
        self.assertEqual(stats["reviews"], 100)
        self.assertEqual(stats["coverage"], .5)
        self.assertEqual(stats["conditional_precision"], .97)
        self.assertTrue(stats["target_97pct_met"])
        self.assertFalse(summarize_shadow_candidates(
            cases, complete=False)["target_97pct_met"])
        cases[0]["shadow_candidate"]["status"] = "SUGERE_NG"
        self.assertFalse(summarize_shadow_candidates(cases)["target_97pct_met"])
        self.assertFalse(summarize_shadow_candidates([])["target_97pct_met"])
        # Mesmo com precisão acima de 97%, qualquer NG classificado OK
        # reprova o gate de auditoria.
        ng_missed = [{"expected_label": "NG",
                      "shadow_candidate": {"status": "SUGERE_OK"}}]
        self.assertFalse(summarize_shadow_candidates(
            [{"expected_label": "OK", "shadow_candidate": {"status": "SUGERE_OK"}}
             for _ in range(99)] + ng_missed)["target_97pct_met"])

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
