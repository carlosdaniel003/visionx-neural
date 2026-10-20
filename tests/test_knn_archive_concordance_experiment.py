"""Candidato C é inteiramente diagnóstico: sem rótulo consultado nem XP."""
from __future__ import annotations

import unittest

from src.services.startup_regression.knn_archive_concordance_experiment import (
    select_concordance_candidate,
)


def candidate(status):
    return {"status": status}


class KNNAOIConcordanceCandidateTests(unittest.TestCase):
    def combine(self, a, b, pa, pb):
        return select_concordance_candidate(
            candidate(a), candidate(b),
            visual_prediction=pa, geometry_prediction=pb,
        )

    def test_b_can_add_review_when_both_base_predictions_agree(self):
        r = self.combine("REVISAO", "SUGERE_OK", "OK", "OK")
        self.assertEqual(r["status"], "SUGERE_OK")
        self.assertEqual(r["source"], "B_COM_VOTO_A")
        self.assertFalse(r["auto_decision_authorized"])
        self.assertFalse(r["production_approved"])
        self.assertIsNone(r["xp_command"])
        self.assertFalse(r["uses_expected_label"])

    def test_a_kept_if_b_reviews_and_both_baselines_agree(self):
        r = self.combine("SUGERE_NG", "REVISAO", "NG", "NG")
        self.assertEqual(r["status"], "SUGERE_NG")
        self.assertEqual(r["source"], "A_COM_VOTO_B")

    def test_both_agree(self):
        r = self.combine("SUGERE_OK", "SUGERE_OK", "OK", "OK")
        self.assertEqual(r["source"], "A_E_B")

    def test_disagreement_of_base_predictions_forces_review(self):
        for a, b in (("SUGERE_OK", "REVISAO"),
                     ("REVISAO", "SUGERE_OK"),
                     ("SUGERE_OK", "SUGERE_OK")):
            with self.subTest(a=a, b=b):
                r = self.combine(a, b, "OK", "NG")
                self.assertEqual(r["status"], "REVISAO")
                self.assertEqual(r["reason"], "DISCORDANCIA_KNN_A_B")

    def test_direct_conflict_in_suggestions_is_never_emitted(self):
        r = self.combine("SUGERE_OK", "SUGERE_NG", "OK", "OK")
        self.assertEqual(r["status"], "REVISAO")
        self.assertEqual(r["reason"], "SUGESTOES_A_B_CONFLITANTES")

    def test_suggestion_against_agreed_baseline_is_reviewed(self):
        r = self.combine("REVISAO", "SUGERE_NG", "OK", "OK")
        self.assertEqual(r["status"], "REVISAO")
        self.assertEqual(r["reason"], "SUGESTAO_DIVERGE_DO_VOTO_KNN")

    def test_missing_geometry_vote_means_review_not_fallback(self):
        r = select_concordance_candidate(
            candidate("SUGERE_OK"), None,
            visual_prediction="OK", geometry_prediction=None,
        )
        self.assertEqual(r["status"], "REVISAO")
        self.assertEqual(r["reason"], "SEM_DOIS_VOTOS_KNN")

    def test_neither_candidates_decide_means_review(self):
        r = self.combine("REVISAO", "REVISAO", "NG", "NG")
        self.assertEqual(r["status"], "REVISAO")


if __name__ == "__main__":
    unittest.main()
