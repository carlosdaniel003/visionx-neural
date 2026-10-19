"""Regressões do encerramento antecipado SIDE >= 97%."""

import math
import unittest

from src.core.side_early_decision import (
    SIDE_EARLY_EXIT_THRESHOLD,
    build_side_early_decision,
)


def _side(verdict="FALHA FALSA", confidence=0.97, **detail):
    ng = verdict == "DEFEITO REAL"
    return {
        "verdict": verdict,
        "is_defect": ng,
        "confidence": confidence,
        "production_review_required": False,
        "lighting_mode": "SIDE",
        "reason": "Resultado local",
        "detail": {
            "recognition_route": "NEW_EXPERTS",
            "decision_trace": {"operator_review_required": False},
            **detail,
        },
    }


class SideEarlyDecisionTests(unittest.TestCase):
    def test_threshold_is_inclusive_for_defect_and_false_failure(self):
        self.assertEqual(SIDE_EARLY_EXIT_THRESHOLD, 0.97)
        for verdict, expected_ng in (
            ("DEFEITO REAL", True),
            ("FALHA FALSA", False),
        ):
            with self.subTest(verdict=verdict):
                original = _side(verdict, 0.97)
                final = build_side_early_decision(original, "FALTANDO")
                self.assertIsNotNone(final)
                self.assertEqual(final["verdict"], verdict)
                self.assertEqual(final["is_defect"], expected_ng)
                self.assertEqual(final["detail"]["side_early_exit"]["analyzed_lights"], ["SIDE"])
                self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"], ["TOP", "MID"])
                self.assertTrue(final["multilight_final"])
                self.assertEqual(final["multilight_category"], "FALTANDO")
                self.assertEqual(final["detail"]["side_early_exit"]["reported_confidence"], 0.97)
                self.assertTrue(final["production_review_required"])
                self.assertTrue(final["detail"]["operator_review_required"])
                self.assertFalse(final["detail"]["side_early_exit"]["production_auto_allowed"])
                self.assertFalse(original["production_review_required"])
                self.assertNotIn("side_early_exit", original["detail"])

    def test_lower_scores_never_skip_other_lights(self):
        for value in (0.0, 0.50, 0.96, 0.9699, float("nan"), float("inf"), -1.0, 1.01, None, "invalid"):
            with self.subTest(value=value):
                self.assertIsNone(build_side_early_decision(_side(confidence=value), "FALTANDO"))

    def test_revisions_conflicts_and_inconsistent_verdict_are_rejected(self):
        cases = [
            _side("REVISÃO OBRIGATÓRIA", 0.99),
            {**_side("DEFEITO REAL", 0.99), "is_defect": False},
            {**_side("FALHA FALSA", 0.99), "production_review_required": True},
            _side(confidence=0.99, operator_review_required=True),
            _side(confidence=0.99, decision_trace={"operator_review_required": True}),
            _side(confidence=0.99, memory_conflict=True),
            _side(confidence=0.99, recognition_route="MEMORY_CONFLICT"),
            _side(confidence=0.99, recognition_route="KNOWN_KNN"),
        ]
        for analysis in cases:
            with self.subTest(case=analysis):
                self.assertIsNone(build_side_early_decision(analysis, "FALTANDO"))

    def test_unverified_or_reviewing_cnn_never_exits(self):
        base = dict(
            recognition_route="NEW_CNN",
            cnn_v2_active=True,
            cnn_v2_status="INFERENCE_OK",
            cnn_v2_checkpoint_verified=True,
        )
        self.assertIsNotNone(build_side_early_decision(_side(confidence=0.98, **base), "FALTANDO"))
        for overrides in (
            {"cnn_v2_checkpoint_verified": False},
            {"cnn_v2_status": "INFERENCE_ERROR"},
            {"cnn_v2_status": "REVIEW"},
            {"cnn_v2_trial_manual_only": True},
            {"cnn_v2_active": False},
        ):
            with self.subTest(overrides=overrides):
                self.assertIsNone(build_side_early_decision(_side(confidence=0.99, **(base | overrides)), "FALTANDO"))

    def test_exact_memory_must_be_human_verified_and_match_verdict(self):
        base = dict(
            recognition_route="KNOWN_KNN",
            recognition_memory_verified=True,
            recognition_match="EXACT_PAIR",
            recognition_known_label="NG",
        )
        self.assertIsNotNone(build_side_early_decision(_side("DEFEITO REAL", 0.99, **base), "FALTANDO"))
        self.assertIsNone(build_side_early_decision(_side("FALHA FALSA", 0.99, **base), "FALTANDO"))
        self.assertIsNone(build_side_early_decision(_side("DEFEITO REAL", 0.99, **(base | {"recognition_memory_verified": False})), "FALTANDO"))


if __name__ == "__main__":
    unittest.main()
