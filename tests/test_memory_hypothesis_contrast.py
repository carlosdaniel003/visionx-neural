import unittest
from pathlib import Path

from src.core.anomaly_signature import VECTOR_SIZE
from src.core.memory_hypothesis_contrast import (
    HYPOTHESIS_CONFLICT_MARGIN,
    HYPOTHESIS_POLICY,
    HYPOTHESIS_REVIEW_CONFIDENCE,
    _fusion_wrapper_factory,
    analyze_anomaly_hypotheses,
    build_hypothesis_result,
)
from src.ui.production_confidence_gate import production_decision_policy


ROOT = Path(__file__).resolve().parents[1]


def signature(similarity=1.0):
    return {
        "schema": "visionx.anomaly.v1",
        "vector": [0.25] * VECTOR_SIZE,
        "test_similarity": float(similarity),
    }


def record(label, similarity, index=0, occurrences=1):
    return {
        "label": label,
        "path": f"{label}_{index}.json",
        "json_path": f"{label}_{index}.json",
        "mode": "anomaly",
        "anomaly_signature": signature(similarity),
        "prototype_occurrences": occurrences,
        "quantity_influence": False,
    }


def comparator(_query, stored):
    value = float(stored.get("test_similarity", 0.0))
    return value, {
        "dual_scale": True,
        "epicenter_similarity": value,
        "context_similarity": value,
    }


def analyze(ok_records, ng_records):
    return analyze_anomaly_hypotheses(
        signature(),
        ok_records,
        ng_records,
        5,
        "categoria",
        comparator,
    )


class HypothesisSelectionTests(unittest.TestCase):
    def test_strong_ng_beats_weak_ok(self):
        result = analyze(
            [record("OK", 0.74)],
            [record("NG", 0.98)],
        )
        self.assertTrue(result["has_memory"])
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["leading_hypothesis"], "NG")
        self.assertAlmostEqual(result["best_ng_similarity"], 0.98)
        self.assertAlmostEqual(result["best_ok_similarity"], 0.74)
        self.assertAlmostEqual(result["hypothesis_margin"], 0.24)
        self.assertEqual(result["memory_score"], 1.0)
        self.assertFalse(result["memory_conflict"])
        self.assertEqual(result["memory_policy"], HYPOTHESIS_POLICY)

    def test_strong_ok_beats_intermediate_ng(self):
        result = analyze(
            [record("OK", 0.97)],
            [record("NG", 0.82)],
        )
        self.assertTrue(result["has_memory"])
        self.assertEqual(result["best_match_label"], "OK")
        self.assertAlmostEqual(result["best_ok_similarity"], 0.97)
        self.assertAlmostEqual(result["best_ng_similarity"], 0.82)
        self.assertAlmostEqual(result["hypothesis_margin"], 0.15)
        self.assertEqual(result["memory_score"], 0.0)
        self.assertFalse(result["memory_conflict"])

    def test_972_ng_vs_975_ok_is_memory_conflict(self):
        result = analyze(
            [record("OK", 0.975)],
            [record("NG", 0.972)],
        )
        self.assertFalse(result["has_memory"])
        self.assertFalse(result["match_reliable"])
        self.assertTrue(result["memory_conflict"])
        self.assertTrue(result["conflicting_tie"])
        self.assertTrue(result["operator_review_required"])
        self.assertEqual(result["leading_hypothesis"], "OK")
        self.assertAlmostEqual(result["best_ok_similarity"], 0.975)
        self.assertAlmostEqual(result["best_ng_similarity"], 0.972)
        self.assertAlmostEqual(result["hypothesis_margin"], 0.003)
        self.assertLessEqual(
            result["hypothesis_margin"],
            HYPOTHESIS_CONFLICT_MARGIN,
        )
        self.assertEqual(result["memory_score"], 0.5)
        self.assertIn("CONFLITO DE MEMÓRIA", result["memory_reason"])

    def test_margin_above_one_percent_is_not_conflict(self):
        result = analyze(
            [record("OK", 0.975)],
            [record("NG", 0.960)],
        )
        self.assertTrue(result["has_memory"])
        self.assertFalse(result["memory_conflict"])
        self.assertEqual(result["best_match_label"], "OK")
        self.assertAlmostEqual(result["hypothesis_margin"], 0.015)

    def test_low_opposing_hypothesis_does_not_create_conflict(self):
        result = analyze(
            [record("OK", 0.748)],
            [record("NG", 0.756)],
        )
        self.assertTrue(result["has_memory"])
        self.assertFalse(result["memory_conflict"])
        self.assertEqual(result["best_match_label"], "NG")

    def test_only_ng_memory_can_still_be_used(self):
        result = analyze([], [record("NG", 0.96)])
        self.assertTrue(result["has_memory"])
        self.assertFalse(result["ok_memory_available"])
        self.assertTrue(result["ng_memory_available"])
        self.assertIsNone(result["hypothesis_margin"])
        self.assertEqual(result["best_match_label"], "NG")

    def test_only_ok_memory_can_still_be_used(self):
        result = analyze([record("OK", 0.96)], [])
        self.assertTrue(result["has_memory"])
        self.assertTrue(result["ok_memory_available"])
        self.assertFalse(result["ng_memory_available"])
        self.assertIsNone(result["hypothesis_margin"])
        self.assertEqual(result["best_match_label"], "OK")

    def test_prototype_occurrence_count_never_changes_hypothesis(self):
        result = analyze(
            [record("OK", 0.91, index=1, occurrences=400)],
            [record("NG", 0.98, index=1, occurrences=1)],
        )
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["memory_score"], 1.0)
        self.assertFalse(result["quantity_influence"])

    def test_many_ok_candidates_do_not_outvote_closest_ng(self):
        ok_records = [record("OK", 0.90 + index * 0.0001, index) for index in range(100)]
        result = analyze(ok_records, [record("NG", 0.98)])
        self.assertEqual(result["best_match_label"], "NG")
        self.assertAlmostEqual(result["best_ng_similarity"], 0.98)
        self.assertLess(result["best_ok_similarity"], 0.92)
        self.assertFalse(result["quantity_influence"])


class ConflictFusionTests(unittest.TestCase):
    @staticmethod
    def base_fusion(_orchestrator, _detail, _category, _missing, _knn):
        return (
            0.91,
            True,
            0.99,
            "Motores físicos apontaram defeito",
            {
                "final_score": 0.91,
                "confidence": 0.99,
                "physical_score": 0.91,
                "dominant_engine": "structural",
                "fusion_rule": "physical_only",
                "weights": {"physical": 1.0, "knn": 0.0},
                "memory": {},
                "engines": [
                    {
                        "id": "knn",
                        "active": False,
                        "triggered": False,
                        "raw_score": 0.5,
                        "effective_score": 0.5,
                        "selected": False,
                        "final_influence": 0.0,
                        "summary": "",
                    }
                ],
            },
        )

    def test_conflict_preserves_physical_score_but_forces_review_confidence(self):
        wrapped = _fusion_wrapper_factory(self.base_fusion)
        knn = {
            "memory_conflict": True,
            "best_ok_similarity": 0.975,
            "best_ng_similarity": 0.972,
            "hypothesis_margin": 0.003,
        }

        score, defect, confidence, reason, trace = wrapped(
            object(), {}, "FALTANDO", None, knn
        )

        self.assertEqual(score, 0.91)
        self.assertTrue(defect)
        self.assertEqual(confidence, HYPOTHESIS_REVIEW_CONFIDENCE)
        self.assertEqual(trace["physical_score"], 0.91)
        self.assertEqual(trace["weights"], {"physical": 1.0, "knn": 0.0})
        self.assertEqual(trace["fusion_rule"], "memory_conflict_operator_review")
        self.assertTrue(trace["operator_review_required"])
        self.assertEqual(trace["operator_review_reason"], "memory_hypothesis_conflict")
        self.assertTrue(trace["memory"]["memory_conflict"])
        self.assertEqual(trace["memory"]["role"], "CONFLITO DE MEMÓRIA")
        self.assertEqual(trace["engines"][0]["final_influence"], 0.0)
        self.assertIn("Decisão automática bloqueada", reason)

    def test_no_conflict_keeps_original_fusion_unchanged(self):
        wrapped = _fusion_wrapper_factory(self.base_fusion)
        result = wrapped(
            object(), {}, "FALTANDO", None, {"memory_conflict": False}
        )
        self.assertEqual(result, self.base_fusion(None, None, None, None, None))

    def test_conflict_fails_existing_production_99_percent_gate(self):
        wrapped = _fusion_wrapper_factory(self.base_fusion)
        _score, defect, confidence, _reason, trace = wrapped(
            object(),
            {},
            "FALTANDO",
            None,
            {
                "memory_conflict": True,
                "best_ok_similarity": 0.975,
                "best_ng_similarity": 0.972,
                "hypothesis_margin": 0.003,
            },
        )
        analysis = {
            "is_defect": defect,
            "confidence": confidence,
            "detail": {"decision_trace": trace},
        }
        policy = production_decision_policy(analysis)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "NG")


class RawResultTests(unittest.TestCase):
    def test_explicit_fields_exist_for_auditing(self):
        result = build_hypothesis_result(
            distances=[
                (0.03, "OK", "ok.json", {"scale": "dual"}),
                (0.04, "NG", "ng.json", {"scale": "dual"}),
            ],
            top_k=5,
            mode="anomaly",
            scope="categoria",
        )
        self.assertIn("best_ok_similarity", result)
        self.assertIn("best_ng_similarity", result)
        self.assertIn("hypothesis_margin", result)
        self.assertIn("hypotheses", result)
        self.assertEqual(result["hypotheses"]["OK"]["path"], "ok.json")
        self.assertEqual(result["hypotheses"]["NG"]["path"], "ng.json")


class InstallationOrderTests(unittest.TestCase):
    def test_hypothesis_contrast_is_after_prototypes(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_prototype_memory("),
            source.index("install_memory_hypothesis_contrast("),
        )
        self.assertLess(
            source.index("install_memory_hypothesis_contrast("),
            source.index("install_inverted_signature_extension()"),
        )


if __name__ == "__main__":
    unittest.main()
