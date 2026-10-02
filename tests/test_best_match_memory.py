import unittest
from unittest.mock import patch

import src.core.anomaly_memory_integration as fusion_module
import src.core.best_match_memory as memory_module
from src.core.anomaly_signature import VECTOR_SIZE
from src.core.experts.knn_expert import KNNExpert


class OrchestratorStub:
    DECISION_CUTOFF = 0.45
    DECISION_SCHEMA = "test"

    @staticmethod
    def _engine_entry(engine_id, label, active, triggered, raw, effective, threshold, summary):
        return {
            "id": engine_id,
            "label": label,
            "active": active,
            "triggered": triggered,
            "raw_score": raw,
            "effective_score": effective,
            "threshold": threshold,
            "selected": False,
            "final_influence": 0.0,
            "summary": summary,
        }


def sig(similarity=1.0):
    return {
        "schema": "visionx.anomaly.v1",
        "vector": [0.25] * VECTOR_SIZE,
        "test_similarity": similarity,
    }


def rec(label, similarity, index):
    return {
        "path": f"{label}_{index}.json",
        "anomaly_signature": sig(similarity),
    }


def compare(_query, stored):
    value = float(stored["test_similarity"])
    return value, {"value": value}


class MemorySelectionTests(unittest.TestCase):
    def analyze(self, ok_items, ng_items):
        expert = KNNExpert.__new__(KNNExpert)
        with patch.object(memory_module, "compare_anomaly_signatures", side_effect=compare):
            return memory_module._analyze_anomaly_memory_best_match(
                expert, sig(), ok_items, ng_items, 5, "categoria"
            )

    def test_single_closest_ng_is_not_overruled_by_many_ok(self):
        result = self.analyze(
            [rec("OK", 0.98, i) for i in range(99)],
            [rec("NG", 0.99, 0)],
        )
        self.assertTrue(result["has_memory"])
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["memory_score"], 1.0)
        self.assertEqual(result["memory_label_counts"], {"OK": 4, "NG": 1})
        self.assertFalse(result["quantity_influence"])

    def test_single_closest_ok_is_not_overruled_by_many_ng(self):
        result = self.analyze(
            [rec("OK", 0.99, 0)],
            [rec("NG", 0.98, i) for i in range(99)],
        )
        self.assertEqual(result["best_match_label"], "OK")
        self.assertEqual(result["memory_score"], 0.0)
        self.assertEqual(result["memory_label_counts"], {"OK": 1, "NG": 4})

    def test_weak_match_has_no_influence(self):
        result = self.analyze([rec("OK", 0.73, 0)], [rec("NG", 0.74, 0)])
        self.assertFalse(result["has_memory"])
        self.assertEqual(result["memory_score"], 0.5)

    def test_conflicting_near_tie_is_inconclusive(self):
        result = self.analyze([rec("OK", 0.989, 0)], [rec("NG", 0.990, 0)])
        self.assertTrue(result["conflicting_tie"])
        self.assertFalse(result["has_memory"])


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = memory_module._best_match_dynamic_fusion_factory(
            fusion_module._dynamic_fusion
        )
        self.orchestrator = OrchestratorStub()

    def run_fusion(self, label, similarity, legacy_vote, detail=None):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": label,
            "best_similarity": similarity,
            "vote_defect": legacy_vote,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        return self.fusion(
            self.orchestrator, detail or {}, "DESLOCADO", None, knn
        )

    def test_strong_ng_ignores_old_low_vote(self):
        score, defect, confidence, reason, trace = self.run_fusion("NG", 0.99, 0.01)
        self.assertEqual(score, 1.0)
        self.assertTrue(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["memory"]["memory_score"], 1.0)
        self.assertFalse(trace["memory"]["quantity_influence"])
        self.assertIn("quantidade de exemplos ignorada", reason)

    def test_strong_ok_ignores_old_high_vote(self):
        score, defect, confidence, _reason, trace = self.run_fusion(
            "OK", 0.99, 0.99, {"silk_error_pct": 0.20}
        )
        self.assertEqual(score, 0.0)
        self.assertFalse(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["memory"]["memory_score"], 0.0)

    def test_real_faltando_exact_ok_witness_beats_raw_hard_missing(self):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": "OK",
            "best_similarity": 0.9999999946355821,
            "best_ok_similarity": 0.9999999946355821,
            "best_ng_similarity": 0.8724773603677749,
            "ng_memory_available": True,
            "hypothesis_margin": 0.1275226342678072,
            "memory_conflict": False,
            "operator_review_required": False,
            "vote_defect": 0.0,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        missing = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_score": 0.979522189040151,
            "missing_tolerance": 0.36,
            "missing_reason": (
                "QUEBRA DA EXPECTATIVA VISUAL DA ROI (98%)"
            ),
            "missing_hard_absence": True,
            "missing_hard_absence_reason": (
                "estrutura esperada colapsou sem correspondência próxima válida"
            ),
        }

        score, defect, confidence, reason, trace = self.fusion(
            self.orchestrator,
            {
                "silk_error_pct": 0.61,
                "semantic_loss": 0.52,
                "local_score": 0.60,
                "ctx_score": 0.0,
                "decision_threshold": 0.45,
                "ssim": 0.26,
                "pct_changed": 0.60,
            },
            "FALTANDO",
            missing,
            knn,
        )

        self.assertEqual(score, 0.0)
        self.assertFalse(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(
            trace["fusion_rule"],
            "hard_missing_exact_ok_witness",
        )
        self.assertEqual(trace["dominant_engine"], "knn")
        self.assertEqual(trace["weights"], {"physical": 0.0, "knn": 1.0})
        self.assertFalse(trace["hard_missing_evidence"])
        self.assertTrue(trace["raw_hard_missing_evidence"])
        self.assertTrue(trace["hard_missing_contradicted_by_exact_ok"])
        self.assertFalse(trace["memory"]["suppressed_by_hard_missing"])
        self.assertEqual(
            trace["memory"]["role"],
            "TESTEMUNHA OK QUASE EXATA",
        )
        self.assertIn("HARD MISSING CONTRADITO", reason)

    def test_hard_missing_cannot_be_vetoed_by_strong_ok_memory(self):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": "OK",
            "best_similarity": 0.99,
            "best_ok_similarity": 0.99,
            "best_ng_similarity": 0.985,
            "hypothesis_margin": 0.005,
            "memory_conflict": True,
            "operator_review_required": True,
            "vote_defect": 0.0,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        missing = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_score": 0.93,
            "missing_tolerance": 0.36,
            "missing_reason": "Componente desapareceu da ROI",
            "missing_hard_absence": True,
            "missing_hard_absence_reason": (
                "conteúdo do gabarito foi substituído pelo fundo da região"
            ),
        }

        score, defect, confidence, reason, trace = self.fusion(
            self.orchestrator,
            {},
            "FALTANDO",
            missing,
            knn,
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "missing_hard_absence")
        self.assertEqual(trace["dominant_engine"], "missing")
        self.assertEqual(trace["weights"], {"physical": 1.0, "knn": 0.0})
        self.assertFalse(trace["operator_review_required"])
        self.assertTrue(trace["hard_missing_evidence"])
        self.assertTrue(
            trace["memory"]["suppressed_by_hard_missing"]
        )
        self.assertEqual(
            trace["memory"]["role"],
            "AUDITORIA — SEM VETO SOBRE AUSÊNCIA FÍSICA",
        )
        self.assertTrue(trace["memory"]["memory_conflict"])
        self.assertFalse(trace["memory"]["operator_review_required"])
        self.assertTrue(trace["memory"]["raw_operator_review_required"])
        self.assertAlmostEqual(trace["memory"]["best_ok_similarity"], 0.99)
        self.assertAlmostEqual(trace["memory"]["best_ng_similarity"], 0.985)
        self.assertIn("AUSÊNCIA FÍSICA FORTE", reason)

    def test_emborcado_hard_absence_cannot_be_vetoed_by_906_ok_memory(self):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": "OK",
            "best_similarity": 0.906379471719265,
            "vote_defect": 0.0,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        guard = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_score": 0.8487753587129022,
            "missing_tolerance": 0.36,
            "missing_reason": (
                "AUSÊNCIA FÍSICA FORTE FORA DA CATEGORIA FALTANDO"
            ),
            "missing_hard_absence": True,
            "missing_cross_category_guard": True,
            "missing_guard_source_category": "EMBORCADO",
            "missing_hard_absence_reason": (
                "aparência do componente desapareceu sem correspondência "
                "próxima, confirmada pelos motores estrutural e semântico"
            ),
        }

        score, defect, confidence, reason, trace = self.fusion(
            self.orchestrator,
            {
                "silk_error_pct": 0.54,
                "semantic_loss": 0.71,
            },
            "EMBORCADO",
            guard,
            knn,
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "missing_hard_absence")
        self.assertEqual(trace["dominant_engine"], "missing")
        self.assertEqual(trace["weights"], {"physical": 1.0, "knn": 0.0})
        self.assertTrue(trace["hard_missing_evidence"])
        self.assertTrue(trace["memory"]["suppressed_by_hard_missing"])
        self.assertEqual(
            trace["memory"]["role"],
            "AUDITORIA — SEM VETO SOBRE AUSÊNCIA FÍSICA",
        )
        self.assertIn("AUSÊNCIA FÍSICA FORTE", reason)

    def test_real_deslocado_false_positive_returns_ok_when_hard_missing_is_blocked(self):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": "OK",
            "best_similarity": 0.9818174609877168,
            "best_ok_similarity": 0.9818174609877168,
            "best_ng_similarity": 0.8975629261136057,
            "vote_defect": 0.0,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        guard = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_score": 0.9885001177301624,
            "missing_tolerance": 0.36,
            "missing_cross_category_guard": True,
            "missing_guard_source_category": "DESLOCADO",
            "missing_hard_absence": False,
            "missing_context_hard_absence": False,
            "missing_hard_absence_reason": (
                "colapso visual extremo sem confirmação física independente"
            ),
        }

        score, defect, confidence, _reason, trace = self.fusion(
            self.orchestrator,
            {
                "silk_error_pct": 0.4867366921844401,
                "semantic_loss": 0.3013883389284213,
            },
            "DESLOCADO",
            guard,
            knn,
        )

        self.assertEqual(score, 0.0)
        self.assertFalse(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "best_match_strong")
        self.assertEqual(trace["dominant_engine"], "knn")
        self.assertFalse(trace["hard_missing_evidence"])
        self.assertFalse(trace["memory"]["suppressed_by_hard_missing"])

    def test_intermediate_uses_best_label_with_partial_weight(self):
        score, defect, confidence, _reason, trace = self.run_fusion("NG", 0.80, 0.01)
        self.assertTrue(defect)
        self.assertGreater(score, self.orchestrator.DECISION_CUTOFF)
        self.assertLess(score, 1.0)
        self.assertLess(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "best_match_intermediate")


class InstallationTests(unittest.TestCase):
    def test_installation_order(self):
        source = open("main.py", encoding="utf-8").read()
        self.assertLess(
            source.index("install_anomaly_memory_integration("),
            source.index("install_best_match_memory("),
        )


if __name__ == "__main__":
    unittest.main()
