"""Etapa 3: contraprova física pode vetar OK, nunca gerar NG automático."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.core.physical_missing_veto import (
    POLICY, apply_physical_missing_veto, physical_absence_probe,
)
from src.ui.production_confidence_gate import production_decision_policy


LIGHTS = ("SIDE", "TOP", "MID")


def cnn_ok():
    return {
        "verdict": "FALHA FALSA",
        "is_defect": False,
        "confidence": 0.99,
        "multilight_final": True,
        "eligible_for_final_decision": True,
        "production_review_required": False,
        "detail": {
            "recognition_route": "NEW_CNN",
            "cnn_v2_active": True,
            "cnn_v2_experimental": True,
            "cnn_v2_supervised_auto_eligible": True,
            "cnn_v2_consensus": "OK",
            "cnn_v2_light_votes": dict.fromkeys(LIGHTS, "OK"),
            "cnn_v2_verified_lights": list(LIGHTS),
            "multilight_final": True,
            "eligible_for_final_decision": True,
            "multilight_category": "FALTANDO",
            "final_score": 0.01,
            "decision_trace": {"operator_review_required": False},
        },
    }


def physical_samples():
    ref = np.full((80, 96, 3), 70, dtype=np.uint8)
    test = ref.copy()
    return {
        light: {
            "lighting_mode": light,
            "sample_image": ref.copy(),
            "test_image": test.copy(),
        }
        for light in LIGHTS
    }


def answer(status):
    return {"status": status, "reason": "fixture", "scales": []}


class IndependentPhysicalVetoTests(unittest.TestCase):
    def test_existing_cnn_auto_ok_without_physical_evidence_is_reachable(self):
        self.assertTrue(production_decision_policy(cnn_ok())["auto_allowed"])

    def test_one_suspicious_light_blocks_all_auto_ok_but_does_not_create_ng(self):
        outputs = iter((
            answer("NO_STRONG_EVIDENCE"),
            answer("SUSPECT_ABSENCE"),
            answer("NO_STRONG_EVIDENCE"),
        ))
        fused = apply_physical_missing_veto(
            cnn_ok(), physical_samples(), "FALTANDO", probe=lambda *_: next(outputs)
        )
        self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(fused["is_defect"])
        self.assertTrue(fused["production_review_required"])
        self.assertEqual(fused["detail"]["fusion_rule"], POLICY)
        audit = fused["detail"]["physical_missing_veto"]
        self.assertEqual(audit["suspect_lights"], ["TOP"])
        self.assertFalse(audit["authorizes_ng"])
        self.assertFalse(audit["authorizes_ok"])
        self.assertEqual(production_decision_policy(fused)["proposed_decision"], "")
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])
        json.dumps(audit)  # Sem PNG, ndarray ou outras evidências pesadas

    def test_unavailable_mid_pair_fails_closed(self):
        samples = physical_samples()
        del samples["MID"]
        fused = apply_physical_missing_veto(
            cnn_ok(), samples, "FALTANDO",
            probe=lambda *_: answer("NO_STRONG_EVIDENCE")
        )
        self.assertEqual(
            fused["detail"]["physical_missing_veto"]["unavailable_lights"], ["MID"]
        )
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])

    def test_probe_error_or_invalid_return_cannot_skip_review(self):
        for fn in (
            lambda *_: (_ for _ in ()).throw(RuntimeError("worker failed")),
            lambda *_: {"status": "OK"},
        ):
            with self.subTest(probe=fn):
                fused = apply_physical_missing_veto(
                    cnn_ok(), physical_samples(), "FALTANDO", probe=fn
                )
                self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")
                self.assertEqual(
                    fused["detail"]["physical_missing_veto"]["status"],
                    "UNAVAILABLE"
                )

    def test_clean_pairs_do_not_grant_ok_only_preserve_existing_policy(self):
        fused = apply_physical_missing_veto(
            cnn_ok(), physical_samples(), "FALTANDO",
            probe=lambda *_: answer("NO_STRONG_EVIDENCE")
        )
        self.assertEqual(fused["verdict"], "FALHA FALSA")
        self.assertFalse(fused["detail"]["physical_missing_veto"]["authorizes_ok"])
        self.assertTrue(production_decision_policy(fused)["auto_allowed"])

    def test_second_production_gate_blocks_even_if_verdict_was_overwritten(self):
        fused = apply_physical_missing_veto(
            cnn_ok(), {}, "FALTANDO"
        )
        fused["verdict"] = "FALHA FALSA"
        fused["production_review_required"] = False
        fused["detail"]["operator_review_required"] = False
        fused["detail"]["cnn_v2_supervised_auto_eligible"] = True
        fused["detail"]["decision_trace"]["operator_review_required"] = False
        policy = production_decision_policy(fused)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["reason"], "cnn_faltando_independent_physical_veto")

    def test_never_veto_ng_known_knn_or_other_categories(self):
        for changed in (
            {"verdict": "DEFEITO REAL", "is_defect": True},
            {"detail": {"recognition_route": "KNOWN_KNN"}},
            {"production_review_required": True},
        ):
            fused = cnn_ok()
            if "detail" in changed:
                fused["detail"].update(changed["detail"])
            else:
                fused.update(changed)
            before = deepcopy(fused)
            result = apply_physical_missing_veto(fused, {}, "FALTANDO")
            self.assertEqual(result, before)
            self.assertNotIn("physical_missing_veto", result["detail"])
        for category in ("MUITO ADESIVO", "INVERTIDO", "Unknown"):
            fused = cnn_ok()
            before = deepcopy(fused)
            self.assertEqual(
                apply_physical_missing_veto(fused, {}, category), before
            )

    def test_real_probe_identical_pair_reports_no_strong_absence(self):
        image = np.full((88, 100, 3), (50, 80, 120), dtype=np.uint8)
        image[22:67, 20:77] = (12, 12, 12)
        result = physical_absence_probe(image, image.copy())
        self.assertEqual(result["status"], "NO_STRONG_EVIDENCE")
        self.assertEqual(len(result["scales"]), 2)
        self.assertTrue(all(not x["suspected_absence"] for x in result["scales"]))

    def test_real_probe_rejects_invalid_geometry_instead_of_marking_clear(self):
        one = np.zeros((60, 60, 3), dtype=np.uint8)
        for bad in (None, one[:30], one.astype(np.float32)):
            self.assertEqual(
                physical_absence_probe(one, bad)["status"], "UNAVAILABLE"
            )

    def test_real_probe_uses_independent_roi_metrics_not_cnn_score(self):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        strong = {
            "missing_active": True, "missing_is_defect": True,
            "missing_classification": "CONTEÚDO ESPERADO AUSENTE",
            "missing_score": 0.95, "missing_changed_coverage": 0.8,
            "missing_background_exposure": 0.65,
            "missing_reference_distinctness": 0.60,
            "missing_direct_similarity": 0.20,
            "missing_best_similarity": 0.1,
        }
        with patch(
            "src.core.physical_missing_veto.ROIPatchExpectationExpert.analyze",
            return_value=strong,
        ) as analyze:
            result = physical_absence_probe(image, image.copy())
        self.assertEqual(result["status"], "SUSPECT_ABSENCE")
        self.assertEqual(analyze.call_count, 2)

    def test_contradiction_without_background_exposure_is_not_diagnosed_absence(self):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        strong = {
            "missing_active": True, "missing_is_defect": True,
            "missing_classification": "CONTEÚDO ESPERADO AUSENTE",
            "missing_score": .99, "missing_changed_coverage": .99,
            "missing_background_exposure": .05,
            "missing_reference_distinctness": .80,
            "missing_direct_similarity": .1, "missing_best_similarity": .05,
        }
        with patch(
            "src.core.physical_missing_veto.ROIPatchExpectationExpert.analyze",
            return_value=strong,
        ):
            result = physical_absence_probe(image, image.copy())
        self.assertEqual(result["status"], "NO_STRONG_EVIDENCE")


class MultilightFinalizationContractTests(unittest.TestCase):
    def test_finalizer_runs_physical_guard_before_exposing_current_analysis(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from src.ui import adhesive_multilight_inspection as ui

        panel = SimpleNamespace(
            adhesive_multilight_analyses={},
            adhesive_multilight_learning_samples=physical_samples(),
            current_aoi_info={"category": "FALTANDO"},
            combo_mode=SimpleNamespace(currentText=lambda: "Modo Sombra"),
            lbl_verdict=MagicMock(), lbl_reason=MagicMock(),
            lbl_timer=MagicMock(), btn_start=MagicMock(), btn_skip=MagicMock(),
            capture_start_time=0.0, capture_start_source="network",
            update_brain_status=MagicMock(),
        )
        with patch.object(ui, "fuse_multilight", return_value=cnn_ok()), \\
             patch.object(ui, "finish_cycle"), \\
             patch(
                 "src.core.physical_missing_veto.physical_absence_probe",
                 return_value=answer("SUSPECT_ABSENCE")
             ):
            result = ui._finalize_multilight_decision(panel)
        self.assertIs(result, panel.current_analysis)
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(result["detail"]["physical_missing_veto"]["suspect_lights"],
                         list(LIGHTS))
        self.assertFalse(production_decision_policy(result)["auto_allowed"])


if __name__ == "__main__":
    unittest.main()
