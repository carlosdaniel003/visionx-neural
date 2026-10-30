"""Override supervisionado SIDE: permissao explicita sem treinamento automatico.

Regressoes sao contratos de telemetria, nao prova de seguranca industrial.
"""
from __future__ import annotations

from copy import deepcopy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.side_memory_first import (
    build_side_memory_first,
    verified_side_supervised_probe_override,
)
from src.services.decision_persistence import _is_unverified_automatic
from src.ui.production_autonomy_controller import (
    KNN_C_SUPERVISED_DELAY_MS,
    ProductionAutonomyController,
)
from src.ui.production_confidence_gate import production_decision_policy
from tests.test_side_memory_first import SideMemoryPriorityTests
from tests.test_production_autonomy_controller import _Panel


def _captured_case():
    # Evento 37a68... 13:27:44 — todos os scores de similaridade reais.
    original = SideMemoryPriorityTests()._oct30_case()
    probe = {
        "status": "SUSPECT_ABSENCE",
        "reason": "TWO_SCALE_ROI_CHECK",
        "geometry": {
            "method": "center_trim_no_rescale", "requires_review": True,
            "reference_hw": [318, 570], "test_hw": [317, 567],
            "comparison_hw": [317, 567],
        },
        "scales": [
            {
                "focus_fraction": .58, "suspected_absence": True,
                "classification": "CONTEÚDO ESPERADO AUSENTE",
                "score": .95587, "coverage": .71882, "background": .84268,
                "distinctness": .68519, "direct_similarity": .34293,
                "nearby_similarity": .18123,
            },
            {
                "focus_fraction": .76, "suspected_absence": False,
                "classification": "DIVERGÊNCIA PARCIAL NA ROI",
                "score": .95122, "coverage": .58179, "background": .28904,
                "distinctness": .71981, "direct_similarity": .45413,
                "nearby_similarity": .27839,
            },
        ],
    }
    result = build_side_memory_first(
        original, "FALTANDO", object(), object(),
        probe=lambda *_: deepcopy(probe),
    )
    assert result is not None
    return result


class SupervisedSideReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_requires_human_even_in_continuous_production(self):
        result = _captured_case()
        self.assertTrue(verified_side_supervised_probe_override(result))
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertEqual(
            result["detail"]["side_early_exit"]["skipped_lights"],
            ["TOP", "MID"],
        )
        for armed in (None, False):
            with self.subTest(armed=armed):
                example = deepcopy(result)
                example["detail"]["production_continuous_auto"] = True
                if armed is not None:
                    example["detail"]["side_strong_ok_supervised_armed"] = armed
                policy = production_decision_policy(example)
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")

    def test_explicit_supervision_allows_zero_not_one_and_marks_quarantine(self):
        result = _captured_case()
        result["detail"]["production_continuous_auto"] = True
        result["detail"]["side_strong_ok_supervised_armed"] = True
        policy = production_decision_policy(result)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertFalse(policy["operator_review_required"])
        self.assertTrue(policy["physical_veto_supervised_override"])
        self.assertEqual(
            policy["reason"],
            "side_strong_human_ok_supervised_probe_override_auto_ok",
        )
        result["detail"]["production_decision_policy"] = {
            **policy, "resolution": "automatic",
        }
        self.assertTrue(_is_unverified_automatic({
            "source": "production_auto",
            "analysis": result,
        }))

    def test_unavailable_both_scales_absent_and_bad_geometry_never_override(self):
        base = _captured_case()
        mutations = (
            lambda r: r["detail"]["side_memory_first"]["physical_probe"].update(
                status="UNAVAILABLE", reason="INVALID_PHYSICAL_ROI"),
            lambda r: r["detail"]["side_memory_first"]["physical_probe"][
                "scales"][1].update(suspected_absence=True),
            lambda r: r["detail"]["side_memory_first"]["physical_probe"][
                "geometry"].update(test_hw=[316, 565]),
            lambda r: r["detail"].update(hard_missing_evidence=True),
            lambda r: r["detail"]["side_memory_first"].update(
                archive_source="public/ok_archive/fabricated.png"),
            lambda r: r["detail"]["side_early_exit"].update(
                production_auto_allowed=True),
            lambda r: r["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.97),
        )
        for change in mutations:
            with self.subTest(change=change):
                result = deepcopy(base)
                change(result)
                result["detail"]["production_continuous_auto"] = True
                result["detail"]["side_strong_ok_supervised_armed"] = True
                self.assertFalse(verified_side_supervised_probe_override(result))
                self.assertFalse(
                    production_decision_policy(result)["auto_allowed"]
                )

    def test_controller_arms_only_between_pieces_and_sends_after_five_seconds(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertFalse(controller.side_strong_ok_supervised_armed)
        self.assertTrue(controller.toggle_side_strong_ok_supervised())
        a = _captured_case()
        panel.current_analysis = a
        self.assertTrue(controller.analysis_ready(a))
        self.assertTrue(a["detail"]["side_strong_ok_supervised_armed"])
        self.assertTrue(controller.toggle_side_strong_ok_supervised())
        controller._stage_timer.stop()
        controller._finish_presentation(controller.generation)
        self.assertEqual(controller.state, "auto_decision_pending")
        self.assertEqual(
            controller._stage_remaining_ms, KNN_C_SUPERVISED_DELAY_MS,
        )
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.saved, [("OK", "production_auto")])
        self.assertEqual(panel.auto_records, ["OK"])
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(len(panel.saved), 1)
        panel.close()

    def test_pause_disarms_and_cannot_send_even_if_resumed(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertTrue(controller.toggle_side_strong_ok_supervised())
        a = _captured_case()
        panel.current_analysis = a
        self.assertTrue(controller.analysis_ready(a))
        controller._stage_timer.stop()
        controller._finish_presentation(controller.generation)
        self.assertTrue(controller.toggle_pause())
        self.assertFalse(controller.side_strong_ok_supervised_armed)
        self.assertFalse(a["detail"]["side_strong_ok_supervised_armed"])
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.saved, [])
        self.assertFalse(controller.toggle_pause())
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.saved, [])
        panel.close()

    def test_armed_outside_production_or_with_pending_piece_is_ignored(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        self.assertFalse(controller.toggle_side_strong_ok_supervised())
        panel.combo_mode.setCurrentText("Modo Produção")
        a = _captured_case()
        panel.current_analysis = a
        controller.analysis_ready(a)
        self.assertFalse(controller.toggle_side_strong_ok_supervised())
        self.assertFalse(a["detail"]["side_strong_ok_supervised_armed"])
        panel.close()


if __name__ == "__main__":
    unittest.main()
