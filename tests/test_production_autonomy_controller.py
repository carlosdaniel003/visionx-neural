import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QComboBox, QScrollArea, QWidget

from src.ui.production_autonomy_controller import (
    VERDICT_HOLD_MS,
    ProductionAutonomyController,
)


ROOT = Path(__file__).resolve().parents[1]


def _complete_binary_ng():
    return {
        "verdict": "DEFEITO REAL",
        "is_defect": True,
        "multilight_final": True,
        "production_review_required": False,
        "detail": {
            "multilight_final": True,
            "recognition_light_routes": {
                light: "NEW_CNN" for light in ("SIDE", "TOP", "MID")
            },
        },
    }


class _Panel(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1000, 700)
        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(
            ["Modo Teste", "Modo Sombra", "Modo Produção"]
        )
        self.root_scroll = QScrollArea(self)
        self.root_scroll.resize(500, 300)
        content = QWidget()
        content.resize(500, 1800)
        self.root_scroll.setWidget(content)

        self.current_analysis = None
        self.production_review_pending = False
        self.production_review_policy = None
        self.last_analysis_time_seconds = 0.0
        self.status = []
        self.saved = []
        self.feedback = []
        self.session_restores = 0
        self.session_visible = False
        self.auto_records = []
        self.manual_records = []
        self.analysis_records = []
        self.paused_records = []
        self.judgments = []
        self.interventions = []
        self.intervention_clears = 0
        self.visible_verdicts = []
        self.confidence_paints = []

    def update_brain_status(self, message, active=False):
        self.status.append((str(message), bool(active)))

    def _update_confidence_panel(self, analysis):
        self.confidence_paints.append(analysis)

    def show_ai_verdict_feedback(self, analysis):
        from src.ui.decision_verdict_feedback import verdict_feedback_state
        self.visible_verdicts.append(verdict_feedback_state(analysis))

    def save_label(self, decision, source="button"):
        self.saved.append((str(decision), str(source)))
        self.last_decision_command_success = True
        self.current_analysis = None

    def show_decision_key_feedback(self, decision, source=""):
        self.feedback.append((str(decision), str(source)))
        return True

    def restore_production_daily_session_feedback(self):
        self.session_restores += 1
        self.session_visible = False

    def show_production_session_feedback(self):
        self.session_visible = True

    def hide_production_session_feedback(self):
        self.session_visible = False

    def record_production_analysis_feedback(self, elapsed):
        self.analysis_records.append(float(elapsed))

    def record_production_automatic_feedback(self, decision):
        self.auto_records.append(str(decision))

    def record_production_manual_feedback(self, decision):
        self.manual_records.append(str(decision))

    def record_production_judgment_feedback(self, decision, origin, elapsed):
        self.judgments.append((str(decision), str(origin), float(elapsed)))
        if origin == "automatic":
            self.auto_records.append(str(decision))
        else:
            self.manual_records.append(str(decision))
        if elapsed > 0:
            self.analysis_records.append(float(elapsed))

    def set_production_paused_feedback(self, paused):
        self.paused_records.append(bool(paused))

    def show_production_intervention_feedback(self, reason):
        self.interventions.append(str(reason))

    def clear_production_intervention_feedback(self):
        self.intervention_clears += 1


class ProductionAutonomyControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_entering_production_restores_daily_session_without_showing_idle_card(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)

        panel.combo_mode.setCurrentText("Modo Produção")
        self.app.processEvents()

        self.assertTrue(controller.is_production())
        self.assertEqual(panel.session_restores, 1)
        self.assertFalse(panel.session_visible)

    def test_switching_modes_restores_same_daily_session_instead_of_resetting(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)

        panel.combo_mode.setCurrentText("Modo Produção")
        panel.combo_mode.setCurrentText("Modo Teste")
        panel.combo_mode.setCurrentText("Modo Produção")

        self.assertEqual(panel.session_restores, 2)
        self.assertFalse(hasattr(panel, "session_resets"))

    def test_cycle_started_shows_session_only_in_production(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)

        self.assertFalse(controller.cycle_started())
        self.assertFalse(panel.session_visible)

        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertTrue(controller.cycle_started())
        self.assertTrue(panel.session_visible)

    def test_binary_gate_is_armed_before_first_production_render(self):
        from src.ui.decision_verdict_feedback import verdict_feedback_state

        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        analysis = {
            "verdict": "REVISÃO OBRIGATÓRIA",
            "is_defect": False,
            "multilight_final": True,
            "production_review_required": True,
            "detail": {
                "multilight_final": True,
                "recognition_light_routes": {
                    light: "NEW_CNN" for light in ("SIDE", "TOP", "MID")
                },
                "decision_trace": {"operator_review_required": True},
            },
        }

        # Teste/Sombra não ativam este gate.
        self.assertFalse(controller.prepare_binary_presentation(analysis))
        self.assertNotIn("production_binary_auto", analysis["detail"])
        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertTrue(controller.prepare_binary_presentation(analysis))
        self.assertEqual(
            verdict_feedback_state(analysis), ("DEFEITO REAL", "ng")
        )
        self.assertTrue(analysis["production_review_required"])
        self.assertEqual(analysis["verdict"], "REVISÃO OBRIGATÓRIA")
        # O veredito de origem fica preservado para auditoria; a tela
        # apresenta apenas o 0/1 decidido pela política binária.
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertIs(controller.pending_analysis, analysis)
        panel.close()

    def test_analysis_ready_records_pre_decision_analysis_time_once(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
        }
        panel.current_analysis = analysis
        panel.last_analysis_time_seconds = 1.37

        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(panel.analysis_records, [])
        self.assertEqual(panel.judgments, [])
        # A mesma análise, reapresentada, não deve alterar os indicadores.
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(panel.analysis_records, [])
        self.assertEqual(panel.judgments, [])

    def test_only_confirmed_auto_or_manual_0_1_commits_timing_once(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        a = {"verdict": "FALHA FALSA", "is_defect": False}
        panel.current_analysis = a
        panel.last_analysis_time_seconds = 4.0
        self.assertTrue(controller.analysis_ready(a))
        self.assertEqual(panel.judgments, [])
        gen = controller.generation
        controller._stage_timer.stop()
        controller._emit_auto_decision(gen, "OK")
        self.assertEqual(panel.judgments, [("OK", "automatic", 4.0)])
        controller._emit_auto_decision(gen, "OK")
        self.assertEqual(len(panel.judgments), 1)

        b = {"verdict": "DEFEITO REAL", "is_defect": True}
        panel.current_analysis = b
        panel.last_analysis_time_seconds = 6.0
        self.assertTrue(controller.analysis_ready(b))
        self.assertEqual(len(panel.judgments), 1)
        controller.operator_decision_completed("NG", source="xp_keyboard")
        self.assertEqual(panel.judgments[-1], ("NG", "manual", 6.0))
        self.assertEqual(panel.analysis_records, [4.0, 6.0])
        panel.close()

    def test_unjudged_cancel_discards_time_and_failed_auto_never_counts(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        a = {"verdict": "FALHA FALSA", "is_defect": False}
        panel.current_analysis = a
        panel.last_analysis_time_seconds = 99.0
        self.assertTrue(controller.analysis_ready(a))
        controller.cancel_current()
        self.assertEqual(panel.judgments, [])

        b = {"verdict": "FALHA FALSA", "is_defect": False}
        panel.current_analysis = b
        panel.last_analysis_time_seconds = 2.25
        self.assertTrue(controller.analysis_ready(b))
        def failed_save(_decision, source="button"):
            panel.last_decision_command_success = False
            return False
        panel.save_label = failed_save
        controller._stage_timer.stop()
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.judgments, [])
        self.assertEqual(panel.analysis_records, [])
        panel.close()

    def test_space_pause_stops_and_resumes_pending_stage(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
        }
        panel.current_analysis = analysis

        controller.analysis_ready(analysis)
        self.assertTrue(controller._stage_timer.isActive())

        self.assertTrue(controller.toggle_pause())
        self.assertTrue(controller.paused)
        self.assertFalse(controller._stage_timer.isActive())
        self.assertTrue(panel.production_autonomy_paused)
        self.assertIn(True, panel.paused_records)

        self.assertFalse(controller.toggle_pause())
        self.assertFalse(controller.paused)
        self.assertTrue(controller._stage_timer.isActive())
        self.assertFalse(panel.production_autonomy_paused)

    def test_paused_manual_zero_cancels_pending_auto_ng_without_advancing_twice(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "multilight_final": True, "production_review_required": False,
            "detail": {
                "multilight_final": True,
                "recognition_light_routes": {
                    light: "NEW_CNN" for light in ("SIDE", "TOP", "MID")
                },
            },
        }
        panel.current_analysis = analysis
        panel.last_analysis_time_seconds = 2.5
        self.assertTrue(controller.analysis_ready(analysis))
        generation = controller.generation
        self.assertFalse(controller.paused_manual_decision("OK"))
        self.assertTrue(controller.toggle_pause())
        self.assertTrue(controller.paused_manual_decision("OK"))
        self.assertEqual(panel.saved, [("OK", "button")])
        self.assertTrue(panel.production_review_pending)
        self.assertEqual(controller.state, "operator_review")
        self.assertFalse(controller._stage_timer.isActive())
        controller._emit_auto_decision(generation, "NG")
        self.assertEqual(panel.saved, [("OK", "button")])
        self.assertFalse(controller.paused_manual_decision("OK"))
        controller.operator_decision_completed("OK", source="button")
        self.assertEqual(panel.judgments, [("OK", "manual", 2.5)])
        self.assertTrue(controller.paused)  # Retoma somente com Space
        self.assertIsNone(controller.pending_analysis)
        self.assertFalse(controller.paused_manual_decision("OK"))

    def test_manual_correction_rejects_incomplete_or_stale_capture(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "multilight_final": False,
            "detail": {"multilight_final": False},
        }
        panel.current_analysis = analysis
        controller.analysis_ready(analysis)
        controller.toggle_pause()
        self.assertFalse(controller.paused_manual_decision("OK"))
        self.assertEqual(panel.saved, [])
        panel.current_analysis = {"verdict": "DEFEITO REAL"}
        self.assertFalse(controller.paused_manual_decision("NG"))
        self.assertEqual(panel.saved, [])

    def test_reinspection_of_judged_xp_event_creates_second_auto_judgment(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")

        original = _complete_binary_ng()
        panel.current_analysis = original
        panel.last_analysis_time_seconds = 4.0
        self.assertTrue(controller.analysis_ready(original))
        first_generation = controller.generation
        controller._stage_timer.stop()
        controller._emit_auto_decision(first_generation, "NG")
        self.assertEqual(panel.saved, [("NG", "production_auto")])
        self.assertEqual(panel.judgments, [("NG", "automatic", 4.0)])

        panel.network_reinspection_of_judged_event = True
        repeated = _complete_binary_ng()
        panel.current_analysis = repeated
        panel.last_analysis_time_seconds = 2.0
        self.assertTrue(controller.analysis_ready(repeated))
        second_generation = controller.generation
        self.assertEqual(controller.state, "rendered_wait")
        self.assertIs(controller.pending_analysis, repeated)
        self.assertTrue(controller._stage_timer.isActive())
        controller._stage_timer.stop()
        controller._emit_auto_decision(second_generation, "NG")

        self.assertEqual(panel.saved, [
            ("NG", "production_auto"), ("NG", "production_auto"),
        ])
        self.assertEqual(panel.judgments, [
            ("NG", "automatic", 4.0), ("NG", "automatic", 2.0),
        ])
        controller._emit_auto_decision(second_generation, "NG")
        controller._emit_auto_decision(first_generation, "NG")
        self.assertEqual(len(panel.saved), 2)
        panel.close()

    def test_pause_before_analysis_keeps_result_waiting_until_resume(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        controller.toggle_pause()

        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
        }
        panel.current_analysis = analysis

        controller.analysis_ready(analysis)

        self.assertTrue(controller.paused)
        self.assertFalse(controller._stage_timer.isActive())
        self.assertTrue(callable(controller._stage_callback))

        controller.toggle_pause()
        self.assertTrue(controller._stage_timer.isActive())

    def test_test_mode_is_untouched(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
            "confidence": 0.2,
        }
        panel.current_analysis = analysis

        self.assertFalse(controller.analysis_ready(analysis))
        self.assertFalse(controller.toggle_pause())
        self.assertEqual(panel.saved, [])
        self.assertEqual(panel.feedback, [])

    def test_false_failure_emits_auto_ok_and_counts_it(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
            "confidence": 0.1,
        }
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1
        controller.cycle_started()

        controller._emit_auto_ok(controller.generation)

        self.assertEqual(
            panel.saved,
            [("OK", "production_auto")],
        )
        self.assertEqual(panel.auto_records, ["OK"])
        self.assertEqual(panel.manual_records, [])
        self.assertEqual(panel.feedback, [("OK", "production_auto")])
        self.assertFalse(panel.session_visible)
        self.assertEqual(controller.state, "idle")

    def test_v3_side_verified_memory_emits_zero_after_one_second(self):
        from src.core.neural.aoi_v3_epoch5_contract import (
            EPOCH5_VOTE_SCHEMA, SNAPSHOT_SHA256,
        )
        from tests.test_side_memory_first import SideMemoryPriorityTests

        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = SideMemoryPriorityTests().decide()
        self.assertIsNotNone(analysis)
        analysis["detail"].update(
            cnn_v3_primary=True,
            aoi_v3_trial={
                "schema": EPOCH5_VOTE_SCHEMA, "status": "READY",
                "epoch": 5, "checkpoint_sha256": SNAPSHOT_SHA256,
                "light": "SIDE", "ng_score_uncalibrated": .000001,
                "predicted": "OK", "ng_veto": False,
                "automatic_ok_from_v3_authorized": False,
            },
        )
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(controller._stage_remaining_ms, VERDICT_HOLD_MS)
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()

        generation = controller.generation
        controller._finish_presentation(generation)
        self.assertEqual(controller.state, "auto_decision_pending")
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()

        controller._emit_auto_decision(generation, "OK")
        self.assertEqual(panel.saved, [("OK", "production_auto")])
        self.assertEqual(panel.auto_records, ["OK"])
        controller._emit_auto_decision(generation, "OK")
        self.assertEqual(len(panel.saved), 1)

    def test_side_97_decides_only_after_presentation_and_once_per_cycle(self):
        from src.core.side_early_decision import build_side_early_decision

        for verdict, decision in (("FALHA FALSA", "OK"), ("DEFEITO REAL", "NG")):
            with self.subTest(verdict=verdict):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                side = {
                    "verdict": verdict,
                    "is_defect": decision == "NG",
                    "confidence": 0.98,
                    "production_review_required": False,
                    "detail": {
                        "recognition_route": "NEW_EXPERTS",
                        "decision_trace": {"operator_review_required": False},
                    },
                }
                analysis = build_side_early_decision(side, "FALTANDO")
                panel.current_analysis = analysis
                controller.pending_analysis = analysis
                controller.generation += 1
                generation = controller.generation

                controller._finish_presentation(generation)
                self.assertEqual(controller.state, "auto_decision_pending")
                self.assertEqual(panel.saved, [])
                self.assertTrue(controller._stage_timer.isActive())
                controller._stage_timer.stop()

                controller._emit_auto_decision(generation, decision)
                self.assertEqual(panel.saved, [(decision, "production_auto")])
                self.assertEqual(panel.auto_records, [decision])
                self.assertFalse(panel.production_review_pending)

                # Even if an expired timer fires, never send the same command twice.
                controller._emit_auto_decision(generation, decision)
                self.assertEqual(panel.saved, [(decision, "production_auto")])

    def test_trial_side_ok_and_ng_dispatch_next_qt_cycle_only_once(self):
        from src.core.side_early_decision import build_side_early_decision
        from tests.test_side_early_decision import SideEarlyDecisionTests

        for decision in ("OK", "NG"):
            with self.subTest(decision=decision):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                side = SideEarlyDecisionTests._trial_ng_side()
                if decision == "OK":
                    side.update(
                        verdict="FALHA FALSA", is_defect=False, confidence=.9997)
                    side["detail"]["cnn_v2_ng_score_uncalibrated"] = .0003
                final = build_side_early_decision(side, "FALTANDO")
                panel.current_analysis = final
                controller.pending_analysis = final
                controller.generation += 1
                generation = controller.generation

                controller._finish_presentation(generation)
                self.assertEqual(controller.state, "auto_decision_pending")
                self.assertEqual(controller._stage_remaining_ms, 1)
                self.assertEqual(panel.saved, [])
                controller._stage_timer.stop()

                controller._emit_auto_decision(generation, decision)
                self.assertEqual(panel.saved, [(decision, "production_auto")])
                self.assertEqual(panel.auto_records, [decision])
                controller._emit_auto_decision(generation, decision)
                self.assertEqual(panel.saved, [(decision, "production_auto")])

    def test_trial_side_pause_or_changed_score_blocks_pending_command(self):
        from src.core.side_early_decision import build_side_early_decision
        from tests.test_side_early_decision import SideEarlyDecisionTests

        for change in ("pause", "alter_score", "mode"):
            with self.subTest(change=change):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                final = build_side_early_decision(
                    SideEarlyDecisionTests._trial_ng_side(), "FALTANDO")
                panel.current_analysis = final
                controller.pending_analysis = final
                controller.generation += 1
                generation = controller.generation
                controller._finish_presentation(generation)
                controller._stage_timer.stop()
                if change == "pause":
                    controller.toggle_pause()
                elif change == "alter_score":
                    final["detail"]["cnn_v2_ng_score_uncalibrated"] = .2
                else:
                    panel.combo_mode.setCurrentText("Modo Teste")
                controller._emit_auto_decision(generation, "NG")
                self.assertEqual(panel.saved, [])
                self.assertEqual(panel.auto_records, [])

    def test_failed_auto_command_is_not_counted_and_requires_operator(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "FALHA FALSA",
            "is_defect": False,
            "confidence": 0.3,
        }
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1

        def failed_save(decision, source="button"):
            panel.saved.append((decision, source))
            panel.last_decision_command_success = False
            return False

        panel.save_label = failed_save

        controller._emit_auto_ok(controller.generation)

        self.assertEqual(panel.auto_records, [])
        self.assertTrue(panel.production_review_pending)
        self.assertEqual(controller.state, "operator_review")
        self.assertEqual(
            panel.interventions[-1],
            "REVISÃO OBRIGATÓRIA",
        )

    def test_real_defect_never_emits_auto_ng(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "confidence": 1.0,
        }
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1

        controller._finish_presentation(controller.generation)

        self.assertEqual(panel.saved, [])
        self.assertEqual(panel.auto_records, [])
        self.assertTrue(panel.production_review_pending)
        self.assertEqual(controller.state, "operator_review")
        self.assertEqual(panel.interventions[-1], "DEFEITO REAL")

    def test_review_never_emits_zero_or_one(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = {
            "verdict": "REVISÃO OBRIGATÓRIA",
            "is_defect": False,
            "production_review_required": True,
        }
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1

        controller._finish_presentation(controller.generation)

        self.assertEqual(panel.saved, [])
        self.assertTrue(panel.production_review_pending)
        self.assertEqual(
            panel.interventions[-1],
            "REVISÃO OBRIGATÓRIA",
        )

    def test_operator_decision_counts_manual_failure_and_hides_session(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        controller.cycle_started()
        controller.state = "operator_review"
        controller.pending_analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
        }

        controller.operator_decision_completed(
            "NG",
            source="xp_keyboard",
        )

        self.assertEqual(controller.state, "idle")
        self.assertIsNone(controller.pending_analysis)
        self.assertEqual(panel.manual_records, ["NG"])
        self.assertEqual(panel.auto_records, [])
        self.assertFalse(panel.session_visible)
        self.assertGreaterEqual(panel.intervention_clears, 1)

    def test_verdict_is_shown_then_auto_ng_after_one_second_without_scrolling(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        bar = panel.root_scroll.verticalScrollBar()
        bar.setValue(230)
        original_scroll = bar.value()
        analysis = _complete_binary_ng()
        panel.current_analysis = analysis

        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(panel.visible_verdicts, [("DEFEITO REAL", "ng")])
        self.assertEqual(controller._stage_remaining_ms, VERDICT_HOLD_MS)
        self.assertEqual(bar.value(), original_scroll)
        self.assertEqual(panel.saved, [])
        QTest.qWait(300)
        self.assertEqual(panel.saved, [])
        self.assertEqual(bar.value(), original_scroll)

        QTest.qWait(850)
        self.assertEqual(panel.saved, [("NG", "production_auto")])
        self.assertEqual(panel.auto_records, ["NG"])
        self.assertEqual(bar.value(), original_scroll)
        self.assertEqual(controller.state, "idle")
        panel.close()

    def test_verdict_ok_and_ng_use_same_one_second_timer(self):
        for label in ("OK", "NG"):
            with self.subTest(label=label):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                if label == "OK":
                    from src.core.side_early_decision import build_side_early_decision
                    from tests.test_side_early_decision import SideEarlyDecisionTests
                    side = SideEarlyDecisionTests._trial_ng_side()
                    side.update(
                        verdict="FALHA FALSA", is_defect=False,
                        confidence=.9997,
                    )
                    side["detail"]["cnn_v2_ng_score_uncalibrated"] = .0003
                    analysis = build_side_early_decision(side, "FALTANDO")
                else:
                    analysis = _complete_binary_ng()
                panel.current_analysis = analysis
                self.assertTrue(controller.analysis_ready(analysis))
                self.assertEqual(controller._stage_remaining_ms, 1000)
                self.assertEqual(panel.visible_verdicts[-1][0],
                                 "FALHA FALSA" if label == "OK" else "DEFEITO REAL")
                self.assertIsNone(controller._auto_dispatch_started_generation)
                self.assertEqual(panel.saved, [])
                # Avançar o Qt simulado sem tempo de parede para exercitar
                # o mesmo gate de autorização após o timer.
                controller._stage_timer.stop()
                controller._run_stage_callback()
                self.assertEqual(controller.state, "auto_decision_pending")
                self.assertEqual(panel.saved, [])
                controller._stage_timer.stop()
                controller._run_stage_callback()
                self.assertEqual(panel.saved, [(label, "production_auto")])
                panel.close()

    def test_space_pause_freezes_one_second_without_sending_auto_decision(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = _complete_binary_ng()
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        QTest.qWait(200)
        self.assertTrue(controller.toggle_pause())
        self.assertTrue(controller.paused)
        self.assertFalse(controller._stage_timer.isActive())
        self.assertTrue(callable(controller._stage_callback))
        # Callback atrasado enquanto pausado não deve ser perdido.
        controller._run_stage_callback()
        self.assertTrue(callable(controller._stage_callback))
        QTest.qWait(1100)
        self.assertEqual(panel.saved, [])

        self.assertFalse(controller.toggle_pause())
        self.assertTrue(controller._stage_timer.isActive())
        QTest.qWait(1000)
        self.assertEqual(panel.saved, [("NG", "production_auto")])
        panel.close()

    def test_cancel_or_change_mode_blocks_pending_one_second_command(self):
        for change in ("cancel", "mode", "replace"):
            with self.subTest(change=change):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                analysis = _complete_binary_ng()
                panel.current_analysis = analysis
                self.assertTrue(controller.analysis_ready(analysis))
                generation = controller.generation
                if change == "cancel":
                    controller.cancel_current()
                elif change == "mode":
                    panel.combo_mode.setCurrentText("Modo Teste")
                else:
                    panel.current_analysis = _complete_binary_ng()
                controller._stage_timer.stop()
                controller._run_stage_callback()
                controller._emit_auto_decision(generation, "NG")
                self.assertEqual(panel.saved, [])
                panel.close()

    def test_one_second_replaces_all_post_analysis_scroll_stages(self):
        self.assertEqual(VERDICT_HOLD_MS, 1_000)
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        panel.current_analysis = _complete_binary_ng()
        self.assertTrue(controller.analysis_ready(panel.current_analysis))
        self.assertNotIn(controller.state, {"scrolling", "presentation_top"})
        self.assertFalse(hasattr(controller, "_scroll_animation"))
        self.assertFalse(hasattr(controller, "_go_to_top"))
        self.assertFalse(hasattr(controller, "_start_scroll"))
        self.assertEqual(controller._stage_remaining_ms, VERDICT_HOLD_MS)
        controller.cancel_current()
        panel.close()


class ProductionAutonomySourceContractTests(unittest.TestCase):
    def test_production_controller_uses_qtimer_not_scroll_animation_or_sleep(self):
        source = (
            ROOT / "src" / "ui" / "production_autonomy_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn("VERDICT_HOLD_MS = 1_000", source)
        self.assertIn("QTimer", source)
        self.assertNotIn("QPropertyAnimation", source)
        self.assertNotIn("verticalScrollBar()", source)
        self.assertNotIn("horizontalScrollBar()", source)
        self.assertNotIn("_start_scroll", source)
        self.assertNotIn("time.sleep(", source)

    def test_space_is_window_shortcut_only_for_production_controller(self):
        source = (
            ROOT / "src" / "ui" / "production_autonomy_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn('QKeySequence("Space")', source)
        self.assertIn("Qt.ShortcutContext.WindowShortcut", source)
        self.assertIn("shortcut.setAutoRepeat(False)", source)
        self.assertIn("shortcut.setEnabled(controller.is_production())", source)

    def test_mode_change_restores_daily_metrics_instead_of_resetting_them(self):
        source = (
            ROOT / "src" / "ui" / "production_autonomy_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '"restore_production_daily_session_feedback"',
            source,
        )
        self.assertNotIn(
            '"reset_production_session_feedback"',
            source,
        )

    def test_control_panel_no_longer_saves_immediately_after_analysis(self):
        source = (
            ROOT / "src" / "ui" / "control_panel.py"
        ).read_text(encoding="utf-8")

        self.assertIn("notify_production_analysis_ready", source)
        self.assertIn("notify_production_cycle_started", source)
        self.assertNotIn(
            'self.save_label(auto_decision, source="auto")',
            source,
        )

    def test_main_installs_feedback_before_autonomy_controller(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        feedback = source.index("install_production_session_feedback(panel)")
        controller = source.index(
            "install_production_autonomy_controller(panel)"
        )
        event_loop = source.index("app.exec()")

        self.assertLess(feedback, controller)
        self.assertLess(controller, event_loop)


if __name__ == "__main__":
    unittest.main()
