import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox, QScrollArea, QWidget

from src.ui.production_autonomy_controller import (
    AUTO_DECISION_DELAY_MS,
    HORIZONTAL_SCROLL_DURATION_MS,
    POST_SCROLL_PAUSE_MS,
    RENDER_SETTLE_MS,
    SCROLL_DURATION_MS,
    TOP_HOLD_MS,
    ProductionAutonomyController,
)


ROOT = Path(__file__).resolve().parents[1]


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
        self.interventions = []
        self.intervention_clears = 0

    def update_brain_status(self, message, active=False):
        self.status.append((str(message), bool(active)))

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

    def test_analysis_ready_records_pre_scroll_analysis_time_once(self):
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
        self.assertEqual(panel.analysis_records, [1.37])

        # A mesma análise não pode entrar duas vezes na média da sessão.
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(panel.analysis_records, [1.37])

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

    def test_presentation_is_intentionally_visible_and_non_instant(self):
        self.assertGreaterEqual(RENDER_SETTLE_MS, 300)
        self.assertGreaterEqual(TOP_HOLD_MS, 300)
        self.assertGreaterEqual(SCROLL_DURATION_MS, 6000)
        self.assertGreaterEqual(HORIZONTAL_SCROLL_DURATION_MS, 2000)
        self.assertGreaterEqual(POST_SCROLL_PAUSE_MS, 500)
        self.assertGreaterEqual(AUTO_DECISION_DELAY_MS, 200)


class ProductionAutonomySourceContractTests(unittest.TestCase):
    def test_controller_uses_qproperty_animation_not_sleep(self):
        source = (
            ROOT / "src" / "ui" / "production_autonomy_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn("QPropertyAnimation", source)
        self.assertIn('b"value"', source)
        self.assertIn("verticalScrollBar()", source)
        self.assertIn("_active_specialist_horizontal_bar", source)
        self.assertIn("horizontalScrollBar()", source)
        self.assertIn("adhesive_multilight_analysis_view", source)
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
