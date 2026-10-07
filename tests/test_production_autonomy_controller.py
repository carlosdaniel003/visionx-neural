import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox, QScrollArea, QWidget

from src.ui.production_autonomy_controller import (
    AUTO_DECISION_DELAY_MS,
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
        self.status = []
        self.saved = []
        self.feedback = []
        self.session_resets = 0
        self.session_increments = []
        self.interventions = []
        self.intervention_clears = 0

    def update_brain_status(self, message, active=False):
        self.status.append((str(message), bool(active)))

    def save_label(self, decision, source="button"):
        self.saved.append((str(decision), str(source)))
        self.current_analysis = None

    def show_decision_key_feedback(self, decision, source=""):
        self.feedback.append((str(decision), str(source)))
        return True

    def reset_production_session_feedback(self):
        self.session_resets += 1

    def show_production_session_feedback(self):
        return None

    def increment_production_session_feedback(self, decision):
        self.session_increments.append(str(decision))

    def show_production_intervention_feedback(self, reason):
        self.interventions.append(str(reason))

    def clear_production_intervention_feedback(self):
        self.intervention_clears += 1


class ProductionAutonomyControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_entering_production_resets_session(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)

        panel.combo_mode.setCurrentText("Modo Produção")
        self.app.processEvents()

        self.assertTrue(controller.is_production())
        self.assertEqual(panel.session_resets, 1)

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
        self.assertEqual(panel.saved, [])
        self.assertEqual(panel.feedback, [])

    def test_false_failure_emits_only_auto_ok(self):
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
        generation = controller.generation

        controller._emit_auto_ok(generation)

        self.assertEqual(
            panel.saved,
            [("OK", "production_auto")],
        )
        self.assertEqual(panel.session_increments, ["OK"])
        self.assertEqual(controller.state, "idle")

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

    def test_operator_decision_rearms_without_incrementing_auto_counter(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        controller.state = "operator_review"
        controller.pending_analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
        }

        controller.operator_decision_completed("NG", source="xp_keyboard")

        self.assertEqual(controller.state, "idle")
        self.assertIsNone(controller.pending_analysis)
        self.assertEqual(panel.session_increments, [])
        self.assertGreaterEqual(panel.intervention_clears, 1)

    def test_presentation_is_intentionally_visible_and_non_instant(self):
        self.assertGreaterEqual(RENDER_SETTLE_MS, 300)
        self.assertGreaterEqual(TOP_HOLD_MS, 300)
        self.assertGreaterEqual(SCROLL_DURATION_MS, 4000)
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
        self.assertNotIn("time.sleep(", source)

    def test_control_panel_no_longer_saves_immediately_after_analysis(self):
        source = (
            ROOT / "src" / "ui" / "control_panel.py"
        ).read_text(encoding="utf-8")

        self.assertIn("notify_production_analysis_ready", source)
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
