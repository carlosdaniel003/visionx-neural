import os
import tempfile
import unittest
from datetime import date

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from src.services.production_daily_session_store import (
    ProductionDailySessionStore,
)
from src.ui.production_session_feedback import (
    ProductionSessionFeedbackOverlay,
)


class _MutableDate:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class ProductionSessionFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clock = _MutableDate(date(2026, 10, 7))
        self.store = ProductionDailySessionStore(
            root_dir=self.temp_dir.name,
            today_provider=self.clock,
        )

        self.panel = QWidget()
        self.panel.resize(1200, 800)
        self.panel.show()
        self.app.processEvents()
        self.overlay = ProductionSessionFeedbackOverlay(
            self.panel,
            daily_store=self.store,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_restore_keeps_card_hidden_until_piece_is_active(self):
        self.overlay.restore_daily_session()

        self.assertFalse(self.overlay.isVisible())
        self.assertEqual(self.overlay.analysis_count, 0)
        self.assertEqual(self.overlay.manual_judgments, 0)
        self.assertEqual(self.overlay.auto_ok, 0)

    def test_metrics_survive_overlay_recreation_same_day(self):
        self.overlay.record_analysis(1.5)
        self.overlay.record_automatic("OK")
        self.overlay.record_manual("NG")

        second_panel = QWidget()
        second_overlay = ProductionSessionFeedbackOverlay(
            second_panel,
            daily_store=ProductionDailySessionStore(
                root_dir=self.temp_dir.name,
                today_provider=self.clock,
            ),
        )

        self.assertEqual(second_overlay.analysis_count, 1)
        self.assertEqual(second_overlay.auto_ok, 1)
        self.assertEqual(second_overlay.manual_judgments, 1)
        self.assertAlmostEqual(second_overlay.accuracy_percent, 50.0)
        self.assertAlmostEqual(second_overlay.average_analysis_time, 1.5)

    def test_new_day_automatically_starts_zero_on_next_piece(self):
        self.overlay.record_analysis(1.0)
        self.overlay.record_automatic("OK")
        self.assertEqual(self.overlay.auto_ok, 1)

        self.clock.value = date(2026, 10, 8)
        self.overlay.show_active()

        self.assertEqual(self.overlay.auto_ok, 0)
        self.assertEqual(self.overlay.manual_judgments, 0)
        self.assertEqual(self.overlay.analysis_count, 0)
        self.assertIn("08/10/2026", self.overlay.header_label.text())

    def test_precision_matches_user_contract_98_of_100(self):
        self.overlay.auto_ok = 98
        self.overlay.manual_judgments = 2
        self.overlay._persist()
        self.overlay._refresh()

        self.assertEqual(self.overlay.completed_judgments, 100)
        self.assertAlmostEqual(self.overlay.accuracy_percent, 98.0)
        self.assertIn("98.0%", self.overlay.accuracy_label.text())

    def test_manual_judgment_reduces_precision(self):
        self.overlay.record_automatic("OK")
        self.assertAlmostEqual(self.overlay.accuracy_percent, 100.0)

        self.overlay.record_manual("NG")

        self.assertAlmostEqual(self.overlay.accuracy_percent, 50.0)
        self.assertEqual(self.overlay.manual_judgments, 1)

    def test_average_uses_only_registered_analysis_times(self):
        self.overlay.record_analysis(1.0)
        self.overlay.record_analysis(2.0)
        self.overlay.record_analysis(3.0)

        self.assertEqual(self.overlay.analysis_count, 3)
        self.assertEqual(self.overlay.analysis_time_count, 3)
        self.assertAlmostEqual(
            self.overlay.average_analysis_time,
            2.0,
        )
        self.assertIn("2.00 s", self.overlay.time_label.text())

    def test_invalid_zero_time_does_not_distort_average(self):
        self.overlay.record_analysis(0.0)
        self.overlay.record_analysis(2.5)

        self.assertEqual(self.overlay.analysis_count, 2)
        self.assertEqual(self.overlay.analysis_time_count, 1)
        self.assertAlmostEqual(
            self.overlay.average_analysis_time,
            2.5,
        )

    def test_pause_updates_fixed_message(self):
        self.overlay.set_paused(True)

        self.assertTrue(self.overlay.paused)
        self.assertIn("PAUSADA", self.overlay.state_label.text())
        self.assertIn("ESPAÇO", self.overlay.state_label.text())

        self.overlay.set_paused(False)

        self.assertIn("ATIVA", self.overlay.state_label.text())

    def test_dynamic_tooltip_explains_precision_and_time_contract(self):
        self.overlay.record_analysis(1.25)
        self.overlay.record_automatic("OK")

        tooltip = self.overlay.toolTip()
        self.assertIn("Precisão =", tooltip)
        self.assertIn("julgamentos automáticos", tooltip)
        self.assertIn("Tempo médio de análise", tooltip)
        self.assertIn("Scroll, pausas", tooltip)
        self.assertIn("permanecem durante todo o mesmo dia", tooltip)
        self.assertIn("07/10/2026", tooltip)


if __name__ == "__main__":
    unittest.main()
