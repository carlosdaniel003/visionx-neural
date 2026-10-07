import unittest
from pathlib import Path

from src.ui.responsive_layout import profile_for_width


ROOT = Path(__file__).resolve().parents[1]


class LowerPanelsResponsiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui_source = (
            ROOT / "src" / "ui" / "control_panel_ui.py"
        ).read_text(encoding="utf-8")

    def test_decision_confidence_reflows_by_profile(self):
        self.assertIn(
            "self._reflow_grid(self.footer_grid, self.footer_cards, profile.footer_columns)",
            self.ui_source,
        )
        self.assertEqual(profile_for_width(1366).footer_columns, 1)
        self.assertEqual(profile_for_width(1600).footer_columns, 2)
        self.assertEqual(profile_for_width(1920).footer_columns, 3)

    def test_operational_groups_stack_on_notebook_and_split_on_large_monitor(self):
        self.assertIn("self.controls_body_grid", self.ui_source)
        self.assertIn("profile.controls_columns", self.ui_source)
        self.assertEqual(profile_for_width(1366).controls_columns, 1)
        self.assertEqual(profile_for_width(1600).controls_columns, 2)
        self.assertEqual(profile_for_width(1920).controls_columns, 2)

    def test_notebook_keeps_three_lighting_buttons_and_two_by_two_actions(self):
        compact = profile_for_width(1366)
        self.assertEqual(compact.light_columns, 3)
        self.assertEqual(compact.action_columns, 2)

    def test_network_debug_actions_stack_only_in_compact_profile(self):
        self.assertIn("self.network_debug_action_buttons", self.ui_source)
        self.assertIn("profile.debug_action_columns", self.ui_source)
        self.assertEqual(profile_for_width(1366).debug_action_columns, 1)
        self.assertEqual(profile_for_width(1920).debug_action_columns, 2)

    def test_reflow_clears_stale_column_stretches(self):
        self.assertIn(
            "previous_columns = max(grid.columnCount(), columns)",
            self.ui_source,
        )
        self.assertIn(
            "grid.setColumnStretch(column, 0)",
            self.ui_source,
        )

    def test_three_bottom_status_items_stack_on_notebook(self):
        self.assertIn("if compact:", self.ui_source)
        self.assertIn(
            "self.status_layout.addWidget(item, row, 0)",
            self.ui_source,
        )
        self.assertIn(
            "self.status_layout.addWidget(item, 0, column)",
            self.ui_source,
        )


if __name__ == "__main__":
    unittest.main()
