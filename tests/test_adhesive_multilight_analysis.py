import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.ui.adhesive_multilight_analysis import (
    AdhesiveMultiLightAnalysisView,
)


class AdhesiveMultiLightAnalysisViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_notebook_stacks_lighting_analyses(self):
        self.assertEqual(
            AdhesiveMultiLightAnalysisView.columns_for_width(1366),
            1,
        )
        self.assertEqual(
            AdhesiveMultiLightAnalysisView.columns_for_width(1499),
            1,
        )

    def test_large_monitor_uses_three_lighting_columns(self):
        self.assertEqual(
            AdhesiveMultiLightAnalysisView.columns_for_width(1500),
            3,
        )
        self.assertEqual(
            AdhesiveMultiLightAnalysisView.columns_for_width(1920),
            3,
        )

    def test_side_top_mid_lanes_are_always_present(self):
        view = AdhesiveMultiLightAnalysisView()

        self.assertEqual(set(view.lanes), {"SIDE", "TOP", "MID"})
        for mode, lane in view.lanes.items():
            self.assertIn(
                f"Aguardando análise da iluminação {mode}",
                lane.status_label.text(),
            )
            self.assertFalse(lane.scroll.isVisible())

    def test_only_side_can_be_filled_without_touching_top_mid(self):
        view = AdhesiveMultiLightAnalysisView()
        analysis = {
            "active_engines": ["ssim_expert.py"],
            "detail": {
                "heat_map_raw": None,
                "local_score": 0.0,
                "ctx_score": 0.0,
            },
        }

        self.assertTrue(view.set_analysis("SIDE", analysis))

        side = view.lanes["SIDE"]
        top = view.lanes["TOP"]
        mid = view.lanes["MID"]

        self.assertTrue(side.scroll.isVisible())
        self.assertIn("Análise SIDE disponível", side.status_label.text())
        self.assertFalse(top.scroll.isVisible())
        self.assertFalse(mid.scroll.isVisible())
        self.assertIn("Aguardando análise", top.status_label.text())
        self.assertIn("Aguardando análise", mid.status_label.text())

    def test_each_lane_has_same_specialist_structure(self):
        view = AdhesiveMultiLightAnalysisView()

        expected = {
            "ssim_expert.py",
            "silk_expert.py",
            "semantic_expert.py",
            "shift_expert.py",
        }
        for lane in view.lanes.values():
            self.assertEqual(set(lane.frames), expected)
            self.assertIsNotNone(lane.radar)


class AdhesiveSpecialistSourceContractTests(unittest.TestCase):
    def test_ui_has_separate_normal_and_adhesive_telemetry_pages(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn("window.normal_telemetry_view", source)
        self.assertIn(
            "window.adhesive_multilight_analysis_view",
            source,
        )
        self.assertIn("window.telemetry_view_stack", source)

    def test_adhesive_mode_switches_images_and_specialists_together(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "window.adhesive_multilight_view",
            source,
        )
        self.assertIn(
            "window.adhesive_multilight_analysis_view",
            source,
        )
        self.assertIn(
            "telemetry_stack.setCurrentWidget(telemetry_target)",
            source,
        )

    def test_current_stage_populates_side_analysis_only(self):
        source = open(
            "src/ui/adhesive_multilight_inspection.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            'analysis_view.set_analysis(\n'
            '                "SIDE",',
            source,
        )
        self.assertNotIn(
            'analysis_view.set_analysis("TOP"',
            source,
        )
        self.assertNotIn(
            'analysis_view.set_analysis("MID"',
            source,
        )

    def test_adhesive_mode_uses_vertical_full_width_stage(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "if adhesive_mode or profile.splitter_vertical",
            source,
        )
        self.assertIn(
            "window.images_section.setMaximumWidth(16777215)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
