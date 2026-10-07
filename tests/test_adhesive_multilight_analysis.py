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
            self.assertTrue(lane.scroll.isHidden())

    def test_side_top_mid_can_receive_independent_analyses(self):
        view = AdhesiveMultiLightAnalysisView()

        def analysis(score):
            return {
                "active_engines": ["ssim_expert.py"],
                "detail": {
                    "heat_map_raw": None,
                    "local_score": score,
                    "ctx_score": score,
                },
            }

        self.assertTrue(view.set_analysis("SIDE", analysis(0.1)))
        self.assertTrue(view.set_analysis("TOP", analysis(0.2)))
        self.assertTrue(view.set_analysis("MID", analysis(0.3)))

        for mode in ("SIDE", "TOP", "MID"):
            lane = view.lanes[mode]
            self.assertFalse(lane.scroll.isHidden())
            self.assertIn(
                f"Análise {mode} disponível",
                lane.status_label.text(),
            )

    def test_adhesive_has_one_master_horizontal_scroll(self):
        view = AdhesiveMultiLightAnalysisView()
        view.resize(720, 1100)
        view.show()
        self.app.processEvents()

        analysis = {
            "active_engines": [
                "ssim_expert.py",
                "silk_expert.py",
                "semantic_expert.py",
                "shift_expert.py",
            ],
            "detail": {},
        }
        for mode in ("SIDE", "TOP", "MID"):
            self.assertTrue(view.set_analysis(mode, analysis))

        self.app.processEvents()
        view._sync_master_scroll_range()

        self.assertEqual(
            view.horizontal_scroll.objectName(),
            "adhesiveExpertHorizontalScroll",
        )
        self.assertGreater(view.horizontal_scroll.maximum(), 0)
        self.assertTrue(view.horizontal_scroll.isEnabled())
        self.assertEqual(view.horizontal_scroll_bars(), [view.horizontal_scroll])

        # As barras internas existem para o QScrollArea, mas ficam ocultas:
        # o operador usa uma única barra mestre para as três iluminações.
        for lane in view.lanes.values():
            self.assertTrue(
                lane.scroll.horizontalScrollBar().isHidden()
                or not lane.scroll.horizontalScrollBar().isVisible()
            )

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

    def test_current_stage_populates_side_and_auxiliary_lighting_analyses(self):
        source = open(
            "src/ui/adhesive_multilight_inspection.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            'analysis_view.set_analysis(\n'
            '                "SIDE",',
            source,
        )
        self.assertIn(
            "lighting_analysis = analyze_lighting(",
            source,
        )
        self.assertIn(
            "analysis_view.set_analysis(\n"
            "                        aux_mode,",
            source,
        )
        self.assertIn(
            'self.adhesive_multilight_analyses["SIDE"]',
            source,
        )
        self.assertIn(
            "self.adhesive_multilight_analyses[aux_mode]",
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
