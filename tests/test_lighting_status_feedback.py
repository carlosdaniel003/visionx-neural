import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from src.ui.lighting_status_feedback import (
    LIGHTING_STATUS_FADE_IN_MS,
    LIGHTING_STATUS_FADE_OUT_MS,
    LIGHTING_STATUS_MARGIN,
    LIGHTING_STATUS_SLIDE_PX,
    LIGHTING_STATUS_TOP_OFFSET,
    LIGHTING_STATUS_WIDTH,
    install_lighting_status_feedback,
    install_lighting_status_feedback_hooks,
)


class LightingStatusOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self):
        panel = QWidget()
        panel.resize(1000, 700)
        panel.lbl_light_value = QLabel("SIDE", panel)
        install_lighting_status_feedback(panel)
        return panel

    def test_card_starts_hidden_until_analysis_finishes(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback

        self.assertEqual(overlay.current_mode, "SIDE")
        self.assertEqual(overlay.value_label.text(), "SIDE")
        self.assertEqual(overlay.key_label.text(), "↓")
        self.assertTrue(overlay.isHidden())
        self.assertTrue(
            overlay.testAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
        )
        self.assertEqual(overlay.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertFalse(hasattr(overlay, "_hide_timer"))

    def test_card_maps_top_side_mid_without_becoming_visible_early(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback

        self.assertTrue(overlay.set_lighting("TOP"))
        self.assertEqual(overlay.key_label.text(), "←")
        self.assertTrue(overlay.isHidden())

        self.assertTrue(overlay.set_lighting("SIDE"))
        self.assertEqual(overlay.key_label.text(), "↓")
        self.assertTrue(overlay.isHidden())

        self.assertTrue(overlay.set_lighting("MID"))
        self.assertEqual(overlay.key_label.text(), "→")
        self.assertTrue(overlay.isHidden())

    def test_final_analysis_shows_lighting_with_verdict_timing(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback
        overlay.set_lighting("TOP")

        shown = panel.show_lighting_status_feedback(
            {"verdict": "FALHA FALSA", "is_defect": False}
        )

        self.assertTrue(shown)
        self.assertFalse(overlay.isHidden())
        self.assertEqual(overlay.value_label.text(), "TOP")
        self.assertEqual(overlay.key_label.text(), "←")
        self.assertEqual(overlay._fade_in.duration(), LIGHTING_STATUS_FADE_IN_MS)
        self.assertEqual(overlay._fade_out.duration(), LIGHTING_STATUS_FADE_OUT_MS)
        self.assertEqual(
            overlay._slide_in.startValue().x(),
            overlay._slide_in.endValue().x() + LIGHTING_STATUS_SLIDE_PX,
        )

    def test_decision_keeps_lighting_visible_until_synchronized_fade(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback
        panel.show_lighting_status_feedback(
            {"verdict": "DEFEITO REAL", "is_defect": True}
        )

        self.assertTrue(panel.prepare_lighting_status_feedback_dismissal())
        self.assertTrue(overlay._decision_dismiss_pending)
        self.assertFalse(panel.clear_lighting_status_feedback())
        self.assertFalse(overlay.isHidden())

        self.assertTrue(panel.start_lighting_status_feedback_fade_out())
        self.assertEqual(overlay._fade_out.endValue(), 0.0)

        overlay._finish_hide()
        self.assertTrue(overlay.isHidden())
        self.assertFalse(overlay._decision_dismiss_pending)

    def test_card_is_anchored_below_verdict_area(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback
        target = overlay._top_right_position()

        self.assertEqual(
            target.x(),
            panel.width() - LIGHTING_STATUS_WIDTH - LIGHTING_STATUS_MARGIN,
        )
        self.assertEqual(target.y(), LIGHTING_STATUS_TOP_OFFSET)


class LightingStatusAnalysisLifecycleHookTests(unittest.TestCase):
    def test_reference_update_shows_and_reset_clears_lighting_card(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return True

            def _update_reference_panel(self, analysis):
                self.events.append(("reference", analysis["verdict"]))
                return "updated"

            def _reset_confidence_panel(self):
                self.events.append(("reset", None))
                return "reset"

            def show_lighting_status_feedback(self, analysis):
                self.events.append(("show_light", analysis["verdict"]))
                return True

            def clear_lighting_status_feedback(self):
                self.events.append(("clear_light", None))
                return True

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel._update_reference_panel(
            {"verdict": "FALHA FALSA", "is_defect": False}
        )
        reset = panel._reset_confidence_panel()

        self.assertEqual(result, "updated")
        self.assertEqual(reset, "reset")
        self.assertEqual(
            panel.events,
            [
                ("reference", "FALHA FALSA"),
                ("show_light", "FALHA FALSA"),
                ("reset", None),
                ("clear_light", None),
            ],
        )


class LightingStatusHookTests(unittest.TestCase):
    def test_odin_keyboard_updates_status_and_shows_actual_xp_key(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return True

            def update_lighting_status_feedback(self, mode):
                self.events.append(("status", mode))
                return True

            def show_operational_key_feedback(self, key, source=""):
                self.events.append(("key", key, source))
                return True

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel.change_lighting("TOP", "odin_keyboard")

        self.assertTrue(result)
        self.assertEqual(
            panel.events,
            [
                ("change", "TOP", "odin_keyboard"),
                ("status", "TOP"),
                ("key", "LEFT", "odin_keyboard"),
            ],
        )


    def test_xp_network_lighting_updates_status_and_shows_physical_key(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return True

            def update_lighting_status_feedback(self, mode):
                self.events.append(("status", mode))
                return True

            def show_operational_key_feedback(self, key, source=""):
                self.events.append(("key", key, source))
                return True

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel.change_lighting("MID", "network")

        self.assertTrue(result)
        self.assertEqual(
            panel.events,
            [
                ("change", "MID", "network"),
                ("status", "MID"),
                ("key", "RIGHT", "xp_keyboard"),
            ],
        )

    def test_failed_xp_command_does_not_change_visual_state(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return False

            def update_lighting_status_feedback(self, mode):
                self.events.append(("status", mode))

            def show_operational_key_feedback(self, key, source=""):
                self.events.append(("key", key, source))

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel.change_lighting("MID", "odin_control")

        self.assertFalse(result)
        self.assertEqual(
            panel.events,
            [("change", "MID", "odin_control")],
        )




class LightingControlSourceContractTests(unittest.TestCase):
    def test_control_panel_and_shortcuts_use_real_aoi_mapping(self):
        panel_source = open("src/ui/control_panel.py", encoding="utf-8").read()
        shortcut_source = open(
            "src/ui/lighting_shortcuts.py",
            encoding="utf-8",
        ).read()

        self.assertIn('("Left", "TOP", "btn_light_top")', shortcut_source)
        self.assertIn('("Down", "SIDE", "btn_light_side")', shortcut_source)
        self.assertIn('("Right", "MID", "btn_light_mid")', shortcut_source)
        self.assertIn('"TOP": "LEFT"', panel_source)
        self.assertIn('"SIDE": "DOWN"', panel_source)
        self.assertIn('"MID": "RIGHT"', panel_source)

    def test_ui_starts_in_side_and_displays_correct_arrows(self):
        ui_source = open("src/ui/control_panel_ui.py", encoding="utf-8").read()
        controls_source = open(
            "src/ui/operational_controls.py",
            encoding="utf-8",
        ).read()

        self.assertIn('window.lbl_light_value = QLabel("SIDE")', ui_source)
        self.assertIn('QPushButton("Luz TOP • ←")', ui_source)
        self.assertIn('QPushButton("Luz SIDE • ↓")', ui_source)
        self.assertIn('QPushButton("Luz MID • →")', ui_source)
        self.assertIn('"Luz TOP  |  ←"', controls_source)
        self.assertIn('"Luz SIDE  |  ↓"', controls_source)
        self.assertIn('"Luz MID  |  →"', controls_source)

    def test_xp_agent_reports_physical_arrows_back_to_odin(self):
        source = open("agente_industrial_xp.py", encoding="utf-8").read()

        self.assertIn('args=("CMD_TOP",)', source)
        self.assertIn('args=("CMD_SIDE",)', source)
        self.assertIn('args=("CMD_MID",)', source)

    def test_main_installs_lighting_hook_and_persistent_overlay(self):
        source = open("main.py", encoding="utf-8").read()
        hook = source.index("install_lighting_status_feedback_hooks(ControlPanel)")
        panel = source.index("panel = ControlPanel()")
        overlay = source.index("install_lighting_status_feedback(panel)")

        self.assertLess(hook, panel)
        self.assertLess(panel, overlay)

if __name__ == "__main__":
    unittest.main()
