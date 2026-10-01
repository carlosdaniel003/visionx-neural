import unittest
from pathlib import Path

from src.ui.branding import (
    CALIBRATION_WINDOW_TITLE,
    DECISION_DEBUG_TITLE,
    DISPLAY_NAME,
    HUD_INITIAL_TEXT,
    MONITOR_WINDOW_TITLE,
    XP_DEBUG_TITLE,
)


class VisualBrandingTests(unittest.TestCase):
    def test_canonical_display_name_is_odin(self):
        self.assertEqual(
            DISPLAY_NAME,
            "ODIN - Observador Digital Inteligente",
        )
        self.assertEqual(
            MONITOR_WINDOW_TITLE,
            "ODIN - Observador Digital Inteligente - Monitoramento IA",
        )
        self.assertEqual(
            CALIBRATION_WINDOW_TITLE,
            (
                "ODIN - Observador Digital Inteligente - "
                "Calibrar Zona de Interesse Avançado"
            ),
        )
        self.assertEqual(
            HUD_INITIAL_TEXT,
            "ODIN - Observador Digital Inteligente: Inicializando...",
        )
        self.assertEqual(
            XP_DEBUG_TITLE,
            (
                "ODIN - Observador Digital Inteligente - "
                "DEBUG DE ENTRADA WINDOWS XP"
            ),
        )
        self.assertEqual(
            DECISION_DEBUG_TITLE,
            "DECISÃO ODIN - Observador Digital Inteligente",
        )

    def test_main_ui_uses_centralized_branding_and_responsive_title(self):
        source = Path("src/ui/control_panel_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("window.setWindowTitle(MONITOR_WINDOW_TITLE)", source)
        self.assertIn("title = QLabel(DISPLAY_NAME)", source)
        self.assertIn("title.setWordWrap(True)", source)
        self.assertNotIn('"VisionX Neural - Monitoramento IA"', source)
        self.assertNotIn('QLabel("VisionX Neural")', source)

    def test_auxiliary_visual_surfaces_use_odin_branding(self):
        hud = Path("src/ui/hud_window.py").read_text(encoding="utf-8")
        calibration = Path("src/ui/calibration_window.py").read_text(
            encoding="utf-8"
        )
        debug = Path("src/ui/network_xp_debug.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("self.log_text = HUD_INITIAL_TEXT", hud)
        self.assertIn(
            "self.setWindowTitle(CALIBRATION_WINDOW_TITLE)",
            calibration,
        )
        self.assertIn("XP_DEBUG_TITLE", debug)
        self.assertIn("DECISION_DEBUG_TITLE", debug)

        self.assertNotIn("VisionX Neural: Inicializando...", hud)
        self.assertNotIn(
            "VisionX Neural - Calibrar Zona de Interesse Avançado",
            calibration,
        )
        self.assertNotIn(
            "VISIONX - DEBUG DE ENTRADA WINDOWS XP",
            debug,
        )
        self.assertNotIn("DECISÃO VISIONX", debug)


if __name__ == "__main__":
    unittest.main()
