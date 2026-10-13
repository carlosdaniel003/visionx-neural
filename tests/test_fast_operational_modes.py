"""Regressoes da captura rapida XP nos tres modos sem afetar julgamento."""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

from src.services.screen_monitor import ScreenMonitor
from src.ui.control_panel import ControlPanel, uses_fast_xp_capture


class OperationalFastCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_all_three_modes_enable_capture_but_invalid_mode_disables_it(self):
        for mode in ("Modo Sombra", "Modo Teste", "Modo Produção"):
            self.assertTrue(uses_fast_xp_capture(mode), mode)
        for mode in ("", "Outro", None):
            self.assertFalse(uses_fast_xp_capture(mode), mode)

    def test_first_xp_packet_enables_fast_capture_in_test_and_production(self):
        class Combo:
            def __init__(self, mode):
                self.mode = mode
            def currentText(self):
                return self.mode

        class FakePanel:
            last_xp_ip = None
            def _send_xp_shadow_control(self, enabled):
                self.enabled = enabled

        for mode in ("Modo Teste", "Modo Produção"):
            panel = FakePanel()
            panel.combo_mode = Combo(mode)
            ControlPanel.handle_xp_peer_discovered(panel, "169.254.95.200")
            self.assertTrue(panel.enabled, mode)
            self.assertEqual(panel.last_xp_ip, "169.254.95.200")

    def test_auxiliary_ocr_reused_in_non_shadow_mode(self):
        monitor = ScreenMonitor()
        self.assertFalse(monitor._shadow_fast)
        frame = np.zeros((70, 90, 3), np.uint8)
        metadata = {
            "board": "PLACA", "parts": "U2~5",
            "value": "FALTANDO", "category": "FALTANDO",
        }
        monitor._shadow_auxiliary_info = dict(metadata)
        with patch.object(monitor, "_ocr_fast",
                          side_effect=AssertionError("OCR repetido em TOP/MID")):
            result = monitor._extract_text_info(
                frame, (0, 0, 30, 8), (35, 0, 30, 8))
        self.assertEqual(result, metadata)
        self.assertIsNot(result, monitor._shadow_auxiliary_info)
        monitor._shadow_auxiliary_info = None
        with patch("src.services.screen_monitor.HAS_TESSERACT", False):
            self.assertIn("raw_text", monitor._extract_text_info(
                frame, (0, 0, 30, 8), (35, 0, 30, 8)))

    def test_png_debug_remains_active_outside_shadow(self):
        monitor = ScreenMonitor()
        image = np.zeros((12, 12, 3), np.uint8)
        with patch("src.services.screen_monitor.DEBUG_DIR") as folder, \
             patch("src.services.screen_monitor.cv2.imwrite") as writer:
            monitor._write_debug_crop(Path("debug.png"), image)
            folder.mkdir.assert_called_once()
            writer.assert_called_once()
        monitor._shadow_fast = True
        with patch("src.services.screen_monitor.cv2.imwrite") as writer:
            monitor._write_debug_crop(Path("debug.png"), image)
            writer.assert_not_called()

    def test_geometry_not_skipped_in_test_and_production(self):
        from tests.test_shadow_fast_capture import ShadowFake
        frame = np.zeros((48, 64, 3), np.uint8)
        for mode in ("Modo Teste", "Modo Produção"):
            panel = ShadowFake(mode)
            panel.capture_start_time = time.perf_counter()
            with patch.object(panel, "numpy_to_pixmap", return_value=None), \
                 patch("src.ui.control_panel.normalize_aoi_text",
                       return_value=("FALTANDO", "FALTANDO")), \
                 patch("src.ui.control_panel.detect_anomalies",
                       side_effect=RuntimeError("geometria preservada")):
                with self.assertRaisesRegex(RuntimeError, "geometria preservada"):
                    ControlPanel.process_aoi_images(
                        panel, frame, frame, {"value": "FALTANDO"})

    def test_auxiliary_light_requires_settle_and_two_stable_frames(self):
        from src.services.network_receiver import NetworkReceiver

        receiver = NetworkReceiver()
        frame = np.zeros((64, 64, 3), np.uint8)
        self.assertFalse(receiver.arm_auxiliary_capture("TOP"))
        receiver.set_auxiliary_image_mode(True)
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=100.0):
            self.assertTrue(receiver.arm_auxiliary_capture("TOP"))
        self.assertFalse(receiver._auxiliary_frame_ready(frame, at=100.20))
        self.assertFalse(receiver._auxiliary_frame_ready(frame, at=100.36))
        self.assertTrue(receiver._auxiliary_frame_ready(frame, at=100.60))
        self.assertFalse(receiver._auxiliary_frame_ready(frame, at=100.80))
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=101.0):
            self.assertTrue(receiver.arm_auxiliary_capture("MID"))
        self.assertFalse(receiver._auxiliary_frame_ready(frame, at=101.36))
        self.assertTrue(receiver._auxiliary_frame_ready(frame, at=101.60))
        receiver.set_auxiliary_image_mode(False)
        self.assertFalse(receiver._auxiliary_frame_ready(frame, at=110.0))

    def test_automation_arms_receiver_before_requesting_light(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/ui/adhesive_multilight_automation.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("arm_capture(normalized)"),
            source.index("if not self._send_mode(normalized):"),
        )

    def test_production_policy_remains_separate_from_fast_capture(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "src/ui/control_panel.py").read_text(encoding="utf-8")
        automation = (root / "src/ui/adhesive_multilight_automation.py").read_text(encoding="utf-8")
        self.assertIn('if current_mode == "Modo Produção":', source)
        self.assertIn("notify_production_analysis_ready", source)
        self.assertIn('if mode == "Modo Produção":', automation)


if __name__ == "__main__":
    unittest.main()
