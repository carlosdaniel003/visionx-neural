"""Value-only OCR contract, Qt responsiveness and Windows cp1252 bootstrap."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QCoreApplication, QTimer

from src.services import screen_monitor


ROOT = Path(__file__).resolve().parents[1]


class StartupConsoleTests(unittest.TestCase):
    def test_main_bootstrap_keeps_cp1252_console_from_crashing(self):
        # Executa SOMENTE o prefixo de main.py; nao abre nenhuma janela.
        script = (ROOT / "main.py").read_text(encoding="utf-8")
        prefix = script.split("from PyQt6.QtWidgets import QApplication", 1)[0]
        self.assertIn("reconfigure", prefix)
        output = io.BytesIO()
        stream = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
        with patch.object(sys, "stdout", stream):
            exec(compile(prefix, str(ROOT / "main.py"), "exec"), {})
            print("\U0001f441\ufe0f OCR inicializado")
            stream.flush()
        self.assertIn(b"OCR inicializado", output.getvalue())
        # O Unicode impossivel vira escape, sem UnicodeEncodeError.
        self.assertIn(b"\\U0001f441", output.getvalue())


class ValueOnlyUiTests(unittest.TestCase):
    def test_board_and_parts_cards_and_bindings_removed(self):
        ui = (ROOT / "src/ui/control_panel_ui.py").read_text("utf-8")
        builder = ui.split("    def _build_aoi_info(", 1)[1].split(
            "    def _create_image_card(", 1
        )[0]
        self.assertNotIn("Placa / Máquina", builder)
        self.assertNotIn('self._create_info_card("Componente"', builder)
        self.assertNotIn("lbl_board_value", builder)
        self.assertNotIn("lbl_parts_value", builder)
        self.assertEqual(builder.count("self._create_info_card("), 3)

        controller = (ROOT / "src/ui/control_panel.py").read_text("utf-8")
        self.assertNotIn("lbl_board_value", controller)
        self.assertNotIn("lbl_parts_value", controller)


class OcrQtResponsivenessTests(unittest.TestCase):
    def test_event_loop_continues_processing_during_tesseract(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        self.assertIsNotNone(app)
        ticks = []
        timer = QTimer()
        timer.setInterval(12)
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start()

        class SlowTesseract:
            def __init__(self):
                self.thread = None

            def image_to_string(self, image, config=""):
                self.thread = threading.current_thread()
                time.sleep(.16)
                return "Value 0 <= 5 <= 10 FALTANDO"

        fake = SlowTesseract()
        monitor = screen_monitor.ScreenMonitor()
        try:
            with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
                screen_monitor, "pytesseract", fake, create=True
            ):
                result = monitor._ocr_fast(np.zeros((30, 80, 3), dtype=np.uint8))
        finally:
            timer.stop()
        self.assertIn("Value", result)
        self.assertIsNotNone(fake.thread)
        self.assertIsNot(fake.thread, threading.current_thread())
        self.assertGreaterEqual(len(ticks), 2)


if __name__ == "__main__":
    unittest.main()
