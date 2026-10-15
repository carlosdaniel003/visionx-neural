"""Integração OCR somente Categoria, sem Board/Parts/Value."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.services import screen_monitor


_RAW = (
    "Board P22-22200 (PRINCIPAL) Li3\n"
    "Parts R375 Block 1\n"
    "Value 12 <= 0 <= 60 FALTANDO"
)


class _Tesseract:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def image_to_string(self, _image, config=""):
        self.calls.append(config)
        return self.responses.pop(0)


class ScreenMonitorOCRTimingTests(unittest.TestCase):
    def setUp(self):
        self.monitor = screen_monitor.ScreenMonitor()
        self.image = np.zeros((840, 1165, 3), dtype=np.uint8)
        self.blue = (200, 400, 220, 15)
        self.red = (700, 400, 200, 15)

    def read(self, fake):
        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", fake, create=True
        ):
            return self.monitor._extract_text_info(
                self.image, self.blue, self.red
            )

    def test_value_only_ignores_board_and_parts_without_extra_tesseract(self):
        fake = _Tesseract([_RAW])
        data = self.read(fake)
        self.assertEqual(data["board"], "")
        self.assertEqual(data["parts"], "")
        self.assertEqual(data["value"], "")
        self.assertEqual(data["category"], "FALTANDO")
        profile = self.monitor.last_ocr_profile
        self.assertEqual(profile["route"], "general_blue")
        self.assertEqual(profile["parts_reason"], "parts_disabled")
        self.assertEqual(profile["call_count"], 1)
        self.assertEqual(profile["calls"][0]["phase"], "general_blue")
        self.assertFalse(profile["general_retry"])
        self.assertNotIn("R375", str(profile))

    def test_corrupt_parts_never_triggers_directed_ocr(self):
        # Antes, FICE~S iniciava image_to_data + duas image_to_string.
        # Agora só a categoria e os limites do Value são utilizados.
        raw = (
            "Board P22-22200 (PRINCIPAL) Li3\n"
            "Parts FICE~S Block 7 Step 1 Process 16\n"
            "Value 0 <= 48.866 <= 10 FALTANDO"
        )
        fake = _Tesseract([raw])
        result = self.read(fake)
        self.assertEqual(result["board"], "")
        self.assertEqual(result["parts"], "")
        self.assertEqual(result["value"], "")
        self.assertEqual(result["category"], "FALTANDO")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(self.monitor.last_ocr_profile["call_count"], 1)
        self.assertEqual(self.monitor.last_ocr_profile["parts_reason"], "parts_disabled")

    def test_valid_historical_e1_pattern_is_not_parsed(self):
        raw = (
            "Board P22-22200 (PRINCIPAL) Li3\n"
            "Parts EL~~~9 Block 2 Step 3 Process 10\n"
            "Value 15 <= 0 <= 70 FALTANDO"
        )
        result = self.read(_Tesseract([raw]))
        self.assertEqual(result["parts"], "")
        self.assertEqual(result["board"], "")
        self.assertEqual(result["value"], "")
        self.assertEqual(result["category"], "FALTANDO")
        self.assertEqual(self.monitor.last_ocr_profile["call_count"], 1)

    def test_category_must_come_from_value_line_not_parts_text(self):
        # Evita que tokens isolados em nome de componente/placa sejam
        # confundidos com a categoria e desviem a inferência.
        fake = _Tesseract([
            "Board MISSING TEST\nParts FALTANDO\nValue 2 <= 3 <= 8 INVERTIDO"
        ])
        result = self.read(fake)
        self.assertEqual(result["category"], "INVERTIDO")
        self.assertEqual(result["value"], "")
        self.assertEqual(result["raw_text"], "")

    def test_unsupported_value_category_returns_unknown_fail_closed(self):
        fake = _Tesseract([
            "Board P24-03804 L05\nParts R37\n"
            "Value 40 <= 25.602 <= 98 Little Solder"
        ])
        result = self.read(fake)
        self.assertEqual(result["category"], "Unknown")
        self.assertEqual(result["value"], "")
        self.assertEqual(self.monitor.last_ocr_profile["call_count"], 1)

    def test_red_retry_is_traced_and_earlier_data_cannot_leak(self):
        self.read(_Tesseract([_RAW]))
        fake = _Tesseract(["", _RAW])
        data = self.read(fake)
        self.assertEqual(data["parts"], "")
        profile = self.monitor.last_ocr_profile
        self.assertTrue(profile["general_retry"])
        self.assertEqual(profile["route"], "general_red_retry")
        self.assertEqual(profile["call_count"], 2)
        self.assertEqual(
            [item["phase"] for item in profile["calls"]],
            ["general_blue", "general_red_retry"],
        )

    def test_auxiliary_reuses_side_fields_without_subprocess(self):
        self.monitor._shadow_auxiliary_info = {
            "board": "PCB", "parts": "C10", "value": "0 <= 1 <= 5",
            "raw_text": "test",
        }
        data = self.read(_Tesseract([]))
        self.assertEqual(data["parts"], "C10")
        profile = self.monitor.last_ocr_profile
        self.assertEqual(profile["route"], "auxiliary_reuse")
        self.assertEqual(profile["call_count"], 0)


class ExternalCropLatencyTests(unittest.TestCase):
    def test_side_crop_io_profile_separates_bar_crop_and_png_times(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        monitor = screen_monitor.ScreenMonitor()
        image = np.zeros((840, 1165, 3), dtype=np.uint8)
        small = np.ones((30, 40, 3), dtype=np.uint8)
        blue, red = (200, 400, 220, 15), (700, 400, 200, 15)

        def fake_crop(frame, bar, height, label=""):
            monitor._write_debug_crop(
                screen_monitor.DEBUG_DIR / (label + "_test.png"), small
            )
            return small.copy()

        info = {"board": "B1", "parts": "R1", "value": "1 <= 2 <= 3"}
        with TemporaryDirectory() as tmp, patch.object(
            screen_monitor, "DEBUG_DIR", Path(tmp)
        ), patch.object(
            monitor, "_find_color_bar", return_value=blue
        ), patch.object(
            monitor, "_find_sibling_bar", return_value=red
        ), patch.object(
            monitor, "_extract_photo", side_effect=fake_crop
        ), patch.object(
            monitor, "_extract_text_info", return_value=info
        ), patch.object(
            screen_monitor.cv2, "imwrite", return_value=True
        ) as writer:
            monitor.process_external_image(image)
            side = dict(monitor.last_external_timing)
            self.assertEqual(side["debug_png_write_count"], 2)
            self.assertGreaterEqual(side["debug_png_write_ms"], 0)
            self.assertGreaterEqual(side["bars_detect_ms"], 0)
            self.assertGreaterEqual(side["sample_crop_ms"], 0)
            self.assertGreaterEqual(side["test_crop_ms"], 0)
            self.assertGreaterEqual(side["layout_crop_ms"], 0)
            self.assertEqual(writer.call_count, 2)
            self.assertIsNone(monitor._external_crop_io_profile)
            self.assertNotIn("B1", str(side))

            # Sombra preserva a politica existente: nenhum PNG gravado.
            monitor._shadow_fast = True
            monitor.process_external_image(image)
            self.assertEqual(
                monitor.last_external_timing["debug_png_write_count"], 0
            )
            self.assertEqual(
                monitor.last_external_timing["debug_png_write_ms"], 0.0
            )
            self.assertEqual(writer.call_count, 2)

    def test_failed_png_write_does_not_leak_crop_profiler(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        monitor = screen_monitor.ScreenMonitor()
        small = np.ones((30, 40, 3), dtype=np.uint8)
        blue, red = (200, 400, 220, 15), (700, 400, 200, 15)

        def broken_crop(frame, bar, height, label=""):
            monitor._write_debug_crop(
                screen_monitor.DEBUG_DIR / "broken.png", small
            )
            raise RuntimeError("Falha no recorte")

        with TemporaryDirectory() as tmp, patch.object(
            screen_monitor, "DEBUG_DIR", Path(tmp)
        ), patch.object(
            monitor, "_find_color_bar", return_value=blue
        ), patch.object(
            monitor, "_find_sibling_bar", return_value=red
        ), patch.object(
            monitor, "_extract_photo", side_effect=broken_crop
        ), patch.object(
            screen_monitor.cv2, "imwrite", side_effect=OSError("sem disco")
        ):
            with self.assertRaises(OSError):
                monitor.process_external_image(
                    np.zeros((840, 1165, 3), dtype=np.uint8)
                )
        self.assertIsNone(monitor._external_crop_io_profile)


class ReceiverStabilityTimingTests(unittest.TestCase):
    def setUp(self):
        from src.services.network_receiver import NetworkReceiver
        self.receiver = NetworkReceiver()
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_side_gap_measured_without_changing_two_frame_gate(self):
        receiver = self.receiver
        signature = receiver._frame_signature(self.frame)
        self.assertEqual(receiver.STABLE_REQUIRED_FRAMES, 2)
        self.assertEqual(
            receiver._stage_latest_candidate(self.frame, signature, "XP", 10.0),
            1,
        )
        self.assertEqual(
            receiver._stage_latest_candidate(self.frame, signature, "XP", 10.42),
            2,
        )
        self.assertEqual(receiver._candidate_last_pair_gap_ms, 420.0)

    def test_auxiliary_settle_and_gap_are_observational(self):
        receiver = self.receiver
        receiver.set_auxiliary_image_mode(True)
        self.assertTrue(receiver.arm_auxiliary_capture("TOP"))
        start = receiver._aux_not_before
        self.assertFalse(
            receiver._auxiliary_frame_ready(self.frame, at=start - 0.01)
        )
        self.assertEqual(receiver._aux_pre_settle_frames, 1)
        self.assertFalse(
            receiver._auxiliary_frame_ready(self.frame, at=start + 0.01)
        )
        self.assertTrue(
            receiver._auxiliary_frame_ready(self.frame, at=start + 0.46)
        )
        self.assertEqual(receiver._aux_stable_frames, 2)
        self.assertAlmostEqual(receiver._aux_last_pair_gap_ms, 450.0)


if __name__ == "__main__":
    unittest.main()
