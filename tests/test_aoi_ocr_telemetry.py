"""Integração das medições OCR reais sem mudar Board/Parts/Value."""
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

    def test_valid_general_ocr_is_one_call_and_fields_unchanged(self):
        fake = _Tesseract([_RAW])
        data = self.read(fake)
        self.assertEqual(data["board"], "P22-22200 (PRINCIPAL) L13")
        self.assertEqual(data["parts"], "R375")
        self.assertEqual(data["value"], "12 <= 0 <= 60 FALTANDO")
        profile = self.monitor.last_ocr_profile
        self.assertEqual(profile["route"], "general_blue")
        self.assertEqual(profile["parts_reason"], "valid_general_reference")
        self.assertEqual(profile["call_count"], 1)
        self.assertEqual(profile["calls"][0]["phase"], "general_blue")
        self.assertFalse(profile["general_retry"])
        self.assertNotIn("R375", str(profile))

    def test_red_retry_is_traced_and_earlier_data_cannot_leak(self):
        self.read(_Tesseract([_RAW]))
        fake = _Tesseract(["", _RAW])
        data = self.read(fake)
        self.assertEqual(data["parts"], "R375")
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
