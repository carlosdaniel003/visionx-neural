import unittest
from unittest.mock import patch

import numpy as np

from src.services.network_receiver import NetworkReceiver
import src.ui.network_aoi_intake_filter as intake_module
from src.ui.network_aoi_intake_filter import (
    install_network_aoi_intake_filter,
    validate_network_inspection,
)


class NetworkLatestCandidateTests(unittest.TestCase):
    def setUp(self):
        self.receiver = NetworkReceiver(port=0)
        self.first = np.full((240, 320, 3), 40, dtype=np.uint8)
        self.second = self.first.copy()
        self.second[80:160, 110:210] = 210

    def test_requires_two_stable_frames(self):
        signature = self.receiver._frame_signature(self.first)
        self.assertEqual(
            self.receiver._stage_latest_candidate(
                self.first,
                signature,
                "192.168.0.10",
            ),
            1,
        )
        self.assertIsNone(self.receiver._candidate_snapshot())

        self.assertEqual(
            self.receiver._stage_latest_candidate(
                self.first.copy(),
                signature.copy(),
                "192.168.0.10",
            ),
            2,
        )
        candidate = self.receiver._candidate_snapshot()
        self.assertIsNotNone(candidate)
        self.assertTrue(np.array_equal(candidate[0], self.first))

    def test_newer_different_frame_replaces_candidate(self):
        first_signature = self.receiver._frame_signature(self.first)
        second_signature = self.receiver._frame_signature(self.second)
        self.receiver._stage_latest_candidate(
            self.first,
            first_signature,
            "192.168.0.10",
        )
        count = self.receiver._stage_latest_candidate(
            self.second,
            second_signature,
            "192.168.0.10",
        )
        self.assertEqual(count, 1)
        self.assertIsNone(self.receiver._candidate_snapshot())
        self.assertTrue(
            np.array_equal(
                self.receiver._candidate_image,
                self.second,
            )
        )

    def test_rejected_central_is_ignored_until_screen_changes(self):
        signature = self.receiver._frame_signature(self.first)
        self.receiver._reserved_signature = signature.copy()
        self.assertTrue(
            self.receiver.mark_reserved_image_rejected(
                "tela sem epicentro de anomalia"
            )
        )
        self.receiver.release_image_gate()

        self.assertTrue(self.receiver._is_rejected_repeat(signature))
        self.assertFalse(self.receiver._require_image_change)
        changed = self.receiver._frame_signature(self.second)
        self.assertFalse(self.receiver._is_rejected_repeat(changed))

    def test_only_confirmed_reservation_becomes_last_valid_anomaly(self):
        signature = self.receiver._frame_signature(self.first)
        self.receiver._reserved_signature = signature.copy()
        self.assertTrue(self.receiver.confirm_reserved_image())
        self.assertTrue(
            np.array_equal(
                self.receiver._last_accepted_signature,
                signature,
            )
        )
        self.receiver.release_image_gate()
        self.assertTrue(self.receiver._require_image_change)


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setText(self, value):
        self.text = str(value)


class FakeCombo:
    def __init__(self, text="Modo Teste"):
        self.text = text

    def currentText(self):
        return self.text


class FakeReceiver:
    def __init__(self, events):
        self.events = events
        self.confirmed = 0
        self.rejected = []
        self.released = 0

    def confirm_reserved_image(self):
        self.confirmed += 1
        self.events.append("confirm")
        return True

    def mark_reserved_image_rejected(self, reason):
        self.rejected.append(str(reason))
        self.events.append("reject")
        return True

    def release_image_gate(self):
        self.released += 1
        self.events.append("release")


class FakePanel:
    def __init__(self):
        self.events = []
        self.network_receiver = FakeReceiver(self.events)
        self.combo_mode = FakeCombo()
        self.capture_cycle_source = "network"
        self.capture_cycle_active = True
        self.capture_cycle_ignored_signals = 0
        self.capture_cycle_network_generation = 1
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = True
        self.status = []
        self.btn_start = FakeButton()
        self.btn_save_ok = FakeButton()
        self.btn_save_ng = FakeButton()
        self.btn_skip = FakeButton()

    def process_aoi_images(self, sample_crop, ng_crop, aoi_info):
        self.events.append("original")
        self.current_sample = sample_crop
        self.current_ng = ng_crop
        self.current_aoi_info = dict(aoi_info)
        self.current_analysis = {"detail": {}}
        return "processed"

    def _reset_confidence_panel(self):
        self.events.append("reset_confidence")

    def _reset_reference_panel(self):
        self.events.append("reset_reference")

    def _reset_aoi_info(self):
        self.current_aoi_info = {}
        self.current_analysis = None

    def update_brain_status(self, message, _active=False):
        self.status.append(str(message))


install_network_aoi_intake_filter(FakePanel)


class NetworkIntakeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.sample = np.full((100, 120, 3), 60, dtype=np.uint8)
        self.test = self.sample.copy()
        self.test[30:70, 40:80] = 190

    def test_invalid_network_screen_never_reaches_analysis(self):
        panel = FakePanel()
        with patch.object(
            intake_module,
            "validate_network_inspection",
            return_value=(
                False,
                "tela sem epicentro de anomalia",
                {"valid": False},
            ),
        ):
            result = panel.process_aoi_images(self.sample, self.test, {})

        self.assertFalse(result)
        self.assertNotIn("original", panel.events)
        self.assertEqual(panel.network_receiver.released, 1)
        self.assertTrue(panel.network_receiver.rejected)
        self.assertFalse(panel.capture_cycle_active)
        self.assertIsNone(panel.capture_cycle_source)
        self.assertFalse(panel.is_locked)
        self.assertTrue(panel.status)

    def test_valid_network_screen_is_confirmed_and_processed(self):
        panel = FakePanel()
        with patch.object(
            intake_module,
            "validate_network_inspection",
            return_value=(
                True,
                "epicentro válido",
                {"valid": True, "focus_box": [10, 10, 30, 30]},
            ),
        ):
            result = panel.process_aoi_images(
                self.sample,
                self.test,
                {"value": "Missing"},
            )

        self.assertEqual(result, "processed")
        self.assertEqual(panel.network_receiver.confirmed, 1)
        self.assertEqual(panel.events[:2], ["original", "confirm"])
        detail = panel.current_analysis["detail"]
        self.assertTrue(detail["network_intake_validation"]["valid"])
        self.assertEqual(detail["network_intake_source"], "windows_xp")

    def test_production_confirms_before_original_can_auto_finish(self):
        panel = FakePanel()
        panel.combo_mode.text = "Modo Produção"
        with patch.object(
            intake_module,
            "validate_network_inspection",
            return_value=(True, "epicentro válido", {"valid": True}),
        ):
            panel.process_aoi_images(self.sample, self.test, {})

        self.assertEqual(panel.events[:2], ["confirm", "original"])

    def test_local_mss_path_is_unchanged(self):
        panel = FakePanel()
        panel.capture_cycle_source = "local"
        with patch.object(
            intake_module,
            "validate_network_inspection",
            side_effect=AssertionError("MSS não deve usar filtro de rede"),
        ):
            result = panel.process_aoi_images(self.sample, self.test, {})

        self.assertEqual(result, "processed")
        self.assertEqual(panel.network_receiver.confirmed, 0)
        self.assertEqual(panel.network_receiver.released, 0)


class NetworkInspectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.sample = np.full((100, 120, 3), 60, dtype=np.uint8)
        self.test = self.sample.copy()
        self.test[30:70, 40:80] = 190

    @patch.object(intake_module.EpicenterExtractor, "extract_focus")
    @patch.object(intake_module, "detect_anomalies")
    def test_missing_epicenter_rejects_central_screen(
        self,
        detect_mock,
        focus_mock,
    ):
        detect_mock.return_value = ([], [], {}, np.array([]), np.array([]))
        focus_mock.return_value = ([], np.array([]), np.array([]))

        valid, reason, audit = validate_network_inspection(
            self.sample,
            self.test,
        )
        self.assertFalse(valid)
        self.assertIn("sem epicentro", reason)
        self.assertEqual(audit["reason"], "missing_epicenter")

    @patch.object(intake_module.EpicenterExtractor, "extract_focus")
    @patch.object(intake_module, "detect_anomalies")
    def test_valid_epicenter_accepts_aoi_anomaly(
        self,
        detect_mock,
        focus_mock,
    ):
        detect_mock.return_value = (
            [object()],
            [(10, 10, 30, 30)],
            {"box": [0, 0, 100, 100]},
            np.array([]),
            np.array([]),
        )
        focus = np.full((30, 30, 3), 100, dtype=np.uint8)
        focus_mock.return_value = ([(10, 10, 30, 30)], focus, focus.copy())

        valid, reason, audit = validate_network_inspection(
            self.sample,
            self.test,
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "epicentro válido")
        self.assertEqual(audit["focus_box"], [10, 10, 30, 30])


if __name__ == "__main__":
    unittest.main()
