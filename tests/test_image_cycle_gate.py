import unittest
from pathlib import Path

import numpy as np

from src.services.image_cycle_gate import ImageCycleGate
from src.services.network_receiver import NetworkReceiver
from src.ui.network_image_cycle_gate import install_network_image_cycle_gate


ROOT = Path(__file__).resolve().parents[1]


class ImageCycleGateTests(unittest.TestCase):
    def test_only_one_image_can_reserve_cycle(self):
        gate = ImageCycleGate()
        self.assertTrue(gate.try_reserve())
        self.assertFalse(gate.try_reserve())
        self.assertFalse(gate.is_open())

        snapshot = gate.snapshot()
        self.assertEqual(snapshot.generation, 1)
        self.assertEqual(snapshot.ignored_images, 1)

        gate.release()
        self.assertTrue(gate.is_open())
        self.assertTrue(gate.try_reserve())
        self.assertEqual(gate.snapshot().generation, 2)

    def test_manual_lock_requires_explicit_release(self):
        gate = ImageCycleGate()
        gate.lock()
        self.assertFalse(gate.try_reserve())
        gate.release()
        self.assertTrue(gate.try_reserve())


class NetworkFrameSignatureTests(unittest.TestCase):
    def test_same_frozen_screen_is_recognized(self):
        image = np.full((480, 640, 3), 35, dtype=np.uint8)
        image[120:360, 170:470] = (80, 145, 210)
        first = NetworkReceiver._frame_signature(image)
        second = NetworkReceiver._frame_signature(image.copy())
        self.assertTrue(NetworkReceiver._same_signature(first, second))

    def test_small_compression_noise_does_not_create_new_piece(self):
        rng = np.random.default_rng(7)
        image = np.full((480, 640, 3), 90, dtype=np.uint8)
        noisy = np.clip(
            image.astype(np.int16) + rng.integers(-1, 2, image.shape),
            0,
            255,
        ).astype(np.uint8)
        self.assertTrue(
            NetworkReceiver._same_signature(
                NetworkReceiver._frame_signature(image),
                NetworkReceiver._frame_signature(noisy),
            )
        )

    def test_real_visual_change_is_accepted_as_new_image(self):
        first_image = np.full((480, 640, 3), 45, dtype=np.uint8)
        second_image = first_image.copy()
        second_image[150:330, 230:410] = (235, 235, 235)
        self.assertFalse(
            NetworkReceiver._same_signature(
                NetworkReceiver._frame_signature(first_image),
                NetworkReceiver._frame_signature(second_image),
            )
        )

    def test_localized_defect_change_is_not_suppressed(self):
        first_image = np.full((480, 640, 3), 65, dtype=np.uint8)
        second_image = first_image.copy()
        second_image[220:270, 300:350] = (230, 230, 230)
        self.assertFalse(
            NetworkReceiver._same_signature(
                NetworkReceiver._frame_signature(first_image),
                NetworkReceiver._frame_signature(second_image),
            )
        )


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.text_value = ""

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setText(self, text):
        self.text_value = str(text)


class FakeReceiver:
    def __init__(self):
        self.locked = False
        self.release_count = 0

    def lock_image_gate(self):
        self.locked = True

    def release_image_gate(self):
        self.locked = False
        self.release_count += 1


class FakePanel:
    def __init__(self):
        self.network_receiver = FakeReceiver()
        self.is_locked = False
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.accepted_images = 0
        self.started_captures = 0
        self.status_messages = []
        self.btn_start = FakeButton()
        self.btn_skip = FakeButton()
        self.btn_save_ok = FakeButton()
        self.btn_save_ng = FakeButton()

    def handle_network_image(self, _image, _ip):
        self.accepted_images += 1
        self.is_locked = True

    def start_monitoring(self):
        self.started_captures += 1
        self.is_locked = True

    def skip_image(self):
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = False

    def save_label(self, _decision, source="button"):
        self.is_locked = False
        return source

    def update_brain_status(self, message, _active=False):
        self.status_messages.append(message)

    def _reset_confidence_panel(self):
        pass

    def _reset_reference_panel(self):
        pass

    def _reset_aoi_info(self):
        self.current_aoi_info = {}
        self.current_analysis = None


class FakePresenter:
    def __init__(self, panel=None):
        self.panel = panel

    def sync(self, force=False):
        return force


class ExplodingDiscardPanel(FakePanel):
    # FakePanel já recebe o wrapper em outro teste; esta flag própria força uma
    # instalação independente sobre o método que lança a exceção.
    _network_image_cycle_gate_installed = False

    def skip_image(self):
        raise RuntimeError("debugger falhou durante reset")


class ExplodingPresenter(FakePresenter):
    pass


class NetworkImageCycleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_network_image_cycle_gate(FakePanel, FakePresenter)
        install_network_image_cycle_gate(ExplodingDiscardPanel, ExplodingPresenter)

    def test_new_network_images_do_not_replace_pending_capture(self):
        panel = FakePanel()

        panel.handle_network_image(object(), "192.168.0.10")
        self.assertEqual(panel.accepted_images, 1)
        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.network_receiver.locked)

        panel.handle_network_image(object(), "192.168.0.10")
        panel.handle_network_image(object(), "192.168.0.10")
        self.assertEqual(panel.accepted_images, 1)
        self.assertEqual(panel.capture_cycle_ignored_signals, 2)

        panel.save_label("OK", source="button")
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)

        panel.handle_network_image(object(), "192.168.0.10")
        self.assertEqual(panel.accepted_images, 2)

    def test_discard_releases_next_image(self):
        panel = FakePanel()
        panel.handle_network_image(object(), "192.168.0.10")
        panel.skip_image()

        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)

        panel.handle_network_image(object(), "192.168.0.10")
        self.assertEqual(panel.accepted_images, 2)

    def test_discard_exception_is_recovered_without_closing_application(self):
        panel = ExplodingDiscardPanel()
        panel.current_analysis = {"confidence": 0.70}
        panel.current_sample = object()
        panel.current_ng = object()
        panel.current_aoi_info = {"category": "FALTANDO"}
        panel.is_locked = True
        panel.capture_cycle_active = True
        panel.network_receiver.locked = True

        panel.skip_image()

        self.assertFalse(panel.is_locked)
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertIsNone(panel.current_analysis)
        self.assertIsNone(panel.current_sample)
        self.assertIsNone(panel.current_ng)
        self.assertTrue(panel.status_messages)

    def test_local_capture_cannot_replace_pending_image(self):
        panel = FakePanel()
        panel.handle_network_image(object(), "192.168.0.10")
        panel.start_monitoring()

        self.assertEqual(panel.started_captures, 0)
        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.status_messages)


class ImageCycleContractTests(unittest.TestCase):
    def test_network_receiver_reserves_before_emitting(self):
        source = (
            ROOT / "src" / "services" / "network_receiver.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if not self._image_gate.try_reserve()", source)
        self.assertLess(
            source.index("if not self._image_gate.try_reserve()"),
            source.index("self.image_received.emit(img, ip_origem)"),
        )
        self.assertIn("if not self._image_gate.is_open()", source)
        self.assertIn("self._discard_payload(conexao, tamanho_total)", source)
        self.assertIn("self._require_image_change", source)
        self.assertIn("self._same_signature", source)

    def test_cycle_gate_is_outermost_wrapper(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        production_call = (
            "install_production_confidence_gate("
            "ControlPanel, OperationalControlsPresenter)"
        )
        cycle_call = (
            "install_network_image_cycle_gate("
            "ControlPanel, OperationalControlsPresenter)"
        )
        self.assertLess(source.index(production_call), source.index(cycle_call))

    def test_commands_remain_available_while_images_are_blocked(self):
        source = (
            ROOT / "src" / "services" / "network_receiver.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            source.index('if cabecalho_str.startswith("CMD_")'),
            source.index("if not self._image_gate.is_open()"),
        )


if __name__ == "__main__":
    unittest.main()
