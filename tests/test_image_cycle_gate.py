import unittest
from pathlib import Path

from src.services.image_cycle_gate import ImageCycleGate
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
        self.accepted_images = 0
        self.started_captures = 0
        self.status_messages = []

    def handle_network_image(self, _image, _ip):
        self.accepted_images += 1
        self.is_locked = True

    def start_monitoring(self):
        self.started_captures += 1
        self.is_locked = True

    def skip_image(self):
        self.is_locked = False

    def save_label(self, _decision, source="button"):
        self.is_locked = False
        return source

    def update_brain_status(self, message, _active=False):
        self.status_messages.append(message)


class FakePresenter:
    def __init__(self, panel=None):
        self.panel = panel

    def sync(self, force=False):
        return force


class NetworkImageCycleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_network_image_cycle_gate(FakePanel, FakePresenter)

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

    def test_cycle_gate_is_outermost_wrapper(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_production_confidence_gate"),
            source.index("install_network_image_cycle_gate"),
        )

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
