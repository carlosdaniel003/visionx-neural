"""SIDE obrigatoria antes de cada captura principal recebida do XP.

Nao atesta estado eletrico da iluminacao; comprova a sequencia de
solicitacao DOWN, acomodacao e novos frames antes de iniciar a analise.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication

from src.services.network_receiver import NetworkReceiver
from src.ui.control_panel import ControlPanel


class SidePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.receiver = NetworkReceiver(port=0)
        self.requests = []
        self.receiver.side_preflight_requested.connect(
            lambda ip, token: self.requests.append((ip, token))
        )
        self.ip = "169.254.95.200"

    def test_preflight_requests_down_before_any_image_arrives(self):
        # O primeiro comando deve existir sem _side_preflight_ready nem PNG.
        self.assertEqual(self.receiver.last_delivered_image_received_at, 0.0)
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=5.0):
            self.assertTrue(self.receiver.prime_side_preflight(self.ip))
        self.assertEqual(len(self.requests), 1)
        token = self.requests[0][1]
        self.assertTrue(self.receiver.is_side_preflight_pending(token))
        self.assertFalse(self.receiver.prime_side_preflight(""))
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=5.1):
            self.assertTrue(self.receiver.prime_side_preflight(self.ip))
        self.assertEqual(len(self.requests), 1, "nao repetir DOWN no mesmo ciclo")

        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=5.2):
            self.assertTrue(self.receiver.complete_side_preflight(token, True))
        self.assertFalse(self.receiver._side_preflight_ready(
            self.ip, received_at=5.549,
        ))
        self.assertTrue(self.receiver._side_preflight_ready(
            self.ip, received_at=5.551,
        ))

    def test_peer_discovery_primes_down_before_capture_rate_negotiation(self):
        class Mode:
            def currentText(self):
                return "Modo Produção"

        class Panel:
            def __init__(self, receiver):
                self.network_receiver = receiver
                self.last_xp_ip = ""
                self.combo_mode = Mode()
                self.events = []

            def send_command_to_xp(self, key):
                self.events.append("send_" + key)
                return True

            def change_lighting(self, lighting, source):
                self.events.append("visual_" + lighting)

            def _send_xp_shadow_control(self, enabled):
                self.events.append("capture_rate_" + str(enabled))

        panel = Panel(self.receiver)
        self.receiver.side_preflight_requested.connect(
            lambda ip, token: ControlPanel.handle_side_preflight_requested(
                panel, ip, token,
            )
        )
        ControlPanel.handle_xp_peer_discovered(panel, self.ip)
        self.assertEqual(panel.events, [
            "send_DOWN", "visual_SIDE", "capture_rate_True",
        ])
        self.assertEqual(panel.last_xp_ip, self.ip)
        ControlPanel.handle_xp_peer_discovered(panel, self.ip)
        self.assertEqual(panel.events.count("send_DOWN"), 1)

    def test_network_cycle_release_primes_next_down_without_next_frame(self):
        from src.ui.network_image_cycle_gate import _release_cycle

        class Panel:
            def __init__(self, receiver, ip):
                self.network_receiver = receiver
                self.last_xp_ip = ip
                self.capture_cycle_source = "network"
                self.events = []

            def send_command_to_xp(self, key):
                self.events.append("send_" + key)
                return True

            def change_lighting(self, lighting, source):
                self.events.append("visual_" + lighting)

        panel = Panel(self.receiver, self.ip)
        self.receiver.side_preflight_requested.connect(
            lambda ip, token: ControlPanel.handle_side_preflight_requested(
                panel, ip, token,
            )
        )
        self.receiver.lock_image_gate()
        with patch(
            "src.services.knn_archive_auto_memory.resume_archive_auto_memory_after_inspection"
        ):
            _release_cycle(panel)
        self.assertEqual(panel.events, ["send_DOWN", "visual_SIDE"])
        self.assertIsNone(panel.capture_cycle_source)
        self.assertEqual(self.receiver.last_delivered_image_received_at, 0.0)
        self.assertEqual(
            self.receiver.last_side_preflight["status"], "SETTLING",
        )
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.receiver._side_preflight_ready(
            self.ip, received_at=0.0,
        ))
        self.assertEqual(panel.events.count("send_DOWN"), 1)

    def test_missing_peer_or_locked_gate_does_not_send_down(self):
        self.assertFalse(self.receiver.prime_side_preflight(""))
        self.assertEqual(self.requests, [])
        self.receiver.lock_image_gate()
        self.assertFalse(self.receiver.prime_side_preflight(self.ip))
        self.assertEqual(self.requests, [])

    def test_first_image_is_dropped_until_down_and_new_settled_image(self):
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=10.0):
            self.assertFalse(
                self.receiver._side_preflight_ready(self.ip, received_at=10.0)
            )
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(
            self.receiver.is_side_preflight_pending(self.requests[0][1] + 1)
        )
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=10.1):
            self.assertTrue(
                self.receiver.complete_side_preflight(self.requests[0][1], True)
            )
        # A foto anterior a PRESS_DOWN e todas as fotos em transicao
        # sao recusadas, mesmo que a analise anterior tenha sido SIDE.
        self.assertFalse(
            self.receiver._side_preflight_ready(self.ip, received_at=10.2)
        )
        self.assertFalse(
            self.receiver._side_preflight_ready(self.ip, received_at=10.449)
        )
        self.assertTrue(
            self.receiver._side_preflight_ready(self.ip, received_at=10.451)
        )
        self.assertEqual(
            self.receiver.last_side_preflight["command"], "PRESS_DOWN"
        )
        self.assertTrue(
            self.receiver.last_side_preflight["transport_sent"]
        )
        self.assertEqual(len(self.requests), 1)

    def test_next_main_cycle_requires_a_new_down_even_after_side(self):
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=10.0):
            self.receiver._side_preflight_ready(self.ip, received_at=10.0)
            token = self.requests[-1][1]
            self.receiver.complete_side_preflight(token, True)
        self.receiver._side_preflight_ready(self.ip, received_at=11.0)
        self.receiver.lock_image_gate()
        self.receiver.release_image_gate()
        self.assertEqual(self.receiver.last_side_preflight, {})
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=12.0):
            self.assertFalse(
                self.receiver._side_preflight_ready(self.ip, received_at=12.0)
            )
        self.assertEqual(len(self.requests), 2)
        self.assertNotEqual(self.requests[0][1], self.requests[1][1])

    def test_failed_down_blocks_frames_and_retries_without_stale_answer(self):
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=20.0):
            self.receiver._side_preflight_ready(self.ip, received_at=20.0)
            first_token = self.requests[-1][1]
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=20.1):
            self.assertTrue(self.receiver.complete_side_preflight(
                first_token, False
            ))
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=21.0):
            self.assertFalse(
                self.receiver._side_preflight_ready(self.ip, received_at=21.0)
            )
        self.assertEqual(len(self.requests), 1)
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=22.2):
            self.assertFalse(
                self.receiver._side_preflight_ready(self.ip, received_at=22.2)
            )
        second_token = self.requests[-1][1]
        self.assertEqual(len(self.requests), 2)
        self.assertNotEqual(first_token, second_token)
        self.assertFalse(self.receiver.complete_side_preflight(first_token, True))
        self.assertTrue(self.receiver.is_side_preflight_pending(second_token))

    def test_xp_top_before_first_capture_forces_fresh_side(self):
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=25.0):
            self.receiver._side_preflight_ready(self.ip, received_at=25.0)
            token = self.requests[-1][1]
            self.assertTrue(self.receiver.complete_side_preflight(
                token, True
            ))
        self.assertTrue(
            self.receiver._side_preflight_ready(self.ip, received_at=25.5)
        )
        self.assertFalse(
            self.receiver.invalidate_side_preflight_for_xp_light("SIDE")
        )
        self.assertTrue(
            self.receiver.invalidate_side_preflight_for_xp_light("TOP")
        )
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=26.0):
            self.assertFalse(
                self.receiver._side_preflight_ready(self.ip, received_at=26.0)
            )
        self.assertEqual(len(self.requests), 2)
        self.assertNotEqual(token, self.requests[-1][1])

    def test_top_during_auxiliary_cycle_does_not_rearm_main_gate(self):
        self.receiver.lock_image_gate()
        self.assertFalse(
            self.receiver.invalidate_side_preflight_for_xp_light("TOP")
        )
        self.assertFalse(
            self.receiver.invalidate_side_preflight_for_xp_light("MID")
        )

    def test_gate_reset_invalidates_late_down_confirmation(self):
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=30.0):
            self.receiver._side_preflight_ready(self.ip, received_at=30.0)
        token = self.requests[-1][1]
        self.receiver.lock_image_gate()
        self.receiver.release_image_gate()
        self.assertFalse(self.receiver.complete_side_preflight(token, True))
        self.assertFalse(self.receiver.is_side_preflight_pending(token))

    def test_panel_sends_down_once_and_only_announces_accepted_request(self):
        class Panel:
            def __init__(self, receiver):
                self.network_receiver = receiver
                self.last_xp_ip = ""
                self.commands = []
                self.visual = []
                self.sent = True

            def send_command_to_xp(self, value):
                self.commands.append(value)
                return self.sent

            def change_lighting(self, light, source):
                self.visual.append((light, source))

        panel = Panel(self.receiver)
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=50.0):
            self.receiver._side_preflight_ready(self.ip, received_at=50.0)
            token = self.requests[-1][1]
            ControlPanel.handle_side_preflight_requested(
                panel, self.ip, token
            )
        self.assertEqual(panel.commands, ["DOWN"])
        self.assertEqual(panel.visual, [("SIDE", "side_preflight")])
        self.assertEqual(panel.last_xp_ip, self.ip)
        ControlPanel.handle_side_preflight_requested(panel, self.ip, token)
        self.assertEqual(panel.commands, ["DOWN"])
        self.assertEqual(len(panel.visual), 1)

    def test_panel_failed_send_never_shows_side_as_selected(self):
        class Panel:
            def __init__(self, receiver):
                self.network_receiver = receiver
                self.last_xp_ip = ""
                self.visual = []

            def send_command_to_xp(self, value):
                return False

            def change_lighting(self, light, source):
                self.visual.append((light, source))

        panel = Panel(self.receiver)
        with patch("src.services.network_receiver.time.perf_counter",
                   return_value=60.0):
            self.receiver._side_preflight_ready(self.ip, received_at=60.0)
            ControlPanel.handle_side_preflight_requested(
                panel, self.ip, self.requests[-1][1]
            )
        self.assertEqual(panel.visual, [])
        self.assertEqual(
            self.receiver.last_side_preflight["status"], "RETRY_PENDING"
        )


if __name__ == "__main__":
    unittest.main()
