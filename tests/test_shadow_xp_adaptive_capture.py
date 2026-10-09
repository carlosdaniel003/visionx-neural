"""XP shadow adaptive capture: lease, ACK and no 0/1 emitted by ODIN."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.ui.control_panel import ControlPanel

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 100.0
    def time(self):
        return self.now


class XpAdaptiveCaptureTests(unittest.TestCase):
    def _xp_functions(self):
        source = (ROOT / "agente_industrial_xp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {"configurar_captura_sombra", "pausa_pos_envio"}
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {
                    "DEFAULT_CAPTURE_PAUSE_SECONDS",
                    "SHADOW_CAPTURE_PAUSE_SECONDS",
                    "SHADOW_LEASE_SECONDS",
                    "_shadow_capture_until",
                } for t in node.targets
            ):
                nodes.append(node)
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
        clock = Clock()
        ns = {"time": clock}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "xp-agent", "exec"), ns)
        return source, clock, ns

    def test_shadow_only_pacing_and_lease_expiry(self):
        source, clock, xp = self._xp_functions()
        self.assertEqual(xp["pausa_pos_envio"](), 3.0)
        xp["configurar_captura_sombra"](True)
        self.assertEqual(xp["pausa_pos_envio"](), 0.18)
        clock.now += 24.0
        self.assertEqual(xp["pausa_pos_envio"](), 0.18)
        clock.now += 2.0
        self.assertEqual(xp["pausa_pos_envio"](), 3.0)
        xp["configurar_captura_sombra"](True)
        xp["configurar_captura_sombra"](False)
        self.assertEqual(xp["pausa_pos_envio"](), 3.0)
        self.assertIn("time.sleep(pausa)", source)
        self.assertIn("ACK_SHADOW_ON", source)
        self.assertIn("ACK_SHADOW_OFF", source)
        ast.parse(source)

    def test_transport_requires_ack_and_never_emits_zero_one(self):
        class Panel:
            last_xp_ip = "169.254.87.100"
        panel = Panel()
        fake = MagicMock()
        connection = fake.return_value.__enter__.return_value
        connection.recv.side_effect = [b"ACK_SHADOW_ON", b"ACK_SHADOW_OFF"]
        with patch("src.ui.control_panel.socket.socket", fake):
            self.assertTrue(ControlPanel._send_xp_shadow_control(panel, True))
            self.assertTrue(ControlPanel._send_xp_shadow_control(panel, False))
        payloads = [call.args[0] for call in connection.sendall.call_args_list]
        self.assertEqual(payloads, [b"VISIONX_SHADOW_ON", b"VISIONX_SHADOW_OFF"])
        self.assertFalse(any(b"PRESS_0" in x or b"PRESS_1" in x for x in payloads))

    def test_old_agent_without_ack_does_not_claim_fast_capture(self):
        class Panel:
            last_xp_ip = "169.254.87.100"
        panel = Panel()
        fake = MagicMock()
        fake.return_value.__enter__.return_value.recv.return_value = b""
        with patch("src.ui.control_panel.socket.socket", fake):
            self.assertFalse(ControlPanel._send_xp_shadow_control(panel, True))
        self.assertIsNone(panel._xp_shadow_last_ack)

    def test_shadow_timer_only_runs_in_shadow(self):
        class Mode:
            current = "Modo Sombra"
            def currentText(self):
                return self.current
        class Panel:
            combo_mode = Mode()
            _xp_shadow_lease_timer = MagicMock()
            def __init__(self):
                self.calls = []
            def _send_xp_shadow_control(self, enabled, force=False):
                self.calls.append((enabled, force))
        panel = Panel()
        ControlPanel._on_xp_mode_changed(panel, "Modo Sombra")
        ControlPanel._renew_xp_shadow_lease(panel)
        self.assertEqual(panel.calls, [(True, True), (True, True)])
        panel.combo_mode.current = "Modo Produção"
        ControlPanel._on_xp_mode_changed(panel, "Modo Produção")
        ControlPanel._renew_xp_shadow_lease(panel)
        self.assertEqual(panel.calls[-1], (False, True))
        self.assertEqual(len(panel.calls), 3)
        panel._xp_shadow_lease_timer.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
