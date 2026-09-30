import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton

from src.ui.network_xp_debug import (
    DEBUG_SCHEMA,
    copy_network_image_to_clipboard,
    format_network_debug_report,
    network_debug_image_available,
    sync_network_debug_controls,
)


class NetworkXPDebugFormatTests(unittest.TestCase):
    def test_report_contains_rejection_context_and_full_json(self):
        record = {
            "schema": DEBUG_SCHEMA,
            "event_id": "evt-001",
            "timestamp": "2026-09-30T07:55:00.000",
            "source_ip": "169.254.95.200",
            "stage": "aoi_intake_validation",
            "mode": "Modo Produção",
            "transport": {
                "image": {
                    "valid": True,
                    "shape": [840, 1165, 3],
                    "dtype": "uint8",
                },
                "stable_required_frames": 2,
            },
            "cycle": {
                "accepting_images": False,
                "generation": 7,
                "ignored_images": 0,
            },
            "validation_message": "tela sem epicentro de anomalia",
            "validation": {
                "valid": False,
                "reason": "missing_epicenter",
                "raw_anomaly_count": 2,
                "old_epicenter_count": 0,
                "real_epicenter_count": 0,
                "global_box_info": {"w": 410, "h": 250},
                "diagnostic_hints": [
                    "A marcação verde aparece no TESTE, mas o Radar não encontrou candidato."
                ],
            },
        }

        report = format_network_debug_report(record)

        self.assertIn("VISIONX - DEBUG DE ENTRADA WINDOWS XP", report)
        self.assertIn("Evento: evt-001", report)
        self.assertIn("169.254.95.200", report)
        self.assertIn("tela sem epicentro de anomalia", report)
        self.assertIn("missing_epicenter", report)
        self.assertIn("Anomalias brutas: 2", report)
        self.assertIn("INDÍCIOS DIAGNÓSTICOS", report)
        self.assertIn("REGISTRO COMPLETO (JSON)", report)

    def test_empty_record_has_safe_message(self):
        report = format_network_debug_report({})
        self.assertIn("Nenhuma imagem recebida", report)


class NetworkXPImageClipboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel(event_id="evt-001"):
        class Panel:
            pass

        panel = Panel()
        panel.network_intake_last_validation = {
            "schema": DEBUG_SCHEMA,
            "event_id": event_id,
        }
        panel.network_intake_last_image_event_id = event_id
        panel.network_intake_last_image = np.zeros((12, 18, 3), dtype=np.uint8)
        panel.network_intake_last_image[:, :] = (10, 20, 230)  # BGR
        panel.btn_copy_network_debug = QPushButton("Copiar debug")
        panel.btn_copy_network_image = QPushButton("Copiar imagem")
        return panel

    def test_same_event_enables_both_copy_actions(self):
        panel = self._panel()
        sync_network_debug_controls(panel)

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertTrue(panel.btn_copy_network_image.isEnabled())
        self.assertTrue(network_debug_image_available(panel))

    def test_different_event_blocks_image_copy(self):
        panel = self._panel()
        panel.network_intake_last_image_event_id = "evt-other"
        sync_network_debug_controls(panel)

        self.assertTrue(panel.btn_copy_network_debug.isEnabled())
        self.assertFalse(panel.btn_copy_network_image.isEnabled())
        self.assertFalse(network_debug_image_available(panel))
        self.assertFalse(copy_network_image_to_clipboard(panel))

    def test_copy_image_places_exact_frame_on_clipboard(self):
        panel = self._panel()

        self.assertTrue(copy_network_image_to_clipboard(panel))
        copied = QApplication.clipboard().image()
        self.assertFalse(copied.isNull())
        self.assertEqual(copied.width(), 18)
        self.assertEqual(copied.height(), 12)

        pixel = copied.pixelColor(0, 0)
        self.assertEqual(pixel.red(), 230)
        self.assertEqual(pixel.green(), 20)
        self.assertEqual(pixel.blue(), 10)


class NetworkXPDebugUILayoutTests(unittest.TestCase):
    def test_debug_actions_live_outside_global_status_bar(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn("def _build_network_debug_bar", source)
        self.assertIn('QFrame#networkDebugFrame', source)
        self.assertIn('btn_copy_network_image', source)
        self.assertIn('copy_network_image_to_clipboard', source)
        self.assertNotIn(
            "status_layout.addWidget(window.btn_copy_network_debug)",
            source,
        )
        self.assertNotIn(
            "status_layout.addWidget(window.btn_copy_network_image)",
            source,
        )

    def test_copy_actions_share_one_responsive_action_group(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "self.network_debug_actions_layout.addWidget(\n"
            "            window.btn_copy_network_debug,",
            source,
        )
        self.assertIn(
            "self.network_debug_actions_layout.addWidget(\n"
            "            window.btn_copy_network_image,",
            source,
        )
        self.assertIn(
            'window.btn_copy_network_image = QPushButton("Copiar imagem XP")',
            source,
        )

        compact = source[
            source.index("if compact:"):
            source.index("else:", source.index("if compact:"))
        ]
        self.assertIn(
            "grid.addWidget(window.network_debug_actions, 2, 0)",
            compact,
        )
        self.assertNotIn("btn_copy_network_debug", compact)
        self.assertNotIn("btn_copy_network_image", compact)

    def test_network_debug_buttons_have_clickable_responsive_policy(self):
        source = open(
            "src/ui/control_panel_ui.py",
            encoding="utf-8",
        ).read()

        self.assertIn("min-height: 40px", source)
        self.assertIn(
            "window.btn_copy_network_debug.setFocusPolicy(Qt.FocusPolicy.StrongFocus)",
            source,
        )
        self.assertIn(
            "window.btn_copy_network_image.setFocusPolicy(Qt.FocusPolicy.StrongFocus)",
            source,
        )
        self.assertIn(
            "window.network_debug_actions.setMinimumWidth(0 if compact else 300)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
