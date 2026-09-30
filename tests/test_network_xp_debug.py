import unittest

from src.ui.network_xp_debug import (
    DEBUG_SCHEMA,
    format_network_debug_report,
)


class NetworkXPDebugFormatTests(unittest.TestCase):
    def test_report_contains_rejection_context_and_full_json(self):
        record = {
            "schema": DEBUG_SCHEMA,
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
        self.assertIn("169.254.95.200", report)
        self.assertIn("tela sem epicentro de anomalia", report)
        self.assertIn("missing_epicenter", report)
        self.assertIn("Anomalias brutas: 2", report)
        self.assertIn("INDÍCIOS DIAGNÓSTICOS", report)
        self.assertIn("REGISTRO COMPLETO (JSON)", report)

    def test_empty_record_has_safe_message(self):
        report = format_network_debug_report({})
        self.assertIn("Nenhuma imagem recebida", report)


if __name__ == "__main__":
    unittest.main()
