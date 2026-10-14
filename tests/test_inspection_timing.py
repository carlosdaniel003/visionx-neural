"""Regressoes do perfil temporal XP independente dos classificadores."""
import time
import unittest
from types import SimpleNamespace

from src.services.inspection_timing import (
    begin_cycle, finish_cycle, format_report, note_command,
    note_extraction, note_frame, note_steps, presentation_mark, snapshot,
)
from src.ui.network_xp_debug import format_network_debug_report


class InspectionTimingTests(unittest.TestCase):
    def test_full_cycle_tracks_three_lights_and_production_separately(self):
        panel = SimpleNamespace()
        base = time.perf_counter() - 10.0
        begin_cycle(
            panel, "event-42", "Modo Produção",
            {
                "payload_at": base, "ready_at": base + 0.01,
                "tcp_payload_ms": 15.0, "decode_ms": 8.0,
                "stability_ms": 220.0, "payload_to_dispatch_ms": 11.0,
            },
            fast_ack=True,
        )
        note_steps(panel, "SIDE", validation_ms=50, geometry_ms=70,
                   inference_ms=620, render_ms=120, processing_ms=850)
        note_command(panel, "TOP", base + 1.0, base + 1.02, True)
        note_frame(panel, "TOP", {
            "payload_at": base + 1.8, "ready_at": base + 1.82,
            "tcp_payload_ms": 24, "decode_ms": 9, "stability_ms": 300,
        })
        note_steps(panel, "TOP", inference_ms=250, ocr_ms=0)
        note_command(panel, "MID", base + 1.85, base + 1.87, True)
        note_frame(panel, "MID", {
            "payload_at": base + 2.5, "ready_at": base + 2.53,
            "tcp_payload_ms": 20, "decode_ms": 7, "stability_ms": 180,
        })
        note_steps(panel, "MID", inference_ms=410, render_ms=90)
        finish_cycle(panel, base + 3.5)

        presentation_mark(panel, "ready", base + 3.5)
        presentation_mark(panel, "presented", base + 10.5)
        presentation_mark(panel, "decision", base + 12.5)
        report = snapshot(panel, "event-42")
        self.assertEqual(report["cycle_side_to_result_ms"], 3500.0)
        self.assertEqual(report["lights"]["TOP"]["command"]["to_payload_ms"], 800.0)
        self.assertEqual(report["lights"]["TOP"]["command"]["to_ready_ms"], 820.0)
        self.assertEqual(report["lights"]["MID"]["command"]["to_ready_ms"], 680.0)
        self.assertEqual(report["production_presentation_ms"], 7000.0)
        self.assertEqual(
            report["production_post_presentation_decision_ms"], 2000.0
        )
        self.assertTrue(report["xp_fast_ack"])

        lines = "\n".join(format_report(report))
        self.assertIn("MAIORES TEMPOS OBSERVADOS", lines)
        self.assertIn("OCR", lines)
        self.assertIn("Producao: apresentacao", lines)

        complete_debug = format_network_debug_report({
            "event_id": "event-42", "source": "windows_xp",
            "inspection_timing": report,
        })
        self.assertIn("PERFIL DE TEMPO POR ETAPA", complete_debug)
        self.assertIn('"cycle_side_to_result_ms": 3500.0', complete_debug)

    def test_different_event_is_not_allowed_to_inherit_old_timings(self):
        panel = SimpleNamespace()
        begin_cycle(panel, "one", "Modo Teste")
        note_steps(panel, "SIDE", inference_ms=9500)
        self.assertEqual(snapshot(panel, "two"), {})
        begin_cycle(panel, "two", "Modo Sombra")
        result = snapshot(panel, "two")
        self.assertEqual(result["mode"], "Modo Sombra")
        self.assertEqual(result["lights"]["SIDE"]["steps"], {})
        self.assertIsNone(result["cycle_side_to_result_ms"])

    def test_missing_xp_timestamps_are_unknown_not_invented(self):
        panel = SimpleNamespace()
        begin_cycle(panel, "event", "Modo Teste")
        note_command(panel, "TOP", 100.0, 100.02, False)
        result = snapshot(panel, "event")
        self.assertIsNone(result["xp_fast_ack"])
        self.assertNotIn("to_ready_ms", result["lights"]["TOP"]["command"])
        self.assertEqual(
            result["lights"]["TOP"]["command"]["send_ms"], 20.0
        )
        self.assertFalse(result["lights"]["TOP"]["command"]["success"])

    def test_extraction_is_collected_before_inference_without_repeating_ocr(self):
        panel = SimpleNamespace()
        begin_cycle(panel, "event", "Modo Sombra")
        monitor = SimpleNamespace(last_external_timing={
            "layout_crop_ms": 45, "ocr_ms": 0.1, "extraction_ms": 45.1,
        })
        note_extraction(panel, "TOP", monitor)
        self.assertEqual(
            snapshot(panel, "event")["lights"]["TOP"]["steps"]["ocr_ms"], 0.1
        )
        note_extraction(panel, "TOP", SimpleNamespace(last_external_timing=None))
        self.assertEqual(
            snapshot(panel, "event")["lights"]["TOP"]["steps"]["ocr_ms"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
