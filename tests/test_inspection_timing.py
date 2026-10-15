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
            fast_ack=True, previous_gate_release_at=base - 0.75,
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
        self.assertEqual(
            report["previous_gate_release_to_side_ms"], 750.0
        )
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

    def test_receiver_backlog_release_after_reinforced_stability(self):
        # Este teste roda no CI de telemetria XP em pull_request.
        import numpy as np
        from src.services.network_receiver import NetworkReceiver

        receiver = NetworkReceiver(port=0)
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        signature = receiver._frame_signature(image)
        for i in range(5):
            at = 100.0 + i * 0.18
            receiver._stage_latest_candidate(
                image, signature, "192.168.0.1", received_at=at,
            )
            defer = receiver._defer_candidate_for_pending_socket(
                pending=True, received_at=at,
            )
            if i < 4:
                self.assertTrue(defer)
            else:
                self.assertFalse(defer)
        self.assertGreaterEqual(receiver._candidate_stable_frames, 5)
        self.assertIsNotNone(receiver._candidate_snapshot())
        changed = image.copy()
        changed[70:140, 90:160] = 255
        changed_sig = receiver._frame_signature(changed)
        self.assertEqual(receiver._stage_latest_candidate(
            changed, changed_sig, "192.168.0.1", received_at=101.0,
        ), 1)
        self.assertIsNone(receiver._candidate_snapshot())

    def test_stream_backlog_telemetry_keeps_count_and_boolean_separate(self):
        panel = SimpleNamespace()
        begin_cycle(panel, "backlog-7", "Modo Teste", transport={
            "stable_frame_count": 5,
            "stability_ms": 720.0,
            "last_stable_pair_gap_ms": 180.0,
            "backlog_deferrals": 3,
            "backlog_release_with_pending": True,
        })
        report = snapshot(panel, "backlog-7")
        network = report["lights"]["SIDE"]["network"]
        self.assertEqual(network["stable_frame_count"], 5)
        self.assertEqual(network["backlog_deferrals"], 3)
        self.assertIs(network["backlog_release_with_pending"], True)
        lines = "\\n".join(format_report(report))
        self.assertIn("Frames adiados por conexoes XP pendentes: 3", lines)
        self.assertIn(
            "Entrega XP com fila ainda ocupada apos estabilidade reforcada: True",
            lines,
        )
        begin_cycle(panel, "new", "Modo Teste")
        self.assertEqual(snapshot(panel, "new")["lights"]["SIDE"]["network"], {})

    def test_network_cycle_defers_background_ocr_until_gate_release(self):
        from unittest.mock import patch
        from src.services.knn_archive_auto_memory import (
            KNNArchiveAutoMemory, activate_archive_auto_memory,
        )
        from src.ui.network_image_cycle_gate import _release_cycle

        watcher = KNNArchiveAutoMemory(poll_interval=0.1)
        activate_archive_auto_memory(watcher)
        try:
            panel = SimpleNamespace(
                capture_cycle_source="network", network_receiver=None,
            )
            begin_cycle(panel, "xp-side-event", "Modo Teste")
            self.assertTrue(watcher.activity()["inspection_priority_active"])
            # Finalização multilight não encerra o gate: em modo Teste
            # o operador ainda precisa salvar OK/NG ou descartar.
            finish_cycle(panel)
            self.assertTrue(watcher.activity()["inspection_priority_active"])
            with patch("src.ui.network_image_cycle_gate._safe_sync"):
                _release_cycle(panel)
            self.assertFalse(watcher.activity()["inspection_priority_active"])
            self.assertIsNotNone(
                panel._inspection_previous_gate_release_at
            )
        finally:
            activate_archive_auto_memory(None)
            watcher.stop(timeout=1)

    def test_live_archive_worker_timing_runs_without_processing_images(self):
        from threading import Event
        from unittest.mock import patch
        from src.services.knn_archive_auto_memory import KNNArchiveAutoMemory

        started = Event()
        release = Event()
        watcher = KNNArchiveAutoMemory(poll_interval=0.1)

        def dummy_scan():
            watcher._set_worker_phase("extracting_png")
            started.set()
            release.wait(timeout=2.0)
            return {"imported": 0}

        try:
            with patch.object(watcher, "scan_once", side_effect=dummy_scan):
                watcher.start()
                self.assertTrue(started.wait(timeout=2.0))
                before = watcher.activity()
                self.assertTrue(before["scan_active"])
                self.assertEqual(before["phase"], "extracting_png")
                self.assertGreaterEqual(before["busy_total_ms"], 0.0)
                release.set()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    after = watcher.activity()
                    if not after["scan_active"] and after["scan_count"] >= 1:
                        break
                    time.sleep(0.01)
                self.assertEqual(after["phase"], "idle")
                self.assertGreaterEqual(after["scan_count"], 1)
                self.assertGreaterEqual(after["busy_total_ms"], 0.0)
        finally:
            release.set()
            watcher.stop(timeout=2.0)

    def test_archive_worker_overlap_is_per_light_and_non_authoritative(self):
        from unittest.mock import patch
        states = [
            {
                "running": True, "scan_active": True,
                "phase": "hashing_png", "busy_total_ms": 200.0,
                "scan_count": 10, "last_scan_ms": 36.0,
                "imports_this_session": 100,
            },
            {
                "running": True, "scan_active": False,
                "phase": "idle", "busy_total_ms": 315.0,
                "scan_count": 11, "last_scan_ms": 115.0,
                "imports_this_session": 101,
            },
        ]
        panel = SimpleNamespace()
        with patch(
            "src.services.inspection_timing._archive_activity",
            side_effect=states,
        ):
            begin_cycle(panel, "worker-overlap", "Modo Teste")
            note_steps(panel, "SIDE", inference_ms=66)
        data = snapshot(panel, "worker-overlap")
        worker = data["lights"]["SIDE"]["archive_worker"]
        self.assertEqual(worker["busy_overlap_ms"], 115.0)
        self.assertEqual(worker["at_frame"]["phase"], "hashing_png")
        self.assertEqual(worker["after_step"]["phase"], "idle")
        self.assertEqual(data["lights"]["SIDE"]["steps"]["inference_ms"], 66)
        self.assertEqual(data["lights"]["TOP"]["archive_worker"], {})
        report = "\n".join(format_report(data))
        self.assertIn(
            "Observador KNN: atividade simultanea estimada: 115.0 ms",
            report,
        )
        self.assertIn("varreduras no periodo: 1", report)
        begin_cycle(panel, "worker-next", "Modo Teste")
        self.assertEqual(
            snapshot(panel, "worker-next")["lights"]["SIDE"]["archive_worker"],
            {},
        )

    def test_crop_io_phase_breakdown_is_reported_without_double_counting(self):
        panel = SimpleNamespace()
        begin_cycle(panel, "crop-1", "Modo Teste")
        monitor = SimpleNamespace(last_external_timing={
            "layout_crop_ms": 2482.68,
            "bars_detect_ms": 61.0,
            "sample_crop_ms": 1340.0,
            "test_crop_ms": 1081.68,
            "debug_png_write_count": 12,
            "debug_png_write_ms": 1750.0,
            "ocr_ms": 4422.25,
            "extraction_ms": 6904.93,
        })
        note_extraction(panel, "SIDE", monitor)
        data = snapshot(panel, "crop-1")["lights"]["SIDE"]["steps"]
        self.assertEqual(data["debug_png_write_count"], 12)
        self.assertEqual(data["debug_png_write_ms"], 1750.0)
        self.assertEqual(data["layout_crop_ms"], 2482.68)
        report = "\n".join(format_report(snapshot(panel, "crop-1")))
        self.assertIn("PNGs de debug gravados: 12", report)
        self.assertIn("Gravacao de PNGs de debug", report)
        self.assertIn("Copia inicial / deteccao de barras", report)
        begin_cycle(panel, "crop-2", "Modo Teste")
        self.assertEqual(
            snapshot(panel, "crop-2")["lights"]["SIDE"]["steps"], {}
        )

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

    def test_ocr_profile_and_xp_frame_diagnostics_are_isolated_per_event(self):
        panel = SimpleNamespace()
        begin_cycle(
            panel, "event-one", "Modo Teste",
            transport={
                "payload_at": 100.0, "ready_at": 100.01,
                "stability_ms": 420.0,
                "last_stable_pair_gap_ms": 420.0,
                "stable_frame_count": 2,
            },
        )
        monitor = SimpleNamespace(last_external_timing={
            "ocr_ms": 750.0,
            "ocr_profile": {
                "schema": "visionx.aoi_ocr_phases.v1",
                "route": "general_blue",
                "parts_reason": "valid_general_reference",
                "general_retry": False,
                "call_count": 1,
                "tesseract_calls_ms": 600.0,
                "calls": [{"phase": "general_blue", "ms": 600.0, "status": "ok"}],
            },
        })
        note_extraction(panel, "SIDE", monitor)
        note_frame(panel, "TOP", {
            "stability_ms": 390.0,
            "last_stable_pair_gap_ms": 390.0,
            "stable_frame_count": 2,
            "pre_settle_frames": 3,
            "configured_settle_ms": 350.0,
        })
        report = snapshot(panel, "event-one")
        self.assertEqual(
            report["lights"]["SIDE"]["ocr_profile"]["call_count"], 1
        )
        self.assertEqual(
            report["lights"]["TOP"]["network"]["pre_settle_frames"], 3
        )
        display = "\n".join(format_report(report))
        self.assertIn("OCR subprocessos Tesseract: 1", display)
        self.assertIn("Tesseract general_blue: 600.0 ms", display)
        self.assertIn("Frames consecutivos estaveis confirmados: 2", display)
        self.assertIn("Frames antes da janela de troca de luz: 3", display)

        # O próximo evento não herda OCR nem temporização da peça anterior.
        begin_cycle(panel, "event-two", "Modo Teste")
        other = snapshot(panel, "event-two")
        self.assertEqual(other["lights"]["SIDE"]["ocr_profile"], {})
        self.assertEqual(other["lights"]["TOP"]["network"], {})

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
