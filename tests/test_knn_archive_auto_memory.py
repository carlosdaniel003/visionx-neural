"""Sincronização permanente de TODA imagem que cair em ok_archive/ng_archive."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Event, RLock
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.experts.knn_expert import KNNExpert
from src.services.knn_archive_auto_memory import (
    KNNArchiveAutoMemory, activate_archive_auto_memory,
    notify_archive_image_saved,
)


def synthetic_png(path: Path, label: str):
    image = np.zeros((64, 70, 3), dtype=np.uint8)
    image[0, 0] = (0, 0, 200 if label == "NG" else 80)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Falha de imagem teste")
    return image


def fake_extractor(frame):
    ref = np.zeros((42, 49, 3), dtype=np.uint8)
    test = ref.copy()
    if int(frame[0, 0, 2]) > 150:
        test[8:19, 9:29, :] = 210
    else:
        test[26:32, 25:35, :] = 65
    return ref, test, {
        "category": "FALTANDO", "lighting_mode": "SIDE",
        "board": "BOARD_1", "parts": "U2",
        "value": "0 <= 60 <= 80 FALTANDO",
    }


class AutoArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = PersistentArchiveKNN(self.root)
        self.events = []
        self.watcher = KNNArchiveAutoMemory(
            root=self.root, store=self.store, extractor=fake_extractor,
            poll_interval=.05, settle_seconds=.2,
            on_committed=self.events.append,
        )
        self.addCleanup(self.watcher.stop)
        self.ok_dir = self.root / "public" / "ok_archive"
        self.ng_dir = self.root / "public" / "ng_archive"
        self.ok_dir.mkdir(parents=True)
        self.ng_dir.mkdir(parents=True)

    def test_new_ok_and_ng_import_automatically_then_survive_deletion(self):
        # Capturas chegam em diferentes momentos; não depende do modo atual.
        a = self.ok_dir / "2026-10-13_0901_FALTANDO_SIDE.png"
        b = self.ng_dir / "2026-10-13_0902_FALTANDO_SIDE.png"
        synthetic_png(a, "OK")
        first = self.watcher.scan_once(now=1.0)
        self.assertEqual(first["imported"], 0)  # arquivo precisa estabilizar
        second = self.watcher.scan_once(now=1.3)
        self.assertEqual(second["imported"], 1)
        synthetic_png(b, "NG")
        self.watcher.scan_once(now=2.0)
        third = self.watcher.scan_once(now=2.3)
        self.assertEqual(third["imported"], 1)
        self.assertEqual(self.store.add([])["total"], 2)
        self.assertEqual(len(self.events), 2)
        a.unlink()
        b.unlink()
        saved = PersistentArchiveKNN(self.root)
        for label in ("OK", "NG"):
            ref, test, info = fake_extractor(np.array([[
                [0, 0, 200 if label == "NG" else 80]
            ]], dtype=np.uint8))
            self.assertEqual(saved.lookup(ref, test, info)["label"], label)
        self.watcher.scan_once(now=3.0)
        self.assertEqual(len(saved.records()), 2)

    def test_reopen_does_not_relearn_or_run_ocr_on_already_imported(self):
        path = self.ok_dir / "2026-10-13_0904_FALTANDO.png"
        synthetic_png(path, "OK")
        self.watcher.scan_once(now=1)
        self.watcher.scan_once(now=2)
        self.assertEqual(self.store.add([])["total"], 1)
        called = []
        second = KNNArchiveAutoMemory(
            self.root, extractor=lambda frame: called.append(frame),
            settle_seconds=0.0,
        )
        second.scan_once(now=3)
        result = second.scan_once(now=4)
        self.assertEqual(result["imported"], 0)
        self.assertEqual(called, [])
        self.assertEqual(len(PersistentArchiveKNN(self.root).records()), 1)

    def test_invalid_png_is_pending_then_fixed_and_imported(self):
        path = self.ng_dir / "2026-10-13_0910_FALTANDO.png"
        path.write_bytes(b"incomplete")
        self.watcher.scan_once(now=1)
        result = self.watcher.scan_once(now=2)
        self.assertEqual(result["errors"], 1)
        self.assertIn("ng_archive", str(self.watcher.status()["pending_or_failed"]))
        # O produtor finalmente terminou a gravação; nova versão é capturada.
        synthetic_png(path, "NG")
        self.watcher.scan_once(now=3)
        result = self.watcher.scan_once(now=4)
        self.assertEqual(result["imported"], 1)
        self.assertFalse(self.watcher.status()["pending_or_failed"])

    def test_current_ocr_category_only_archives_ng_and_ok(self):
        """OCR atual: value vazio e category válida não podem ser rejeitados."""
        cases = [
            (self.ng_dir / "2026-10-15_1540_FALTANDO_SIDE.png", "NG"),
            (self.ok_dir / "2026-10-15_1541_FALTANDO_SIDE.png", "OK"),
        ]
        for path, label in cases:
            synthetic_png(path, label)

        def current_ocr(frame):
            reference, test, info = fake_extractor(frame)
            info["value"] = ""
            return reference, test, info

        self.watcher._extractor = current_ocr
        self.watcher.scan_once(now=1.0)
        result = self.watcher.scan_once(now=2.0)
        self.assertEqual(result["imported"], 2)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(len(self.store.records()), 2)
        self.assertFalse(self.watcher.status()["pending_or_failed"])
        self.assertFalse(self.watcher._rejections_path.is_file())

    def test_old_false_rejections_are_ignored_after_contract_fix(self):
        path = self.ng_dir / "2026-10-15_1238_FALTANDO_SIDE.png"
        synthetic_png(path, "NG")
        old_path = self.root / "data" / "knn_archive_memory" / "archive_rejections_v1.json"
        old_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = path.stat()
        old_path.write_text(json.dumps({
            "schema": "visionx.knn_archive_rejections.v1",
            "files": {
                path.relative_to(self.root).as_posix(): {
                    "stamp": [stamp.st_size, stamp.st_mtime_ns],
                    "reason": "Categoria AOI 'UNKNOWN' não confere com nome 'FALTANDO'",
                },
            },
        }), encoding="utf-8")

        def current_ocr(frame):
            ref, test, info = fake_extractor(frame)
            info["value"] = ""
            return ref, test, info

        watcher = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=current_ocr,
            settle_seconds=0,
        )
        watcher.scan_once(now=1.0)
        self.assertEqual(watcher.scan_once(now=2.0)["imported"], 1)
        self.assertEqual(len(self.store.records()), 1)
        self.assertTrue(old_path.is_file())
        self.assertNotIn("UNKNOWN", str(watcher.status()["pending_or_failed"]))

    def test_unknown_category_is_cached_no_retries_or_terminal_spam(self):
        """A mesma imagem com OCR UNKNOWN não deve ser reprocessada para sempre."""
        path = self.ng_dir / "2026-10-15_1238_FALTANDO_SIDE.png"
        synthetic_png(path, "NG")
        calls = []

        def unknown_extractor(frame):
            calls.append(1)
            ref, test, info = fake_extractor(frame)
            info["category"] = "Unknown"
            info["value"] = ""
            return ref, test, info

        self.watcher._extractor = unknown_extractor
        self.watcher.scan_once(now=1.0)
        with self.assertLogs("src.services.knn_archive_auto_memory", level="WARNING") as logs:
            attempt = self.watcher.scan_once(now=2.0)
        self.assertEqual(attempt["errors"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("1 PNG(s)", logs.output[0])
        self.assertNotIn("2026-10-15", logs.output[0])
        self.assertFalse(self.store.records())

        # Backoff antigo tentaria 2, 4, 8, 16, 32 e 60 s.
        for tick in (4, 8, 16, 32, 64, 1024):
            self.assertEqual(self.watcher.scan_once(now=tick)["errors"], 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("UNKNOWN", str(self.watcher.status()["pending_or_failed"]).upper())
        self.assertTrue(self.watcher._rejections_path.is_file())

        # A rejeição também impede novo OCR de mesmo PNG após reiniciar.
        reopened = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=unknown_extractor,
            settle_seconds=0.0,
        )
        reopened.scan_once(now=1.0)
        reopened.scan_once(now=2.0)
        self.assertEqual(len(calls), 1)
        self.assertIn("UNKNOWN", str(reopened.status()["pending_or_failed"]).upper())

        # Mudou imagem/categoria OCR: liberar automaticamente a nova versão.
        reopened._extractor = fake_extractor
        stamp = path.stat()
        import os
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
        reopened.scan_once(now=3.0)
        self.assertEqual(reopened.scan_once(now=4.0)["imported"], 1)
        self.assertEqual(len(self.store.records()), 1)
        self.assertEqual(reopened._rejections, {})
        self.assertEqual(reopened.status()["pending_or_failed"], {})

    def test_corrupt_rejections_cache_cannot_block_learning(self):
        path = self.ok_dir / "2026-10-15_1400_FALTANDO_SIDE.png"
        synthetic_png(path, "OK")
        location = self.watcher._rejections_path
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text("{invalid json", encoding="utf-8")
        recovered = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=fake_extractor,
            settle_seconds=0.0,
        )
        recovered.scan_once(now=1.0)
        self.assertEqual(recovered.scan_once(now=2.0)["imported"], 1)
        self.assertEqual(len(self.store.records()), 1)

    def test_duplicate_source_is_idempotent_but_second_path_keeps_provenance(self):
        a = self.ok_dir / "2026-10-13_0911_FALTANDO.png"
        b = self.ok_dir / "2026-10-13_0912_FALTANDO.png"
        synthetic_png(a, "OK")
        self.watcher.scan_once(now=1)
        self.watcher.scan_once(now=2)
        synthetic_png(b, "OK")
        self.watcher.scan_once(now=3)
        result = self.watcher.scan_once(now=4)
        self.assertEqual(result["imported"], 0)
        entries = self.store.records()
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0]["sources"]), 2)
        self.assertTrue(self.store.path.with_suffix(".json.bak").is_file())

    def test_opposite_label_identical_images_becomes_conflict_not_ok(self):
        a = self.ok_dir / "2026-10-13_0911_FALTANDO.png"
        b = self.ng_dir / "2026-10-13_0911_FALTANDO.png"
        image = synthetic_png(a, "OK")
        b.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(b), image))
        self.watcher.scan_once(now=1)
        self.watcher.scan_once(now=2)
        record = self.store.records()[0]
        self.assertEqual(self.store.lookup_identity(record["identity"])["status"], "CONFLICT")

    def test_corrupt_snapshot_refuses_new_learning_without_overwrite(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text('{"schema":"not-compatible"}', encoding="utf-8")
        synthetic_png(self.ok_dir / "2026-10-13_0911_FALTANDO.png", "OK")
        out = self.watcher.scan_once(now=1)
        self.assertTrue(out["fatal"])
        self.assertIn("SNAPSHOT", self.watcher.status()["pending_or_failed"])
        self.assertEqual(self.store.path.read_text("utf-8"),
                         '{"schema":"not-compatible"}')

    def test_live_worker_activity_reports_only_sanitized_load(self):
        from src.services.knn_archive_auto_memory import (
            archive_auto_memory_activity,
        )
        started = Event()
        release = Event()

        def slow_pass():
            self.watcher._set_worker_phase("extracting_png")
            started.set()
            release.wait(timeout=2)
            return {"imported": 0}

        self.addCleanup(release.set)
        activate_archive_auto_memory(self.watcher)
        self.addCleanup(activate_archive_auto_memory, None)
        with patch.object(self.watcher, "scan_once", side_effect=slow_pass):
            self.watcher.start()
            self.assertTrue(started.wait(timeout=2))
            busy = archive_auto_memory_activity()
            self.assertTrue(busy["running"])
            self.assertTrue(busy["scan_active"])
            self.assertEqual(busy["phase"], "extracting_png")
            self.assertGreaterEqual(busy["busy_total_ms"], 0)
            self.assertNotIn("source_path", busy)
            self.assertNotIn("png_path", busy)
            release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                finished = archive_auto_memory_activity()
                if finished["scan_count"] >= 1 and not finished["scan_active"]:
                    break
                time.sleep(.01)
            self.assertGreaterEqual(finished["scan_count"], 1)
            self.assertGreaterEqual(finished["last_scan_ms"], 0)
            self.assertEqual(finished["phase"], "idle")
            self.watcher.stop(timeout=2)
        activate_archive_auto_memory(None)
        self.assertIsNone(archive_auto_memory_activity())

    def test_inspection_priority_defers_scan_and_imports_after_release(self):
        path = self.ng_dir / "2026-10-15_0900_FALTANDO_SIDE.png"
        synthetic_png(path, "NG")
        self.watcher.pause_for_inspection()
        paused = self.watcher.scan_once(now=1.0)
        self.assertTrue(paused["deferred_for_inspection"])
        self.assertEqual(self.store.records(), [])
        self.assertTrue(self.watcher.activity()["inspection_priority_active"])
        self.assertEqual(self.watcher.activity()["deferred_scans"], 1)
        self.watcher.resume_after_inspection()
        self.assertFalse(self.watcher.activity()["inspection_priority_active"])
        self.assertEqual(self.watcher.scan_once(now=2.0)["imported"], 0)
        self.assertEqual(self.watcher.scan_once(now=3.0)["imported"], 1)
        self.assertEqual(len(self.store.records()), 1)

    def test_pause_mid_scan_finishes_current_png_then_defers_next(self):
        first = self.ok_dir / "2026-10-15_0910_FALTANDO_SIDE.png"
        second = self.ng_dir / "2026-10-15_0911_FALTANDO_SIDE.png"
        synthetic_png(first, "OK")
        synthetic_png(second, "NG")
        self.assertEqual(self.watcher.scan_once(now=1.0)["pending"], 2)
        called = []

        def trigger_pause(frame):
            called.append(1)
            self.watcher.pause_for_inspection()
            return fake_extractor(frame)

        self.watcher._extractor = trigger_pause
        result = self.watcher.scan_once(now=2.0)
        self.assertEqual(len(called), 1)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(len(self.store.records()), 1)
        self.assertEqual(self.watcher.activity()["deferred_candidates"], 1)
        self.watcher._extractor = fake_extractor
        self.watcher.resume_after_inspection()
        result = self.watcher.scan_once(now=3.0)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(len(self.store.records()), 2)

    def test_background_thread_does_not_begin_scans_during_inspection(self):
        from src.services.knn_archive_auto_memory import (
            pause_archive_auto_memory_for_inspection,
            resume_archive_auto_memory_after_inspection,
        )
        scanned = Event()
        calls = []

        def track():
            calls.append(1)
            scanned.set()
            return {"imported": 0}

        activate_archive_auto_memory(self.watcher)
        self.addCleanup(activate_archive_auto_memory, None)
        # O teste desta função avalia pause/resume; grace tem teste próprio.
        self.watcher.post_inspection_quiet_seconds = 0.05
        pause_archive_auto_memory_for_inspection()
        try:
            with patch.object(self.watcher, "scan_once", side_effect=track):
                self.watcher.start()
                time.sleep(.18)
                self.assertEqual(calls, [])
                self.assertEqual(
                    self.watcher.activity()["phase"], "deferred_inspection"
                )
                resume_archive_auto_memory_after_inspection()
                self.assertTrue(scanned.wait(timeout=2.0))
        finally:
            self.watcher.stop()
        self.assertGreaterEqual(len(calls), 1)

    def test_post_inspection_grace_prevents_ocr_between_fast_cycles(self):
        """Liberação->nova SIDE em menos de 3s não abre scan pesado."""
        done = Event()
        entered = Event()
        scanned = []

        def scan():
            scanned.append(time.monotonic())
            entered.set()
            return {"imported": 0}

        watcher = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=fake_extractor,
            poll_interval=.02, post_inspection_quiet_seconds=.24,
            max_idle_deferral_seconds=.6,
        )
        try:
            watcher.pause_for_inspection()
            with patch.object(watcher, "scan_once", side_effect=scan):
                watcher.start()
                watcher.resume_after_inspection()
                self.assertTrue(watcher.activity()["idle_grace_active"])
                self.assertFalse(entered.wait(timeout=.09))
                self.assertEqual(scanned, [])
                self.assertGreater(watcher.activity()["idle_grace_deferrals"], 0)
                self.assertTrue(entered.wait(timeout=1.2))
                self.assertFalse(watcher.activity()["idle_grace_active"])
        finally:
            done.set()
            watcher.stop(timeout=2)

    def test_immediate_second_inspection_keeps_grace_and_archives_resume(self):
        path = self.ng_dir / "2026-10-15_1015_FALTANDO.png"
        synthetic_png(path, "NG")
        started = Event()

        def on_commit(_result):
            started.set()

        watcher = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=fake_extractor,
            poll_interval=.02, settle_seconds=.01, batch_size=2,
            post_inspection_quiet_seconds=.22,
            max_idle_deferral_seconds=.5,
            on_committed=on_commit,
        )
        try:
            watcher.pause_for_inspection()
            watcher.start()
            watcher.resume_after_inspection()
            self.assertFalse(started.wait(timeout=.07))
            watcher.pause_for_inspection()
            self.assertTrue(watcher.activity()["inspection_priority_active"])
            self.assertFalse(started.wait(timeout=.23))
            watcher.resume_after_inspection()
            self.assertTrue(started.wait(timeout=2.0))
            self.assertEqual(len(self.store.records()), 1)
        finally:
            watcher.stop(timeout=2)

    def test_repeated_gate_release_is_idempotent_and_timeout_bounded(self):
        watcher = KNNArchiveAutoMemory(
            self.root, store=self.store, extractor=fake_extractor,
            poll_interval=.02, post_inspection_quiet_seconds=.22,
            max_idle_deferral_seconds=.3,
        )
        watcher.pause_for_inspection()
        watcher.resume_after_inspection()
        initial = watcher._idle_grace_until
        time.sleep(.02)
        watcher.resume_after_inspection()
        self.assertEqual(watcher._idle_grace_until, initial)
        with watcher._lock:
            watcher._last_scan_started_at = time.monotonic() - .5
        self.assertFalse(watcher.activity()["idle_grace_active"])
        watcher.stop()

    def test_scandir_nested_pngs_are_sorted_and_do_not_follow_symlinks(self):
        """Enumera o mesmo conjunto seguro, sem resolver cada caminho."""
        from pathlib import Path

        a = self.ng_dir / "sub" / "Z-last.PNG"
        b = self.ok_dir / "sub" / "a-first.png"
        c = self.ok_dir / "root.png"
        for path in (a, b, c):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png candidate for enumeration only")
        (self.ok_dir / "ignored.jpeg").write_bytes(b"not png")
        discovered = self.watcher._list_files()
        self.assertEqual(
            discovered,
            sorted(
                [(a, "NG"), (b, "OK"), (c, "OK")],
                key=lambda pair: str(pair[0]).casefold(),
            ),
        )

        outside = self.root / "untrusted"
        outside.mkdir()
        (outside / "escape.png").write_bytes(b"not imported")
        linked_dir = self.ok_dir / "linked_dir"
        linked_png = self.ng_dir / "linked.png"
        try:
            linked_dir.symlink_to(outside, target_is_directory=True)
            linked_png.symlink_to(outside / "escape.png")
        except (OSError, NotImplementedError):
            # Criacao de links no Windows corporativo pode requerer direitos.
            pass
        else:
            self.assertEqual(self.watcher._list_files(), discovered)
        # A varredura nao avalia valores ou integra memoria.
        self.assertEqual(self.store.records(), [])

    def test_win32_background_priority_only_uses_current_thread(self):
        import ctypes
        from unittest.mock import MagicMock
        from src.services import knn_archive_auto_memory as service

        api = MagicMock()
        api.GetCurrentThread.return_value = 1234
        api.SetThreadPriority.return_value = 1
        with patch.object(service.sys, "platform", "win32"), patch.object(
            ctypes, "WinDLL", return_value=api, create=True,
        ):
            self.assertTrue(service._set_background_thread_mode(True))
            self.assertTrue(service._set_background_thread_mode(False))
        self.assertEqual(
            [call.args for call in api.SetThreadPriority.call_args_list],
            [(1234, 0x00010000), (1234, 0x00020000)],
        )
        self.assertEqual(api.GetCurrentThread.call_count, 2)
        api.SetThreadPriority.return_value = 0
        with patch.object(service.sys, "platform", "win32"), patch.object(
            ctypes, "WinDLL", return_value=api, create=True,
        ), patch.object(ctypes, "get_last_error", return_value=5, create=True):
            self.assertFalse(service._set_background_thread_mode(True))

    def test_priority_is_set_and_restored_inside_watcher_thread(self):
        from src.services import knn_archive_auto_memory as service

        entered = Event()
        release = Event()
        mode_calls = []

        def scan():
            entered.set()
            release.wait(timeout=2.0)
            return {"imported": 0}

        def mode(enable):
            mode_calls.append(enable)
            return True

        try:
            with patch.object(
                service, "_set_background_thread_mode", side_effect=mode,
            ), patch.object(self.watcher, "scan_once", side_effect=scan):
                self.watcher.start()
                self.assertTrue(entered.wait(timeout=2.0))
                self.assertTrue(
                    self.watcher.activity()["background_priority_applied"]
                )
                release.set()
                self.watcher.stop(timeout=2.0)
                self.assertFalse(
                    self.watcher.activity()["background_priority_applied"]
                )
        finally:
            release.set()
            self.watcher.stop(timeout=2.0)
        self.assertEqual(mode_calls, [True, False])

    def test_worker_thread_and_instant_notification(self):
        done = Event()
        self.watcher.on_committed = lambda _result: done.set()
        self.watcher.start()
        try:
            path = self.ok_dir / "2026-10-13_0930_FALTANDO.png"
            synthetic_png(path, "OK")
            activate_archive_auto_memory(self.watcher)
            notify_archive_image_saved(path)
            self.assertTrue(done.wait(timeout=4.0))
            self.assertTrue(self.watcher.status()["running"])
            self.assertEqual(len(self.store.records()), 1)
        finally:
            activate_archive_auto_memory(None)
            self.watcher.stop()

    def test_hot_refresh_ram_without_rescanning_old_json(self):
        entry_source = self.ok_dir / "2026-10-13_0911_FALTANDO.png"
        synthetic_png(entry_source, "OK")
        self.watcher.scan_once(now=1)
        self.watcher.scan_once(now=2)
        expert = object.__new__(KNNExpert)
        expert._memory_lock = RLock()
        expert.signatures_ok = [{"label": "OK", "mode": "legacy_image"}]
        expert.signatures_ng = []
        with patch("src.core.experts.knn_expert.PersistentArchiveKNN",
                   return_value=self.store):
            self.assertEqual(expert.refresh_archive_memory(), 1)
            self.assertEqual(expert.refresh_archive_memory(), 1)
        self.assertEqual(len(expert.signatures_ok), 2)
        self.assertEqual(
            sum(bool(r.get("archive_persistent")) for r in expert.signatures_ok), 1
        )
        self.assertEqual(len(expert.signatures_ng), 0)


if __name__ == "__main__":
    unittest.main()
