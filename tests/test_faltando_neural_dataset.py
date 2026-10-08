"""Preparação do dataset FALTANDO: evidências locais nunca modificadas."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.faltando_neural_dataset import (
    _group_hint, prepare_faltando, AOIPairExtractor,
)


class FakeSignal:
    def __init__(self):
        self.listeners = []

    def connect(self, callback):
        self.listeners.append(callback)

    def disconnect(self, callback):
        self.listeners.remove(callback)

    def emit(self, *args):
        for callback in tuple(self.listeners):
            callback(*args)


class FakeMonitor:
    def __init__(self):
        self.layout_detected = FakeSignal()
        self.log_updated = FakeSignal()
        self._replay_no_debug = False
        self.last_capture_frame = None

    def process_external_image(self, frame):
        self.last_capture_frame = frame
        self.layout_detected.emit(
            frame[0:25, 0:30], frame[0:25, 30:60],
            {"board": "B", "parts": "R1", "category": "FALTANDO"},
        )
        self.log_updated.emit("SUCESSO")


class FaltandoDatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ok = self.root / "public" / "ok_archive"
        self.ng = self.root / "public" / "ng_archive"
        self.ok.mkdir(parents=True)
        self.ng.mkdir(parents=True)
        self.extractor = AOIPairExtractor(FakeMonitor())

    def put(self, folder, name, value):
        image = np.full((30, 60, 3), value, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        target = folder / name
        target.write_bytes(encoded.tobytes())
        return target

    def test_group_hint_handles_explicit_light_and_copy_without_linking(self):
        self.assertEqual(
            _group_hint("public/ok_archive/2026-10-08_0735_FALTANDO_TOP.png"),
            "2026-10-08_0735_FALTANDO",
        )
        self.assertEqual(
            _group_hint("public/ok_archive/2026-10-08_0735_FALTANDO_TOP_2.png"),
            "2026-10-08_0735_FALTANDO_COPY_2",
        )
        self.assertIsNone(_group_hint("old_FALTANDO.png"))

    def test_derivatives_no_source_changes_and_unlinked_triplets(self):
        ng = self.put(self.ng, "2026-10-01_07-36_FALTANDO.png", 44)
        sources = [ng]
        for i, mode in enumerate(("SIDE", "TOP", "MID")):
            sources.append(
                self.put(self.ok, f"2026-10-08_0735_FALTANDO_{mode}.png", 80+i)
            )
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        report, out = prepare_faltando(self.root, extractor=self.extractor)
        self.assertEqual(report["summary"]["total"], 4)
        self.assertEqual(report["summary"]["by_label"], {"NG": 1, "OK": 3})
        self.assertEqual(report["summary"]["extracted"], 4)
        self.assertEqual(report["summary"]["training_ready"], 0)
        self.assertEqual(report["summary"]["name_only_triplets"], 1)
        self.assertEqual(report["summary"]["verified_multilight_events"], 0)
        self.assertFalse(report["dataset_ready"])
        self.assertFalse(report["training_performed"])
        self.assertTrue((out / "manifest.json").is_file())
        self.assertTrue((out / "summary.txt").is_file())
        for item in report["samples"]:
            self.assertEqual(item["status"], "EXTRACTED_PENDING_REVIEW")
            self.assertFalse(item["training_ready"])
            for key in ("reference_path", "test_path"):
                self.assertTrue((out / item[key]).is_file())
            self.assertEqual(item["reference_size"], [30, 25])
            self.assertEqual(item["test_size"], [30, 25])
        self.assertEqual(
            before,
            {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        )

    def test_unreadable_png_never_becomes_training_candidate(self):
        (self.ng / "invalid_FALTANDO.png").write_bytes(b"not png")
        report, _ = prepare_faltando(self.root, extractor=self.extractor)
        self.assertEqual(report["summary"]["unusable"], 1)
        self.assertEqual(report["summary"]["training_ready"], 0)
        self.assertEqual(report["samples"][0]["status"], "INVALID_SOURCE")

    def test_failed_extraction_is_excluded_without_reusing_previous_pair(self):
        self.put(self.ng, "first_FALTANDO.png", 45)
        self.put(self.ng, "second_FALTANDO.png", 85)
        calls = [0]

        def intermittent(frame):
            calls[0] += 1
            if calls[0] == 2:
                raise ValueError("barras AOI não encontradas")
            return self.extractor(frame)

        report, out = prepare_faltando(self.root, extractor=intermittent)
        self.assertEqual(report["summary"]["extracted"], 1)
        self.assertEqual(report["summary"]["unusable"], 1)
        self.assertEqual(len(list((out / "pairs").glob("*/reference.png"))), 1)

    def test_never_write_inside_public_or_dataset(self):
        with self.assertRaises(ValueError):
            prepare_faltando(self.root, self.ng, extractor=self.extractor)
        with self.assertRaises(ValueError):
            prepare_faltando(
                self.root, self.root / "public" / "dataset" / "faltando",
                extractor=self.extractor,
            )
        self.assertEqual(list(self.ng.iterdir()), [])

    def test_report_is_serializable_without_binary_frames(self):
        self.put(self.ok, "old_FALTANDO.png", 76)
        report, out = prepare_faltando(self.root, extractor=self.extractor)
        saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["schema"], report["schema"])
        self.assertTrue(saved["samples"][0]["ocr_unverified"])


if __name__ == "__main__":
    unittest.main()
