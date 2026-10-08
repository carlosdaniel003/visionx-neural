"""Revisão visual FALTANDO: persistência auditável e operação Qt offline."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.services.faltando_neural_qualification import (
    QualificationStore, latest_manifest, visual_similarities,
)


class QualificationFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "reports" / "faltando_neural" / "run_20261008T151051"
        self.run_dir.mkdir(parents=True)
        (self.root / "public" / "ok_archive").mkdir(parents=True)
        (self.root / "public" / "ng_archive").mkdir(parents=True)
        self.samples = []
        self.group_paths = {}
        self._add("old_FALTANDO.png", "NG", "SIDE", 21)
        for index, mode in enumerate(("SIDE", "TOP", "MID")):
            path = self._add(
                f"2026-10-08_0735_FALTANDO_{mode}.png", "OK", mode, 70 + index
            )
            self.group_paths[mode] = path
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest = {
            "schema": "visionx.faltando_neural_preparation.v1",
            "root": str(self.root),
            "samples": self.samples,
            "name_only_triplet_candidates": [{
                "id": "2026-10-08_0735_FALTANDO",
                "label": "OK",
                "paths": self.group_paths,
                "status": "NAME_ONLY_NEEDS_HUMAN_QUALIFICATION",
            }],
        }
        self._write_manifest()

    def _write_png(self, path, value):
        img = np.zeros((44, 68, 3), dtype=np.uint8)
        img[:] = (value, value+2, value+4)
        img[10:20, 15:35] = (2, 80, 190)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())

    def _add(self, name, label, mode, value):
        path = self.root / "public" / ("ok_archive" if label == "OK" else "ng_archive") / name
        self._write_png(path, value)
        fingerprint = sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(self.root).as_posix()
        base = Path("pairs") / f"{label.lower()}_{fingerprint}"
        ref, test = base / "reference.png", base / "test.png"
        self._write_png(self.run_dir / ref, value)
        self._write_png(self.run_dir / test, value+1)
        self.samples.append({
            "source_path": relative,
            "source_sha256": fingerprint,
            "expected_label_from_archive": label,
            "lighting_mode": mode,
            "lighting_source": (
                "LEGACY_DEFAULT" if name == "old_FALTANDO.png"
                else "EXPLICIT_SUFFIX"
            ),
            "status": "EXTRACTED_PENDING_REVIEW",
            "reference_path": ref.as_posix(),
            "test_path": test.as_posix(),
            "ocr_observed": {"board": "B1", "parts": "C2", "value": "FALTANDO"},
        })
        return relative

    def _write_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False), encoding="utf-8"
        )

    def store(self):
        return QualificationStore(self.manifest_path)


class QualificationCoreTests(QualificationFixtures):
    def test_source_integrity_and_atomic_autosave_resume(self):
        store = self.store()
        source = self.root / self.samples[0]["source_path"]
        before = sha256(source.read_bytes()).hexdigest()
        store.review_case(self.samples[0]["source_path"], "CONFIRMED_NG", "Ausente.")
        saved = json.loads(store.output.read_text(encoding="utf-8"))
        self.assertFalse(saved["automated_training_allowed"])
        self.assertEqual(len(saved["case_reviews"]), 1)
        self.assertEqual(saved["case_reviews"][self.samples[0]["source_path"]]["notes"], "Ausente.")
        self.assertEqual(sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual(self.store().case_status(self.samples[0]["source_path"]), "CONFIRMED_NG")
        self.assertFalse((self.run_dir / "qualification.json.tmp").exists())

    def test_confirmation_versus_archive_and_rejection(self):
        store = self.store()
        path = self.samples[0]["source_path"]
        store.review_case(path, "CONFIRMED_OK", "Rótulo da pasta incorreto")
        self.assertTrue(store.case_reviews[path]["label_conflict"])
        self.assertEqual(store.case_reviews[path]["archive_label"], "NG")
        store.review_case(path, "REJECTED")
        self.assertEqual(store.case_status(path), "REJECTED")
        self.assertFalse(store.case_reviews[path]["label_conflict"])

    def test_group_requires_explicit_three_individual_confirmations(self):
        store = self.store()
        name = "2026-10-08_0735_FALTANDO"
        with self.assertRaisesRegex(ValueError, "Confirme cada par"):
            store.review_group(name, "CONFIRMED_VISUAL_ASSOCIATION")
        for mode in ("SIDE", "TOP", "MID"):
            store.review_case(self.group_paths[mode], "CONFIRMED_OK")
        store.review_group(name, "CONFIRMED_VISUAL_ASSOCIATION", "Conferi as luzes")
        self.assertEqual(store.summary()["confirmed_groups"], 1)
        self.assertIsNone(store.group_reviews[name]["original_event_id"])
        self.assertEqual(
            self.store().group_status(name), "CONFIRMED_VISUAL_ASSOCIATION"
        )
        store.review_case(self.group_paths["TOP"], "CONFIRMED_NG")
        self.assertEqual(store.group_status(name), "PENDING")
        with self.assertRaises(ValueError):
            store.review_group(name, "CONFIRMED_VISUAL_ASSOCIATION")
        store.review_group(name, "REJECTED", "Imagens não são a mesma peça")
        self.assertEqual(store.group_status(name), "REJECTED")

    def test_changed_source_blocks_approval(self):
        store = self.store()
        path = self.root / self.samples[0]["source_path"]
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "modificado"):
            store.review_case(self.samples[0]["source_path"], "CONFIRMED_NG")
        self.assertFalse(store.output.exists())

    def test_missing_pair_or_escape_rejected(self):
        store = self.store()
        rel = self.samples[0]["source_path"]
        (self.run_dir / self.samples[0]["test_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "indisponível"):
            store.review_case(rel, "CONFIRMED_NG")
        self.samples[0]["test_path"] = "../outside.png"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "fora de pairs"):
            self.store()

    def test_similarity_is_only_suggestion_not_label(self):
        store = self.store()
        pairs = visual_similarities(store.samples, store.run_dir)
        self.assertIn(self.group_paths["SIDE"], pairs)
        self.assertTrue(
            any(
                x["other_path"] == self.samples[0]["source_path"]
                for x in pairs[self.group_paths["SIDE"]]
            )
        )
        self.assertEqual(store.summary()["pending"], 4)
        self.assertEqual(store.summary()["confirmed_groups"], 0)

    def test_manifest_outside_reports_rejected(self):
        moved = self.root / "public" / "ok_archive" / "manifest.json"
        moved.write_bytes(self.manifest_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "fora de reports"):
            QualificationStore(moved)

    def test_latest_manifest_uses_lexical_run_order(self):
        self.assertEqual(latest_manifest(self.root).resolve(), self.manifest_path.resolve())


class QualificationWindowTests(QualificationFixtures):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(["qualification-test"])

    def test_window_shows_case_and_group_and_saves_click(self):
        from PyQt6.QtCore import Qt
        from src.ui.faltando_neural_review import FaltandoReviewWindow

        store = self.store()
        window = FaltandoReviewWindow(store)
        self.addCleanup(window.close)
        self.assertEqual(window.items.count(), 5)
        self.assertEqual(window.selected[0], "GROUP")
        self.assertIn("TRINCA CANDIDATA", window.heading.text())
        window.items.setCurrentRow(1)
        self.assertEqual(window.selected[0], "CASE")
        window.confirm_box.setChecked(True)
        window.notes.setText("Inspeção de teste")
        window.ok.click()
        self.assertEqual(store.summary()["confirmed_ok"], 1)
        self.assertIn("qualification.json", str(store.output))
        self.assertEqual(store.summary()["pending"], 3)
        window.filter.setText("FALTANDO_SIDE.png")
        self.assertGreaterEqual(
            sum(not window.items.item(i).isHidden()
                for i in range(window.items.count())), 1
        )


if __name__ == "__main__":
    unittest.main()
