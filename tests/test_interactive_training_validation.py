"""Botão Treinamento: replay de OK/NG não mutante e cancelamento."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import unittest

import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN, make_record
from src.core.anomaly_signature import build_anomaly_signature
from src.services.startup_regression.interactive_training_validation import (
    SCHEMA, save_report, summarize, validate,
)


def make_pair(label):
    ref = np.zeros((48, 65, 3), dtype=np.uint8)
    ref[8:36, 13:42] = (65, 90, 50)
    test = ref.copy()
    if label == "NG":
        test[15:32, 21:35] = (1, 1, 1)
    return ref, test


def snapshot(root, label):
    ref, test = make_pair(label)
    store = PersistentArchiveKNN(root)
    entry = make_record(
        label=label, category="FALTANDO", lighting="SIDE",
        source_path=f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/"
                    f"2026-10-13_FALTANDO_{label}.png",
        source_sha256=("a" if label == "OK" else "b") * 64,
        reference=ref, test=test,
        anomaly_signature=build_anomaly_signature(
            ref, test, {}, {"category": "FALTANDO"}
        ),
    )
    store.add([entry])
    return store


def inventory(labels, extra=None):
    items = []
    for idx, label in enumerate(labels):
        category = "FALTANDO"
        items.append({
            "path": f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/"
                    f"{idx}_FALTANDO.png",
            "expected_label": label, "file_sha256": str(idx) * 64,
            "status": "VALID_PNG", "category_hint": category,
            "lighting_mode": "SIDE",
        })
    if extra:
        items.extend(extra)
    return {
        "schema": "visionx.archive_inventory.v1",
        "summary": {"png_count": len(items)},
        "images": items, "cross_label_conflicts": [],
    }


class MockCNN:
    def __init__(self, incorrect_ng=False):
        self.calls = []
        self.incorrect_ng = incorrect_ng

    def inspect(self, ref, test, light):
        original_ng = bool(np.count_nonzero(test[15:32, 21:35] == 1))
        self.calls.append(light)
        predicts_ng = original_ng and not self.incorrect_ng
        return {
            "verdict": "DEFEITO REAL" if predicts_ng else "FALHA FALSA",
            "detail": {
                "cnn_v2_active": True,
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_ng_score_uncalibrated": .999 if predicts_ng else .001,
                "cnn_v2_checkpoint_sha256": "f" * 64,
            },
        }


class ArchiveValidationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.memory = PersistentArchiveKNN(self.root)
        for label in ("OK", "NG"):
            snapshot(self.root, label)
        self.sample_inventory = inventory(["OK", "NG"])

    @staticmethod
    def extractor(label):
        ref, test = make_pair(label)
        return ref, test, {
            "value": "0 <= 55 <= 80 FALTANDO",
            "board": "PCB", "parts": "U2",
            "lighting_mode": "SIDE",
        }

    def run_validation(self, **kwargs):
        return validate(
            self.root, inventory=kwargs.pop("inventory", self.sample_inventory),
            frame_loader=kwargs.pop(
                "frame_loader", lambda root, item: item["expected_label"]
            ),
            extractor=kwargs.pop("extractor", self.extractor),
            memory=kwargs.pop("memory", self.memory),
            cnn=kwargs.pop("cnn", MockCNN()),
            **kwargs,
        )

    def test_correct_all_and_save_json_txt_without_updating_memory(self):
        previous = self.memory.path.read_bytes()
        signals = []
        result = self.run_validation(progress=signals.append)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(result["complete"])
        self.assertEqual(result["archive_png_count"], 2)
        self.assertEqual(result["processed_model_checks"], 4)
        for name in ("KNN_MEMORIA", "CNN_FALTANDO_V2"):
            value = result["models"][name]
            self.assertEqual(value["passed"], 2)
            self.assertEqual(value["by_label"]["OK"]["passed"], 1)
            self.assertEqual(value["by_label"]["NG"]["passed"], 1)
            self.assertTrue(value["target_98pct_per_label"])
        self.assertEqual([p["event"] for p in signals],
                         ["phase", "started"] + ["progress"] * 4 + ["summary"])
        json_path, txt_path = save_report(
            result, self.root / "reports" / "startup_regression"
        )
        self.assertTrue(json_path.is_file())
        self.assertTrue(txt_path.is_file())
        self.assertIn("NG: 1/1", txt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            json.loads(json_path.read_text(encoding="utf-8"))["models"],
            result["models"],
        )
        self.assertEqual(self.memory.path.read_bytes(), previous)

    def test_ng_as_ok_is_explicit_failure_and_not_98pct(self):
        result = self.run_validation(cnn=MockCNN(incorrect_ng=True))
        stats = result["models"]["CNN_FALTANDO_V2"]
        self.assertEqual(stats["ng_as_ok"], 1)
        self.assertEqual(stats["incorrect"], 1)
        self.assertEqual(stats["by_label"]["NG"]["accuracy"], 0.0)
        self.assertFalse(stats["target_98pct_per_label"])

    def test_no_coverage_knn_is_not_fake_correct(self):
        empty = PersistentArchiveKNN(
            self.root, path=self.root / "memory_empty.json"
        )
        result = self.run_validation(memory=empty)
        stats = result["models"]["KNN_MEMORIA"]
        self.assertEqual(stats["passed"], 0)
        self.assertEqual(stats["no_coverage"], 2)
        self.assertFalse(stats["target_98pct_per_label"])

    def test_cnn_adhesive_is_outside_scope_knn_still_checked(self):
        row = {
            "path": "public/ok_archive/3_MUITO_ADESIVO.png",
            "expected_label": "OK", "status": "VALID_PNG",
            "category_hint": "MUITO ADESIVO", "lighting_mode": "SIDE",
        }
        evidence = inventory(["OK", "NG"], extra=[row])
        # Same image for KNN; OCR category is adhesive but memory absence.
        def extract(label):
            ref, test, metadata = self.extractor("OK" if label == "ADESIVO" else label)
            if label == "ADESIVO":
                metadata["value"] = "0 <= 55 <= 80 MUITO ADESIVO"
            return ref, test, metadata

        result = self.run_validation(
            inventory=evidence,
            frame_loader=lambda root, item: (
                "ADESIVO" if "ADESIVO" in item["path"] else item["expected_label"]
            ),
            extractor=extract,
        )
        self.assertEqual(result["models"]["CNN_FALTANDO_V2"]["out_of_scope"], 1)
        self.assertEqual(result["models"]["CNN_FALTANDO_V2"]["eligible"], 2)
        self.assertEqual(result["models"]["KNN_MEMORIA"]["eligible"], 3)
        self.assertEqual(len(result["cases"]), 6)

    def test_cancel_after_first_verification_saves_partial_without_approval(self):
        gate = {"stop": False}
        seen = []
        def progress(packet):
            seen.append(packet)
            if packet.get("event") == "progress":
                gate["stop"] = True
        result = self.run_validation(
            progress=progress,
            cancelled=lambda: gate["stop"],
        )
        self.assertTrue(result["canceled"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["processed_model_checks"], 1)
        self.assertEqual(result["models"]["KNN_MEMORIA"]["not_evaluated"], 1)
        self.assertEqual(result["models"]["CNN_FALTANDO_V2"]["not_evaluated"], 2)
        self.assertFalse(result["models"]["KNN_MEMORIA"]["target_98pct_per_label"])
        json_path, txt_path = save_report(
            result, self.root / "reports" / "startup_regression"
        )
        self.assertTrue(json_path.exists())
        self.assertIn("CANCELADO - PARCIAL", txt_path.read_text(encoding="utf-8"))

    def test_invalid_png_remains_denominator(self):
        evidence = inventory(["OK", "NG"])
        evidence["images"][1]["status"] = "INVALID_PNG"
        result = self.run_validation(inventory=evidence)
        for name in ("KNN_MEMORIA", "CNN_FALTANDO_V2"):
            stats = result["models"][name]
            self.assertEqual(stats["invalid"], 1)
            self.assertFalse(stats["target_98pct_per_label"])


if __name__ == "__main__":
    unittest.main()
