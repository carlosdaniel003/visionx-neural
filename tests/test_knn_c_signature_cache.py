"""Cache visual KNN C: sem pickle, SHA verificado e falha fechada."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.core.knn_c_live import LiveKNNStrategyC
from src.services.knn_c_signature_cache import read_cache, write_cache
from src.services.startup_regression.archive_inventory import inventory_archives
from src.ui.network_xp_debug import _knn_c_evidence_lines


def fake_pair(label):
    reference = np.full((64, 64, 3), 42, dtype=np.uint8)
    test = reference.copy()
    cv2.rectangle(test, (7, 10), (30, 35), (99, 99, 99), -1)
    if label == "NG":
        cv2.rectangle(test, (32, 37), (48, 54), (212, 212, 212), -1)
    return reference, test, {"category": "FALTANDO"}


def populate(root, labels=("OK", "OK", "OK", "NG", "NG", "NG")):
    for i, label in enumerate(labels):
        folder = root / "public" / ("ok_archive" if label == "OK" else "ng_archive")
        folder.mkdir(parents=True, exist_ok=True)
        screen = np.full((840, 1165, 3), i + 7, np.uint8)
        cv2.rectangle(screen, (40, 320), (170, 500), (100 + i, 80, 40), -1)
        cv2.rectangle(screen, (630, 320), (770, 500), (30, 70 + i, 150), -1)
        assert cv2.imwrite(str(folder / f"2026-10-23_0{i}24_FALTANDO.png"), screen)


class CacheResilienceTests(unittest.TestCase):
    def test_first_warmup_persists_cache_and_second_skips_all_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            calls = []
            def loader(_, item):
                calls.append(item["path"])
                return item["expected_label"]
            bank = LiveKNNStrategyC(root=root, extractor=fake_pair, frame_loader=loader)
            first = bank.warmup()
            self.assertEqual(first["status"], "READY")
            self.assertEqual(first["cached"], 0)
            self.assertEqual(len(calls), 6)
            self.assertTrue(first["cache_saved"])
            cached = read_cache(root)
            self.assertEqual(len(cached), 6)
            self.assertEqual(bank.progress()["indexed_by_label"], {"OK": 3, "NG": 3})
            def forbid_decode(_, item):
                raise AssertionError("cache should avoid PNG descriptor extraction")
            second = LiveKNNStrategyC(root=root, extractor=fake_pair,
                                      frame_loader=forbid_decode).warmup()
            self.assertEqual(second["status"], "READY")
            self.assertEqual(second["cached"], 6)
            self.assertEqual(second["excluded"], 0)

    def test_one_changed_png_invalidates_only_its_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            bank = LiveKNNStrategyC(root=root, extractor=fake_pair,
                                    frame_loader=lambda _, item: item["expected_label"])
            self.assertEqual(bank.warmup()["status"], "READY")
            path = next((root / "public" / "ok_archive").glob("*.png"))
            image = cv2.imread(str(path))
            image[20:100, 20:100] = (221, 19, 55)
            self.assertTrue(cv2.imwrite(str(path), image))
            used = []
            next_bank = LiveKNNStrategyC(
                root=root, extractor=fake_pair,
                frame_loader=lambda _, item: used.append(item["path"]) or item["expected_label"],
            )
            report = next_bank.warmup()
            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["cached"], 5)
            self.assertEqual(report["images"], 6)
            self.assertEqual(len(used), 1)

    def test_bad_png_does_not_destroy_good_cache_or_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            broken = root / "public/ok_archive/2026-10-23_0800_FALTANDO.png"
            broken.write_bytes(b"corrupt")
            bank = LiveKNNStrategyC(
                root=root, extractor=fake_pair,
                frame_loader=lambda _, item: item["expected_label"],
            )
            result = bank.warmup()
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["images"], 6)
            self.assertEqual(result["excluded"], 1)
            self.assertEqual(bank.progress()["failure_reasons"]["INVALID_PNG_OR_SHA"], 1)
            self.assertEqual(len(bank.progress()["sample_failures"]), 1)
            self.assertFalse(bank.diagnose(*fake_pair("OK")[:2], "SIDE")["auto_decision_authorized"])

    def test_corrupt_disk_cache_is_ignored_and_rebuilt_from_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            f = root / "reports/startup_regression/knn_c_signature_cache.json.gz"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"not-a-gzip-file")
            self.assertEqual(read_cache(root), {})
            result = LiveKNNStrategyC(
                root=root, extractor=fake_pair,
                frame_loader=lambda _, item: item["expected_label"],
            ).warmup()
            self.assertEqual(result["status"], "READY")
            self.assertTrue(result["cache_saved"])
            self.assertEqual(len(read_cache(root)), 6)

    def test_same_pixels_both_classes_quarantined_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            sample = next((root / "public/ok_archive").glob("*.png"))
            duplicate = root / "public/ng_archive/2026-10-23_0726_FALTANDO.png"
            duplicate.write_bytes(sample.read_bytes())
            bank = LiveKNNStrategyC(
                root=root, extractor=fake_pair,
                frame_loader=lambda _, item: item["expected_label"],
            )
            result = bank.warmup()
            self.assertEqual(result["status"], "READY")
            self.assertGreaterEqual(result["excluded"], 2)
            self.assertEqual(bank.progress()["failure_reasons"]["CROSS_LABEL_CONFLICT"], 2)
            self.assertFalse(any(
                row["source_path"] in (
                    sample.relative_to(root).as_posix(),
                    duplicate.relative_to(root).as_posix(),
                )
                for group in bank._pools.values() for row in group
            ))

    def test_debug_reports_progress_and_never_claims_loading_vote(self):
        data = {
            "bank_status": "LOADING", "status": "REVISAO",
            "reason": "MEMORIA_VISUAL_EM_PREPARACAO",
            "bank_progress": {
                "phase": "BUILD_SIGNATURES", "total": 1042,
                "processed": 253, "usable": 251, "excluded": 2,
                "cached": 225, "skipped_duplicates": 0,
                "elapsed_ms": 14032,
                "failure_reasons": {"INVALID_PNG_OR_SHA": 2},
                "sample_failures": [{"path": "public/ok_archive/broken.png",
                                     "reason": "INVALID_PNG_OR_SHA"}],
            },
        }
        info = "\n".join(_knn_c_evidence_lines(data))
        self.assertIn("253/1042", info)
        self.assertIn("251", info)
        self.assertIn("225", info)
        self.assertIn("INVALID_PNG_OR_SHA", info)
        self.assertIn("public/ok_archive/broken.png", info)
        self.assertIn("Ainda sem votos A/B", info)
        self.assertNotIn("Voto A visual: OK", info)


if __name__ == "__main__":
    unittest.main()
