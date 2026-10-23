"""Retomada rápida da KNN C: SHA-256 impede cache obsoleto."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.core.knn_c_live import LiveKNNStrategyC
from src.services.knn_c_signature_cache import read_cache
from src.services.knn_c_fast_inventory import inventory_for_knn_c
from src.services.startup_regression.archive_inventory import _inspect_png


def make_pngs(root):
    for label, offset in (("OK", 0), ("NG", 10)):
        folder = root / "public" / ("ok_archive" if label == "OK" else "ng_archive")
        folder.mkdir(parents=True, exist_ok=True)
        for n in range(2):
            frame = np.full((840, 1165, 3), 31 + offset + n, np.uint8)
            cv2.rectangle(frame, (45, 320), (500, 650),
                          (45+n, 133+offset, 35+n), -1)
            assert cv2.imwrite(str(folder / f"2026-10-23_08{n}0_FALTANDO.png"),
                               frame)


def extractor(label):
    a = np.full((64, 64, 3), 33, np.uint8)
    b = a.copy()
    cv2.rectangle(b, (9, 12), (36, 42), (120, 120, 120), -1)
    if label == "NG":
        cv2.rectangle(b, (29, 33), (49, 52), (215, 215, 215), -1)
    return a, b, {"category": "LABEL_IS_NOT_AOI_CATEGORY"}


class FastInventoryTests(unittest.TestCase):
    def prepare(self, root):
        return LiveKNNStrategyC(
            root=root, extractor=extractor,
            frame_loader=lambda _root, row: row["expected_label"],
        ).warmup()

    def test_valid_sha_skips_all_expensive_png_decoding_on_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_pngs(root)
            self.assertEqual(self.prepare(root)["status"], "READY")
            cache = read_cache(root)
            self.assertEqual(len(cache), 4)
            with patch("src.services.knn_c_fast_inventory._inspect_png",
                       side_effect=AssertionError("unchanged PNG re-decoded")):
                inv = inventory_for_knn_c(root, cache)
            self.assertEqual(inv["reused_pixel_hashes_from_sha_verified_cache"], 4)
            self.assertEqual(len(inv["images"]), 4)
            progress_bank = LiveKNNStrategyC(
                root=root, extractor=extractor,
                frame_loader=lambda _root, row: (_ for _ in ()).throw(
                    AssertionError("unchanged PNG re-extracted")),
            )
            self.assertEqual(progress_bank.warmup()["status"], "READY")
            progress = progress_bank.progress()
            self.assertEqual(progress["inventory_reused_pixel_hashes"], 4)
            self.assertEqual(progress["cached"], 4)
            self.assertIn("inventory_elapsed_ms", progress)
            self.assertIn("build_signatures_elapsed_ms", progress)

    def test_modified_png_is_revalidated_even_if_mtime_restored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_pngs(root)
            self.prepare(root)
            cache = read_cache(root)
            path = next((root / "public/ok_archive").glob("*.png"))
            before = path.stat()
            image = cv2.imread(str(path))
            image[24:110, 30:130] = (128, 255, 35)
            self.assertTrue(cv2.imwrite(str(path), image))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with patch("src.services.knn_c_fast_inventory._inspect_png",
                       wraps=_inspect_png) as reader:
                inv = inventory_for_knn_c(root, cache)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(inv["reused_pixel_hashes_from_sha_verified_cache"], 3)
            current = next(r for r in inv["images"]
                           if r["path"] == path.relative_to(root).as_posix())
            self.assertNotEqual(
                current["pixel_sha256"],
                cache[current["path"]]["source_pixel_sha256"]
            )

    def test_cross_label_identical_content_is_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_pngs(root)
            self.prepare(root)
            sample = next((root / "public/ok_archive").glob("*.png"))
            dest = root / "public/ng_archive/2026-10-23_0859_FALTANDO.png"
            dest.write_bytes(sample.read_bytes())
            inv = inventory_for_knn_c(root, read_cache(root))
            self.assertEqual(len(inv["cross_label_conflicts"]), 1)
            self.assertEqual(len(inv["cross_label_conflicts"][0]["paths"]), 2)
            bank = LiveKNNStrategyC(root=root, inventory=inv, extractor=extractor,
                         frame_loader=lambda _root, row: row["expected_label"])
            result = bank.warmup()
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["excluded"], 2)

    def test_relabelled_path_never_reuses_other_class_signature(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_pngs(root)
            self.prepare(root)
            cache = read_cache(root)
            path = next((root / "public/ok_archive").glob("*.png"))
            target = root / "public/ng_archive" / "2026-10-23_0812_FALTANDO.png"
            path.rename(target)
            inv = inventory_for_knn_c(root, cache)
            self.assertEqual(inv["reused_pixel_hashes_from_sha_verified_cache"], 3)
            moved = next(r for r in inv["images"]
                         if r["path"] == target.relative_to(root).as_posix())
            self.assertEqual(moved["expected_label"], "NG")

    def test_diagnostic_cli_help_starts_without_contacting_xp(self):
        r = subprocess.run(
            [sys.executable, "-m", "src.services.knn_c_preload_cli", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, timeout=35,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        self.assertIn("--root", r.stdout)


if __name__ == "__main__":
    unittest.main()
