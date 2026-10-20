"""KNN de robustez com variações leves: rótulo só da pasta, holdout real."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.startup_regression.knn_archive_augmented_replay import (
    perturb_aoi_pair, replay_augmented_archive,
)
from src.services.startup_regression.knn_archive_augmented_replay_cli import (
    write_augmented_reports,
)


def sample(path, label, category, num):
    return {
        "path": f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/{path}",
        "status": "VALID_PNG",
        "expected_label": label,
        "category_hint": category,
        "lighting_mode": "SIDE",
        "event_id": None,
        "pixel_sha256": f"pixel-{num}",
        "file_sha256": f"file-{num}",
    }


def inventory(rows, *, conflicts=()):
    return {
        "schema": "visionx.archive_inventory.v1",
        "summary": {"png_count": len(rows)},
        "images": rows,
        "cross_label_conflicts": list(conflicts),
    }


def extract(label):
    reference = np.full((64, 64, 3), 50, dtype=np.uint8)
    test = reference.copy()
    if label == "OK":
        cv2.rectangle(test, (10, 10), (20, 20), (105, 105, 105), -1)
    else:
        cv2.rectangle(test, (36, 36), (53, 53), (210, 210, 210), -1)
    # OCR intencionalmente ERRADO: não muda a verdade da pasta.
    return reference, test, {"category": "INVERTIDO", "value": ""}


class AugmentedKNNReplayTests(unittest.TestCase):
    def setUp(self):
        self.images = [
            sample("2026-10-01_0900_DESLOCADO_SIDE.png", "OK", "DESLOCADO", 1),
            sample("2026-10-02_1000_FALTANDO_SIDE.png", "NG", "FALTANDO", 2),
            sample("2026-10-03_0900_EMBORCADO_SIDE.png", "OK", "EMBORCADO", 3),
            sample("2026-10-04_1000_MUITO_ADESIVO_SIDE.png", "NG", "MUITO ADESIVO", 4),
        ]

    def execute(self, variations=2, **kwargs):
        return replay_augmented_archive(
            Path("/fake-root"), inventory=inventory(
                self.images, conflicts=kwargs.pop("conflicts", ())
            ),
            extractor=extract,
            frame_loader=lambda _root, img: img["expected_label"],
            variations=variations, k=1, **kwargs,
        )

    def test_two_variants_are_deterministic_and_preserve_source(self):
        reference, test, _ = extract("OK")
        before_ref, before_test = reference.copy(), test.copy()
        a_ref, a_test, a_info = perturb_aoi_pair(reference, test, "path", 0)
        again_ref, again_test, _ = perturb_aoi_pair(reference, test, "path", 0)
        self.assertTrue(np.array_equal(reference, before_ref))
        self.assertTrue(np.array_equal(test, before_test))
        np.testing.assert_array_equal(a_ref, reference)
        np.testing.assert_array_equal(a_ref, again_ref)
        np.testing.assert_array_equal(a_test, again_test)
        self.assertFalse(np.array_equal(a_test, test))
        self.assertEqual(a_info["name"], "exposure_and_sensor_noise")
        b_ref, b_test, b_info = perturb_aoi_pair(reference, test, "path", 1)
        self.assertEqual(b_ref.shape, reference.shape)
        self.assertEqual(b_test.shape, test.shape)
        self.assertEqual(b_info["name"], "shared_one_pixel_shift_and_soft_focus")
        self.assertNotEqual(b_info["dx_both"], 0)
        self.assertNotEqual(b_info["dy_both"], 0)

    def test_green_aoi_overlay_is_not_randomized(self):
        ref, test, _ = extract("OK")
        cv2.rectangle(test, (3, 4), (58, 59), (0, 255, 0), 2)
        _a, altered, _details = perturb_aoi_pair(ref, test, "frame", 0)
        original_green = np.all(test == [0, 255, 0], axis=2)
        np.testing.assert_array_equal(altered[original_green], test[original_green])

    def test_category_agnostic_reference_and_no_self_leakage(self):
        report = self.execute()
        self.assertEqual(report["originals_decoded"], 4)
        self.assertEqual(report["expected_query_count"], 8)
        self.assertEqual(report["queries_completed"], 8)
        self.assertEqual(len(report["cases"]), 8)
        self.assertTrue(report["complete"])
        self.assertFalse(report["category_used_for_selection"])
        self.assertFalse(report["writes_memory_or_dataset"])
        self.assertFalse(report["is_production_approved"])
        self.assertEqual(report["results"]["total"], 8)
        self.assertEqual(report["geometry_experiment"]["results"]["total"], 8)
        self.assertEqual(report["geometry_experiment"]["consensus_candidate"]["total"], 8)
        for case in report["cases"]:
            self.assertEqual(case["status"], "PASSOU")
            self.assertIsNone(case["error"])
            self.assertEqual(case["prediction"], case["expected_label"])
            self.assertTrue(case["neighbors"])
            self.assertTrue(all(
                item["source_path"] != case["source_path"]
                for item in case["neighbors"]
            ))
            self.assertFalse(
                case["geometry_consensus_candidate"].get("auto_decision_authorized")
            )
        self.assertEqual(report["results"]["NG_as_OK"], 0)
        self.assertEqual(report["results"]["OK_as_NG"], 0)

    def test_incomplete_conflict_and_cancel_never_pass_target(self):
        bad = self.execute(conflicts=[{"paths": [self.images[0]["path"]]}])
        self.assertFalse(bad["complete"])
        self.assertIn(self.images[0]["path"], bad["original_errors"])
        self.assertFalse(bad["shadow_candidate"]["target_97pct_met"])
        self.assertFalse(
            bad["geometry_experiment"]["consensus_candidate"]["target_97pct_met"]
        )
        canceled = self.execute(cancelled=lambda: True)
        self.assertTrue(canceled["canceled"])
        self.assertFalse(canceled["complete"])
        self.assertEqual(canceled["expected_query_count"], 8)
        self.assertFalse(canceled["shadow_candidate"]["target_97pct_met"])

    def test_writer_only_writes_under_reports(self):
        report = self.execute(variations=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report["root"] = str(root)
            jp, tp = write_augmented_reports(
                report, root / "reports" / "startup_regression",
            )
            self.assertTrue(jp.is_file())
            self.assertIn("categoria AOI", tp.read_text("utf-8"))
            with self.assertRaisesRegex(ValueError, "reports"):
                write_augmented_reports(report, root / "public" / "ok_archive")

    def test_reject_invalid_variant_or_data(self):
        ref, test, _ = extract("OK")
        with self.assertRaises(ValueError):
            perturb_aoi_pair(ref, test, "name", 2)
        with self.assertRaises(ValueError):
            perturb_aoi_pair(ref, test[:, :-1], "name", 0)
        with self.assertRaises(ValueError):
            self.execute(variations=3)


if __name__ == "__main__":
    unittest.main()
