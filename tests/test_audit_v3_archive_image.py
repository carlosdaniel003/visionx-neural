"""Auditoria V3 é exclusivamente diagnóstica, sem exceção para arquivo."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.scripts.audit_v3_archive_image import audit_image


class AuditV3ArchiveTests(unittest.TestCase):
    def test_archive_ok_misclassified_by_model_is_reported_not_overridden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "public/ok_archive/2026-10-02_1029_FALTANDO.png"
            source.parent.mkdir(parents=True)
            frame = np.full((500, 900, 3), 70, dtype=np.uint8)
            cv2.rectangle(frame, (80, 110), (820, 420), (125, 40, 10), -1)
            self.assertTrue(cv2.imwrite(str(source), frame))
            contents_before = source.read_bytes()

            def extractor(img):
                return img[100:330, 100:430], img[100:330, 450:780], {
                    "category": "FALTANDO",
                }

            class Predictor:
                def inspect(self, ref, test, mode):
                    self_last = mode
                    return {
                        "verdict": "DEFEITO REAL",
                        "detail": {
                            "cnn_v3_status": "INFERENCE_OK",
                            "cnn_v3_ng_score_uncalibrated": 1.0,
                            "cnn_v3_checkpoint_sha256": "a"*64,
                        },
                    }

            result = audit_image(
                root, source.relative_to(root),
                extractor=extractor, predictor=Predictor(),
            )
            self.assertEqual(result["archive_label"], "OK")
            self.assertFalse(result["archive_label_physical_quality_verified"])
            self.assertEqual(result["original_model_vs_folder"], "DISAGREES")
            self.assertEqual(result["light"], "SIDE")
            self.assertEqual(result["original"]["v3_ng_score_uncalibrated"], 1.0)
            self.assertEqual(source.read_bytes(), contents_before)

    def test_outside_archive_or_unsupported_name_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wrong = root / "tmp.png"
            self.assertTrue(cv2.imwrite(
                str(wrong), np.zeros((400, 800, 3), dtype=np.uint8),
            ))
            with self.assertRaisesRegex(ValueError, "pastas públicas"):
                audit_image(root, wrong.relative_to(root),
                            extractor=lambda _: (), predictor=object())


if __name__ == "__main__":
    unittest.main()
