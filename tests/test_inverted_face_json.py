import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.services.dataset_manager import DatasetManager


class InvertedFaceJSONTests(unittest.TestCase):
    def test_inverted_signature_is_persisted_without_images(self):
        reference = np.full((60, 80, 3), 35, dtype=np.uint8)
        test = reference.copy()
        cv2.line(reference, (40, 12), (40, 48), (220, 220, 220), 5)
        cv2.line(test, (18, 48), (60, 14), (220, 220, 220), 5)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[10:52, 16:64] = 255
        detail = {
            "inverted_score": 0.84,
            "inverted_classification": "ORIENTAÇÃO DIVERGENTE",
            "inverted_signature_strength": 0.77,
            "inverted_direct_similarity": 0.31,
            "inverted_feature_loss": 0.68,
            "inverted_extra_structure": 0.61,
            "inverted_topology_mismatch": 0.72,
            "inverted_orientation_mismatch": 0.81,
            "inverted_alternate_face_signal": 0.66,
            "inverted_changed_coverage": 0.42,
            "inverted_expected_angle": 90.0,
            "inverted_observed_angle": 37.5,
            "inverted_orientation_hist_reference": [0.0] * 12,
            "inverted_orientation_hist_test": [0.0] * 12,
            "inverted_best_transform": "rot180",
            "inverted_best_transform_similarity": 0.74,
            "inverted_transform_gain": 0.26,
            "inverted_edge_grid_reference": [[0.0] * 6 for _ in range(6)],
            "inverted_edge_grid_test": [[0.0] * 6 for _ in range(6)],
            "inverted_polarity_grid_reference": [[0.0] * 6 for _ in range(6)],
            "inverted_polarity_grid_test": [[0.0] * 6 for _ in range(6)],
            "inverted_roi_box": (0, 0, 80, 60),
            "diff_mask": mask,
            "semantic_debug": {},
        }
        detail["anomaly_signature"] = build_anomaly_signature(
            reference,
            test,
            detail,
            {"category": "INVERTIDO"},
            (0, 0, 80, 60),
        )
        analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "confidence": 0.88,
            "reason": "Orientação divergente",
            "detail": detail,
        }

        old_anomaly = settings.ANOMALY_DIR
        old_normal = settings.NORMAL_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                settings.ANOMALY_DIR = root / "anomalia"
                settings.NORMAL_DIR = root / "normal"
                path = DatasetManager.save_sample(
                    test,
                    "NG",
                    sample_image=reference,
                    aoi_info={"category": "INVERTIDO", "parts": "U1"},
                    analysis=analysis,
                    save_images=False,
                    source="button",
                    ai_decision="NG",
                )
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                png_files = list(root.rglob("*.png"))
        finally:
            settings.ANOMALY_DIR = old_anomaly
            settings.NORMAL_DIR = old_normal

        inverted = payload["analysis"]["engines"]["inverted"]
        self.assertAlmostEqual(inverted["score"], 0.84)
        self.assertEqual(inverted["classification"], "ORIENTAÇÃO DIVERGENTE")
        self.assertAlmostEqual(inverted["orientation_mismatch"], 0.81)
        self.assertEqual(inverted["best_transform"]["name"], "rot180")
        self.assertAlmostEqual(inverted["best_transform"]["gain"], 0.26)
        self.assertEqual(len(inverted["orientation"]["reference_histogram"]), 12)
        self.assertEqual(payload["storage"]["mode"], "json_only")
        self.assertEqual(png_files, [])


if __name__ == "__main__":
    unittest.main()
