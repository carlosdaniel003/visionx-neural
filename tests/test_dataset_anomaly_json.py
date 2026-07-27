import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.services.dataset_manager import DatasetManager


class DatasetAnomalyJSONTests(unittest.TestCase):
    def test_json_only_memory_persists_anomaly_without_images(self):
        reference = np.full((64, 80, 3), 100, dtype=np.uint8)
        test = reference.copy()
        cv2.rectangle(test, (42, 34), (58, 50), (20, 40, 135), -1)
        signature = build_anomaly_signature(reference, test, {}, {})
        analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "confidence": 0.88,
            "reason": "Teste",
            "detail": {
                "anomaly_signature": signature,
                "semantic_debug": {},
            },
        }

        old_anomaly_dir = settings.ANOMALY_DIR
        old_normal_dir = settings.NORMAL_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                settings.ANOMALY_DIR = root / "anomalia"
                settings.NORMAL_DIR = root / "nao_anomalia"

                json_path = DatasetManager.save_sample(
                    test,
                    "NG",
                    sample_image=reference,
                    aoi_info={
                        "board": "P1",
                        "parts": "R1",
                        "category": "Much Adhesive",
                    },
                    analysis=analysis,
                    save_images=False,
                    source="button",
                    ai_decision="NG",
                )

                payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
                png_files = list(root.rglob("*.png"))
        finally:
            settings.ANOMALY_DIR = old_anomaly_dir
            settings.NORMAL_DIR = old_normal_dir

        self.assertEqual(payload["schema"], "visionx.memory.v2")
        self.assertEqual(payload["storage"]["mode"], "json_only")
        self.assertFalse(payload["storage"]["images_required_for_knn"])
        self.assertEqual(payload["image_file"], "")
        self.assertEqual(len(payload["analysis"]["anomaly_memory"]["vector"]), 224)
        self.assertEqual(png_files, [])


if __name__ == "__main__":
    unittest.main()
