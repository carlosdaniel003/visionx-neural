import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.services.dataset_manager import DatasetManager


class MissingComponentJSONTests(unittest.TestCase):
    def test_roi_expectation_metrics_are_persisted_without_images(self):
        reference = np.full((40, 60, 3), 80, dtype=np.uint8)
        test = reference.copy()
        test[10:30, 20:40] = 180
        detail = {
            "missing_score": 0.86,
            "missing_expectation_mode": "structure",
            "missing_classification": "COMPONENTE DESLOCADO",
            "missing_structure_loss": 0.72,
            "missing_extra_structure": 0.41,
            "missing_changed_coverage": 0.64,
            "missing_coverage": 0.64,
            "missing_appearance_loss": 0.58,
            "missing_background_exposure": 0.31,
            "missing_presence_retention": 0.28,
            "missing_direct_similarity": 0.24,
            "missing_best_similarity": 0.82,
            "missing_displacement_dx": 8.0,
            "missing_displacement_dy": -3.0,
            "missing_displacement_pixels": 8.54,
            "missing_displacement_pct": 0.12,
            "missing_reference_distinctness": 0.74,
            "semantic_debug": {},
        }
        detail["anomaly_signature"] = build_anomaly_signature(
            reference,
            test,
            detail,
            {"category": "FALTANDO"},
            (0, 0, 60, 40),
        )
        analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "confidence": 0.89,
            "reason": "Quebra da expectativa da ROI",
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
                    aoi_info={"category": "FALTANDO", "parts": "R1"},
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

        missing = payload["analysis"]["engines"]["missing"]
        self.assertAlmostEqual(missing["score"], 0.86)
        self.assertEqual(missing["expectation_mode"], "structure")
        self.assertEqual(missing["classification"], "COMPONENTE DESLOCADO")
        self.assertAlmostEqual(missing["structure_loss"], 0.72)
        self.assertAlmostEqual(missing["extra_structure"], 0.41)
        self.assertAlmostEqual(missing["coverage"], 0.64)
        self.assertAlmostEqual(missing["direct_similarity"], 0.24)
        self.assertAlmostEqual(missing["best_nearby_similarity"], 0.82)
        self.assertAlmostEqual(missing["displacement"]["dx"], 8.0)
        self.assertAlmostEqual(missing["displacement"]["pixels"], 8.54)
        self.assertEqual(payload["storage"]["mode"], "json_only")
        self.assertEqual(png_files, [])


if __name__ == "__main__":
    unittest.main()
