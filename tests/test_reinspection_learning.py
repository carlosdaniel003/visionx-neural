"""Regressões: reinspeção intencional cria julgamento e treino independentes."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.services.dataset_manager import DatasetManager
from src.services.neural_online_learning import eligible_online_case


class ReinspectionLearningTests(unittest.TestCase):
    def test_same_ok_image_keeps_second_human_judgment_without_duplicate_pngs(self):
        ref = np.full((64, 80, 3), 110, dtype=np.uint8)
        test = ref.copy()
        test[22:36, 30:44] = (30, 80, 180)
        info = {"category": "FALTANDO", "lighting_mode": "SIDE",
                "board": "P1", "parts": "R2"}
        analysis = {
            "is_defect": True, "verdict": "DEFEITO REAL",
            "detail": {"anomaly_signature": build_anomaly_signature(
                ref, test, {}, info,
            )},
        }

        old_normal = settings.NORMAL_DIR
        old_anomaly = settings.ANOMALY_DIR
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings.NORMAL_DIR = root / "ok"
                settings.ANOMALY_DIR = root / "ng"
                args = dict(
                    ng_image=test, label="OK", sample_image=ref,
                    aoi_info=info, analysis=analysis, save_images=True,
                    source="button", lighting_mode="SIDE",
                )
                first = DatasetManager.save_sample(
                    **args, event_id="first-event",
                )
                second = DatasetManager.save_sample(
                    **args, event_id="second-event",
                    reinspection_of_judged_event=True,
                )
                self.assertTrue(first)
                self.assertTrue(second)
                self.assertNotEqual(first, second)
                records = sorted((root / "ok").rglob("*.json"))
                self.assertEqual(len(records), 2)
                revised = json.loads(
                    Path(second).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    revised["multilight"]["event_id"], "second-event"
                )
                self.assertTrue(
                    revised["decision"]["reinspection_of_judged_event"]
                )
                self.assertEqual(
                    revised["storage"]["duplicate_visual_of_json"], first
                )
                self.assertEqual(len(list(root.rglob("*.png"))), 2)
        finally:
            settings.NORMAL_DIR = old_normal
            settings.ANOMALY_DIR = old_anomaly

    def test_known_cnn_route_can_retrain_only_after_human_reinspection(self):
        task = {
            "label": "OK",
            "source": "button",
            "aoi_info": {"category": "FALTANDO"},
            "analysis": {"detail": {"recognition_route": "KNOWN_KNN"}},
        }
        self.assertFalse(eligible_online_case(task))
        task["reinspection_of_judged_event"] = True
        self.assertTrue(eligible_online_case(task))
        task["source"] = "production_auto"
        self.assertFalse(eligible_online_case(task))
        task["source"] = "button"
        task["aoi_info"]["category"] = "MUITO ADESIVO"
        self.assertFalse(eligible_online_case(task))


if __name__ == "__main__":
    unittest.main()
