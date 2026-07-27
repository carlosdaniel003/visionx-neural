import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config.settings import settings
from src.core.experts.semantic_expert import SemanticExpert
from src.services.dataset_manager import DatasetManager


class DatasetSemanticJSONTests(unittest.TestCase):
    def test_dataset_json_persists_semantic_embeddings_and_reconstruction(self):
        reference = np.full((48, 48, 3), (60, 80, 100), dtype=np.uint8)
        query = reference.copy()
        query[26:46, 26:46] = (15, 25, 190)

        semantic_result = SemanticExpert().analyze(
            reference,
            query,
            aoi_info={"category": "Much Adhesive"},
        )
        analysis = {
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "reason": "Teste de persistência",
            "detail": {
                "query_embedding": [0.1, 0.2],
                **semantic_result,
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
                    query,
                    "NG",
                    aoi_info={"category": "Much Adhesive"},
                    analysis=analysis,
                    save_images=False,
                )
                payload = json.loads(
                    Path(json_path).read_text(encoding="utf-8")
                )
        finally:
            settings.ANOMALY_DIR = old_anomaly_dir
            settings.NORMAL_DIR = old_normal_dir

        semantic = payload["analysis"]["semantic"]
        self.assertEqual(semantic["schema"], "visionx.semantic.v2")
        self.assertEqual(len(semantic["reference_embedding"]), 128)
        self.assertEqual(len(semantic["query_embedding"]), 128)
        self.assertEqual(
            semantic["debug"]["spatial"]["grid_shape"],
            [4, 4],
        )
        self.assertEqual(
            len(semantic["debug"]["delta_vector"]),
            128,
        )


if __name__ == "__main__":
    unittest.main()
