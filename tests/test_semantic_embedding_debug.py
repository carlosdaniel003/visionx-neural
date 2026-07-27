import json
import unittest

import cv2
import numpy as np

from src.core.experts.semantic_expert import SemanticExpert


class SemanticEmbeddingDebugTests(unittest.TestCase):
    @staticmethod
    def _make_scene() -> np.ndarray:
        image = np.full((96, 96, 3), (55, 70, 85), dtype=np.uint8)
        cv2.rectangle(image, (16, 18), (80, 76), (100, 105, 110), -1)
        cv2.rectangle(image, (24, 28), (72, 66), (35, 35, 40), -1)
        cv2.line(image, (20, 82), (76, 82), (220, 220, 220), 3)
        return image

    def test_identical_images_produce_zero_delta_and_empty_reconstruction(self):
        reference = self._make_scene()
        result = SemanticExpert().analyze(
            reference,
            reference.copy(),
            aoi_info={"category": "Unknown"},
            aoi_epicenters=[(0, 0, 96, 96)],
        )

        self.assertEqual(result["semantic_loss"], 0.0)
        self.assertEqual(max(result["semantic_delta"]), 0.0)
        self.assertIsNone(
            result["semantic_debug"]["spatial"]["approximate_box"]
        )
        self.assertEqual(
            result["semantic_reconstruction_map"].shape,
            (96, 96),
        )

    def test_local_anomaly_is_reconstructed_in_the_corresponding_grid_region(self):
        reference = self._make_scene()
        query = reference.copy()
        cv2.rectangle(query, (68, 68), (95, 95), (20, 20, 220), -1)

        result = SemanticExpert().analyze(
            reference,
            query,
            aoi_info={"category": "Much Adhesive"},
            aoi_epicenters=[(0, 0, 96, 96)],
        )
        peak = result["semantic_debug"]["spatial"]["peak_cell"]

        self.assertGreaterEqual(peak["row"], 2)
        self.assertGreaterEqual(peak["column"], 2)
        self.assertGreater(peak["value"], 0.10)
        self.assertIsNotNone(
            result["semantic_debug"]["spatial"]["approximate_box"]
        )

    def test_debug_payload_describes_all_128_dimensions(self):
        reference = self._make_scene()
        query = reference.copy()
        query[10:32, 12:40] = (180, 30, 30)

        result = SemanticExpert().analyze(reference, query)
        debug = result["semantic_debug"]

        self.assertEqual(debug["schema"], "visionx.semantic.v2")
        self.assertEqual(debug["embedding_size"], 128)
        self.assertEqual(len(result["ref_emb"]), 128)
        self.assertEqual(len(result["query_emb"]), 128)
        self.assertEqual(len(debug["delta_vector"]), 128)
        self.assertEqual(len(debug["top_dimensions"]), 12)
        self.assertEqual(debug["spatial"]["grid_shape"], [4, 4])
        self.assertEqual(
            set(debug["groups"]),
            {
                "edge_density",
                "brightness",
                "hue_histogram",
                "saturation_histogram",
                "value_histogram",
            },
        )

    def test_debug_payload_is_json_serializable(self):
        reference = self._make_scene()
        query = reference.copy()
        query[50:75, 50:75] = (10, 20, 180)

        result = SemanticExpert().analyze(reference, query)
        encoded = json.dumps(result["semantic_debug"], ensure_ascii=False)

        self.assertIn("visionx.semantic.v2", encoded)
        self.assertIn("top_dimensions", encoded)


if __name__ == "__main__":
    unittest.main()
