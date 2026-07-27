import json
import unittest

import cv2
import numpy as np

from src.core.anomaly_signature import (
    VECTOR_SIZE,
    build_anomaly_signature,
    compare_anomaly_signatures,
    valid_anomaly_signature,
)


class AnomalySignatureTests(unittest.TestCase):
    def test_signature_is_fixed_size_and_json_serializable(self):
        reference = np.full((80, 100, 3), 100, dtype=np.uint8)
        test = reference.copy()
        cv2.rectangle(test, (60, 45), (75, 60), (25, 45, 135), -1)
        signature = build_anomaly_signature(reference, test, {}, {})

        self.assertTrue(valid_anomaly_signature(signature))
        self.assertEqual(len(signature["vector"]), VECTOR_SIZE)
        self.assertEqual(len(signature["anomaly_map_8x8"]), 8)
        json.dumps(signature)

    def test_same_anomaly_is_more_similar_than_anomaly_in_other_location(self):
        reference = np.full((80, 100, 3), 100, dtype=np.uint8)
        test_a = reference.copy()
        test_b = reference.copy()
        cv2.rectangle(test_a, (60, 45), (75, 60), (25, 45, 135), -1)
        cv2.rectangle(test_b, (10, 10), (25, 25), (25, 45, 135), -1)

        signature_a = build_anomaly_signature(reference, test_a, {}, {})
        signature_same = build_anomaly_signature(reference, test_a, {}, {})
        signature_other = build_anomaly_signature(reference, test_b, {}, {})

        same_similarity, _ = compare_anomaly_signatures(signature_a, signature_same)
        other_similarity, _ = compare_anomaly_signatures(signature_a, signature_other)

        self.assertGreater(same_similarity, other_similarity)
        self.assertGreater(same_similarity - other_similarity, 0.08)


if __name__ == "__main__":
    unittest.main()
