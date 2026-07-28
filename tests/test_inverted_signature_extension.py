import unittest

import numpy as np

import src.core.inverted_face_integration as inverted_integration
from src.core.inverted_signature_extension import (
    PHYSICS_NAMES,
    install_inverted_signature_extension,
)


class InvertedSignatureExtensionTests(unittest.TestCase):
    def test_inverted_category_replaces_physics_block_without_changing_size(self):
        install_inverted_signature_extension()
        reference = np.full((32, 48, 3), 40, dtype=np.uint8)
        test = reference.copy()
        test[8:24, 18:30] = 180
        detail = {
            "inverted_score": 0.82,
            "inverted_witness_loss": 0.71,
            "inverted_feature_loss": 0.66,
            "inverted_topology_mismatch": 0.54,
            "inverted_orientation_mismatch": 0.43,
            "inverted_alternate_face_signal": 0.58,
            "inverted_signature_strength": 0.74,
            "inverted_test_signature_strength": 0.41,
            "inverted_relocation_gain": 0.22,
            "inverted_relocation_dx": 6.0,
            "inverted_relocation_dy": -3.0,
            "inverted_roi_width": 48,
            "inverted_roi_height": 32,
            "inverted_direct_similarity": 0.29,
            "inverted_changed_coverage": 0.31,
            "inverted_extra_structure": 0.36,
            "inverted_transform_gain": 0.18,
            "inverted_witness_coverage": 0.24,
            "diff_mask": np.zeros((32, 48), dtype=np.uint8),
        }
        signature = inverted_integration.build_anomaly_signature(
            reference,
            test,
            detail,
            {"category": "INVERTIDO"},
            (0, 0, 48, 32),
        )
        self.assertEqual(signature["vector_size"], 224)
        self.assertEqual(signature["specialist"], "inverted_witness")
        self.assertEqual(tuple(signature["physics"].keys()), PHYSICS_NAMES)
        self.assertAlmostEqual(signature["physics"]["inverted_score"], 0.82)
        self.assertAlmostEqual(signature["physics"]["witness_loss"], 0.71)
        start, end = signature["ranges"]["physics"]
        self.assertEqual(end - start, 16)
        self.assertEqual(len(signature["vector"]), 224)
        self.assertAlmostEqual(signature["vector"][start], 0.82)

    def test_other_category_keeps_standard_physics(self):
        install_inverted_signature_extension()
        image = np.full((24, 24, 3), 70, dtype=np.uint8)
        signature = inverted_integration.build_anomaly_signature(
            image,
            image.copy(),
            {"adhesive_score": 0.33},
            {"category": "MUITO ADESIVO"},
            (0, 0, 24, 24),
        )
        self.assertNotIn("specialist", signature)
        self.assertIn("adhesive_score", signature["physics"])


if __name__ == "__main__":
    unittest.main()
