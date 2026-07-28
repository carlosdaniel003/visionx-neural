import unittest

import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.knn_expert import KNNExpert
from src.core.strict_category_memory import (
    canonical_memory_category,
    install_strict_category_memory,
)


def signature(category: str, value: int = 150):
    reference = np.full((28, 36, 3), 45, dtype=np.uint8)
    test = reference.copy()
    test[8:20, 12:25] = value
    detail = {
        "semantic_delta": [0.0] * 128,
        "diff_mask": (np.any(test != reference, axis=2).astype(np.uint8) * 255),
        "silk_error_pct": 0.42,
        "semantic_loss": 0.51,
    }
    return build_anomaly_signature(
        reference,
        test,
        detail,
        {"category": category},
        (0, 0, 36, 28),
    )


def record(category: str, label: str, anomaly_signature: dict, mode="anomaly"):
    return {
        "category": category,
        "part": "U1",
        "path": f"{category}_{label}.json",
        "label": label,
        "mode": mode,
        "anomaly_signature": anomaly_signature if mode == "anomaly" else None,
        "sig": np.ones(16, dtype=np.float32) if mode == "legacy_image" else None,
    }


class StrictCategoryMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_strict_category_memory(KNNExpert)

    def setUp(self):
        self.expert = KNNExpert.__new__(KNNExpert)
        self.expert.net = None
        self.expert.signatures_ok = []
        self.expert.signatures_ng = []

    def test_inverted_uses_only_inverted_jsons(self):
        query = signature("INVERTIDO")
        self.expert.signatures_ng = [
            record("INVERTIDO", "NG", query),
        ]
        self.expert.signatures_ok = [
            # Similaridade também perfeita, mas categoria incompatível.
            record("MUITOADESIVO", "OK", query),
            record("FALTANDO", "OK", query),
        ]
        result = self.expert.analyze(
            None,
            None,
            aoi_info={"category": "INVERTIDO", "parts": "U1"},
            anomaly_signature=query,
        )
        self.assertTrue(result["has_memory"])
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["vote_defect"], 1.0)
        self.assertEqual(result["memory_category"], "INVERTIDO")
        self.assertEqual(result["memory_candidate_count"], 1)
        self.assertTrue(result["memory_filter_strict"])
        self.assertEqual(result["memory_scope"], "categoria")

    def test_absent_category_memory_does_not_fallback_to_global(self):
        query = signature("INVERTIDO")
        self.expert.signatures_ng = [
            record("MUITOADESIVO", "NG", query),
            record("FALTANDO", "NG", query),
        ]
        result = self.expert.analyze(
            None,
            None,
            aoi_info={"category": "INVERTIDO"},
            anomaly_signature=query,
        )
        self.assertFalse(result["has_memory"])
        self.assertEqual(result["n_neighbors"], 0)
        self.assertEqual(result["memory_candidate_count"], 0)
        self.assertEqual(result["memory_category"], "INVERTIDO")
        self.assertIn("Nenhum JSON", result["memory_reason"])

    def test_legacy_image_is_ignored_even_in_same_category(self):
        query = signature("FALTANDO")
        self.expert.signatures_ng = [
            record("FALTANDO", "NG", query, mode="legacy_image"),
        ]
        result = self.expert.analyze(
            None,
            np.full((30, 30, 3), 120, dtype=np.uint8),
            aoi_info={"category": "FALTANDO"},
            anomaly_signature=query,
        )
        self.assertFalse(result["has_memory"])
        self.assertEqual(result["memory_mode"], "anomaly")
        self.assertEqual(result["memory_candidate_count"], 0)
        self.assertIsNone(self.expert.net)

    def test_aliases_resolve_to_same_category(self):
        self.assertEqual(canonical_memory_category("Much Adhesive"), "MUITOADESIVO")
        self.assertEqual(canonical_memory_category("MUITO ADESIVO"), "MUITOADESIVO")
        self.assertEqual(canonical_memory_category("Reverse"), "INVERTIDO")
        self.assertEqual(canonical_memory_category("Missing"), "FALTANDO")


if __name__ == "__main__":
    unittest.main()
