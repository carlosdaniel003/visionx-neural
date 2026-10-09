"""Contrato do adaptador de MEMÓRIA KNN: mesma consulta exata da produção."""
import unittest
from unittest.mock import Mock

import numpy as np

from src.services.startup_regression.knn_archive_predictor import (
    KNNArchivePredictor,
)


class MemoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.knn = object()
        self.index = Mock()
        self.reader = KNNArchivePredictor(knn=self.knn, index=self.index)
        self.img = np.zeros((32, 32, 3), dtype=np.uint8)
        self.info = {
            "board": "B1", "parts": "U2~5",
            "category": "MUITO ADESIVO",
            "value": "10 <= 2 <= 80 MUITO ADESIVO",
        }

    def test_known_ok_and_ng_are_returned_only_from_verified_record(self):
        for label, expected in (
            ("OK", "FALHA FALSA"), ("NG", "DEFEITO REAL")
        ):
            self.index.lookup.return_value = {
                "status": "KNOWN", "label": label,
                "source_json": "memory/confirmed.json", "matches": 1,
            }
            output = self.reader.inspect(self.img, self.img, "TOP", self.info)
            self.assertEqual(output["verdict"], expected)
            self.assertEqual(output["detail"]["memory_label"], label)
            self.assertTrue(output["detail"]["verified_exact_match"])
            args = self.index.lookup.call_args.args
            self.assertIs(args[0], self.knn)
            self.assertEqual(args[3]["lighting_mode"], "TOP")
            self.assertEqual(args[3]["category"], "MUITOADESIVO")

    def test_new_is_no_coverage_not_false_ok(self):
        self.index.lookup.return_value = {
            "status": "NEW", "reason": "par não cadastrado",
        }
        result = self.reader.inspect(self.img, self.img, "SIDE", self.info)
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(result["detail"]["memory_status"], "NEW")
        self.assertFalse(result["detail"]["verified_exact_match"])

    def test_conflict_is_not_auto_classified(self):
        self.index.lookup.return_value = {
            "status": "CONFLICT", "reason": "rótulos divergentes",
        }
        result = self.reader.inspect(self.img, self.img, "SIDE", self.info)
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")

    def test_incomplete_ocr_is_rejected_before_lookup(self):
        info = {**self.info, "parts": ""}
        with self.assertRaisesRegex(ValueError, "Board/Parts/Value"):
            self.reader.inspect(self.img, self.img, "SIDE", info)
        self.index.lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
