"""Regressões da KNN após OCR reduzido à categoria (PR #33).

Testa separadamente: identidade exata no roteador de produção e vizinhos
aproximados APENAS em ferramenta offline.
"""
from __future__ import annotations

import unittest

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import PersistentArchiveKNN, make_operator_record
from src.services.startup_regression.knn_visual_neighbors import diagnose_visual_neighbors
from tests.test_verified_memory_router import MemoryFirstFixture


class LegacyWithoutOCRTests(MemoryFirstFixture):
    def setUp(self):
        super().setUp()
        self.category_only = {
            "category": "FALTANDO", "board": "", "parts": "",
            "value": "", "lighting_mode": "SIDE",
        }

    def test_exact_old_human_ng_recovered_with_category_only_ocr(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        outcome = self.inspect(system, info=self.category_only)
        self.assertEqual(outcome["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertEqual(outcome["verdict"], "DEFEITO REAL")
        self.assertEqual(system.specialist_calls, [])
        timing = outcome["detail"]["verified_memory_timing"]
        self.assertEqual(timing["legacy_lookup_mode"], "exact_visual_category_light")
        self.assertEqual(timing["matched_count"], 1)

    def test_no_similar_pixels_are_falsely_treated_as_known(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        altered = self.test.copy()
        altered[10, 10] ^= 1
        outcome = self.inspect(system, info=self.category_only, tst=altered)
        self.assertEqual(outcome["detail"]["recognition_route"], "NEW_CNN")

    def test_no_cross_category_or_light_recovery(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        for info in (
            dict(self.category_only, category="INVERTIDO"),
            dict(self.category_only, lighting_mode="TOP"),
        ):
            self.assertEqual(
                self.inspect(system, info=info)["detail"]["recognition_route"],
                "NEW_CNN",
            )

    def test_opposite_human_labels_same_visual_pair_force_review(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(self.make_record("OK"))
        system.experts["knn"].signatures_ng.append(
            self.make_record("NG", info=dict(self.info, board="PCB-999"))
        )
        outcome = self.inspect(system, info=self.category_only)
        self.assertEqual(outcome["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(outcome["detail"]["recognition_route"], "MEMORY_CONFLICT")

    def test_automatic_ok_and_invalid_png_cannot_approve(self):
        system = self.create_router()
        system.experts["knn"].signatures_ok.append(
            self.make_record("OK", source="production_auto")
        )
        self.assertEqual(
            self.inspect(system, info=self.category_only)["detail"]["recognition_route"],
            "NEW_CNN",
        )
        system.experts["knn"].signatures_ok = [self.make_record("OK")]
        self.assertEqual(
            self.inspect(system, info=self.category_only)["detail"]["recognition_route"],
            "KNOWN_KNN",
        )
        system.experts["knn"].signatures_ok[0]["part"] = "INVALID"
        system.reload_memory()
        self.assertEqual(
            self.inspect(system, info=self.category_only)["detail"]["recognition_route"],
            "NEW_CNN",
        )

    def test_partial_old_metadata_does_not_enable_wildcard(self):
        system = self.create_router()
        system.experts["knn"].signatures_ng.append(self.make_record("NG"))
        incomplete = dict(self.category_only, board="PCB-123")
        self.assertEqual(
            self.inspect(system, info=incomplete)["detail"]["recognition_route"],
            "NEW_CNN",
        )

    def test_known_persistent_record_stays_independent_of_old_ocr(self):
        store = PersistentArchiveKNN(self.root)
        signature = build_anomaly_signature(
            self.reference, self.test, {}, self.category_only
        )
        store.add([make_operator_record(
            label="NG", source="button", event_id="human1",
            category="FALTANDO", lighting="SIDE",
            reference=self.reference, test=self.test,
            anomaly_signature=signature, aoi_info=self.category_only,
        )])
        result = store.lookup(self.reference, self.test, self.category_only)
        self.assertEqual(result["status"], "KNOWN")
        self.assertEqual(result["label"], "NG")


class VisualNeighborsDiagnosticTests(MemoryFirstFixture):
    def test_similar_ng_can_be_seen_without_becoming_known(self):
        signature = build_anomaly_signature(
            self.reference, self.test, {}, self.info
        )
        records_ng = [{
            "mode": "anomaly", "label": "NG",
            "category": "FALTANDO", "lighting_mode": "SIDE",
            "anomaly_signature": signature, "path": "ng/known.json",
            "archive_persistent": True,
        }]
        alternate = self.test.copy()
        alternate[40, 55] ^= 1
        report = diagnose_visual_neighbors(
            self.reference, alternate,
            category="FALTANDO", lighting="SIDE",
            records_ok=[], records_ng=records_ng,
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["decision_authorized"])
        self.assertIsNone(report["recognition_route"])
        self.assertEqual(report["neighbors_by_label"]["NG"]["candidates"], 1)
        self.assertEqual(report["neighbors_by_label"]["OK"]["candidates"], 0)
        self.assertGreater(report["neighbors_by_label"]["NG"]["nearest"][0]["similarity"], 0)

    def test_labels_and_lighting_are_not_mixed(self):
        signature = build_anomaly_signature(self.reference, self.test, {}, self.info)
        record = {
            "mode": "anomaly", "category": "FALTANDO",
            "lighting_mode": "TOP", "anomaly_signature": signature,
            "path": "wrong_light.json",
        }
        result = diagnose_visual_neighbors(
            self.reference, self.test,
            category="FALTANDO", lighting="SIDE",
            records_ok=[record], records_ng=[],
        )
        self.assertEqual(result["neighbors_by_label"]["OK"]["candidates"], 0)
        with self.assertRaises(ValueError):
            diagnose_visual_neighbors(
                self.reference, self.test, category="Unknown",
                lighting="SIDE", records_ok=[], records_ng=[],
            )


if __name__ == "__main__":
    unittest.main()
