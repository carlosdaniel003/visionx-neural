"""Regressões de contrato do validador CNN de acervo, sem hardware ou pesos."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.services.startup_regression.cnn_archive_validation import (
    MODEL_FALTANDO, MODEL_MEMORY, scope_for_model,
    validate_archive_cnns, write_cnn_report,
)


class FakePredictor:
    def __init__(self, name: str, bad: set[tuple[str, int]] | None = None):
        self.name = name
        self.calls = []
        self.bad = bad or set()

    def inspect(self, reference, test, mode: str, aoi_info=None):
        marker = int(test[0, 0, 0])
        self.calls.append((marker, mode))
        # Simula classe real (50-99 OK; >=100 NG); adulteração intencional
        # permite verificar que o relatório não mascara nenhum falso OK/NG.
        ng = marker >= 100
        if (mode, marker) in self.bad:
            ng = not ng
        verdict = "DEFEITO REAL" if ng else "FALHA FALSA"
        if self.name == MODEL_FALTANDO:
            return {
                "verdict": verdict,
                "detail": {
                    "cnn_v2_active": True,
                    "cnn_v2_experimental": True,
                    "cnn_v2_checkpoint_verified": True,
                    "cnn_v2_ng_score_uncalibrated": .97 if ng else .02,
                    "cnn_v2_checkpoint_sha256": "a" * 64,
                },
            }
        return {
            "verdict": verdict,
            "detail": {
                "model_kind": "knn_verified_exact",
                "verified_exact_match": True,
                "memory_status": "KNOWN",
                "memory_label": "NG" if ng else "OK",
                "memory_source_json": "memory/verified.json",
            },
        }


class CnnArchiveContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ok = self.root / "public" / "ok_archive"
        self.ng = self.root / "public" / "ng_archive"
        self.ok.mkdir(parents=True)
        self.ng.mkdir(parents=True)

    def png(self, label: str, category: str, light: str = "", marker=None):
        path = (self.ok if label == "OK" else self.ng)
        source_marker = (
            marker if marker is not None else (50 if label == "OK" else 150)
        )
        image = np.full((48, 64, 3), source_marker, dtype=np.uint8)
        suffix = ("_" + light) if light else ""
        filename = f"2026-10-09_0955_{category}{suffix}.png"
        target = path / filename
        ok, blob = cv2.imencode(".png", image)
        self.assertTrue(ok)
        target.write_bytes(blob.tobytes())
        return target

    def fake_extractor(self, image):
        marker = int(image[0, 0, 0])
        values = {
            52: "FALTANDO", 151: "FALTANDO",
            54: "INVERTIDO", 153: "EMBORCADO",
            56: "DESLOCADO", 58: "MUITO ADESIVO",
            55: "FALTANDO",
        }
        return image.copy(), image.copy(), {
            "board": "B1", "parts": "R5",
            "value": values.get(marker, "UNKNOWN"),
        }

    def corpus(self):
        self.png("OK", "FALTANDO", marker=52)
        self.png("NG", "FALTANDO", marker=151)
        self.png("OK", "INVERTIDO", marker=54)
        self.png("NG", "EMBORCADO", marker=153)
        self.png("OK", "DESLOCADO", "TOP", marker=56)
        self.png("OK", "MUITO_ADESIVO", marker=58)

    def test_category_contract_and_adhesive_exclusion(self):
        for category in ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO"):
            self.assertEqual(scope_for_model(MODEL_FALTANDO, category), "IN_SCOPE")
        self.assertEqual(
            scope_for_model(MODEL_FALTANDO, "MUITO ADESIVO"), "OUT_OF_SCOPE"
        )
        self.assertEqual(
            scope_for_model(MODEL_FALTANDO, "UNKNOWN"), "UNKNOWN"
        )
        self.assertEqual(
            scope_for_model(MODEL_MEMORY, "MUITO ADESIVO"), "IN_SCOPE"
        )
        self.assertEqual(scope_for_model(MODEL_MEMORY, "DESLOCADO"), "IN_SCOPE")

    def test_complete_corpus_requires_both_independent_cnns(self):
        self.corpus()
        faltando = FakePredictor(MODEL_FALTANDO)
        memory = FakePredictor(MODEL_MEMORY)
        report = validate_archive_cnns(
            self.root,
            extractor=self.fake_extractor,
            predictors={MODEL_FALTANDO: faltando, MODEL_MEMORY: memory},
        )
        self.assertEqual(report["full_archive_png_count"], 6)
        self.assertEqual(report["models"][MODEL_FALTANDO]["eligible"], 5)
        self.assertEqual(report["models"][MODEL_MEMORY]["eligible"], 6)
        self.assertTrue(report["cnn_and_knn_passed"])
        self.assertTrue(report["knn_used"])
        self.assertFalse(report["specialist_moe_used"])
        self.assertFalse(report["training_enabled"])
        self.assertFalse(report["production_blocking_enabled"])
        self.assertEqual(len(faltando.calls), 5)
        self.assertEqual(len(memory.calls), 6)
        self.assertIn((56, "TOP"), memory.calls)
        self.assertEqual(
            {row["model"] for row in report["cases"]}, {MODEL_FALTANDO, MODEL_MEMORY}
        )
        self.assertEqual(len(list(self.ok.iterdir())), 4)
        self.assertEqual(len(list(self.ng.iterdir())), 2)

    def test_missing_memory_predictor_never_counts_as_pass(self):
        self.corpus()
        report = validate_archive_cnns(
            self.root,
            extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: None,
            },
        )
        self.assertEqual(report["models"][MODEL_FALTANDO]["passed"], 5)
        self.assertEqual(report["models"][MODEL_MEMORY]["model_unavailable"], 6)
        self.assertFalse(report["cnn_and_knn_passed"])
        self.assertFalse(report["can_release_operational_startup"])
        self.assertFalse(report["knn_model_present"])

    def test_one_false_ok_ng_fails_entire_model_without_hiding_other_results(self):
        self.corpus()
        memory = FakePredictor(MODEL_MEMORY, bad={("SIDE", 151)})
        report = validate_archive_cnns(
            self.root,
            extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: memory,
            },
        )
        self.assertTrue(report["models"][MODEL_FALTANDO]["passed_all"])
        self.assertEqual(report["models"][MODEL_MEMORY]["regressions"], 1)
        self.assertFalse(report["cnn_and_knn_passed"])
        self.assertTrue(any(
            row["model"] == MODEL_MEMORY and row["expected_label"] == "NG"
            and row["verdict"] == "FALHA FALSA"
            for row in report["cases"]
        ))

    def test_memory_new_is_reported_without_coverage_not_as_false_ng(self):
        self.png("OK", "FALTANDO", marker=55)

        class UnknownMemory(FakePredictor):
            def inspect(self, reference, test, mode, info):
                return {
                    "verdict": "REVISÃO OBRIGATÓRIA",
                    "detail": {
                        "model_kind": "knn_verified_exact",
                        "verified_exact_match": False,
                        "memory_status": "NEW",
                        "reason": "Não há par exato humano",
                    },
                }

        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: UnknownMemory(MODEL_MEMORY),
            },
        )
        self.assertEqual(report["models"][MODEL_MEMORY]["without_memory_coverage"], 1)
        self.assertEqual(report["models"][MODEL_MEMORY]["regressions"], 0)
        self.assertFalse(report["cnn_and_knn_passed"])

    def test_review_never_counts_as_correct(self):
        self.png("OK", "FALTANDO", marker=55)

        class Review(FakePredictor):
            def inspect(self, *args):
                row = super().inspect(*args)
                row["verdict"] = "REVISÃO OBRIGATÓRIA"
                return row

        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: Review(MODEL_FALTANDO),
                MODEL_MEMORY: Review(MODEL_MEMORY),
            },
        )
        self.assertEqual(report["models"][MODEL_FALTANDO]["regressions"], 1)
        self.assertEqual(report["models"][MODEL_MEMORY]["invalid"], 1)
        self.assertFalse(report["cnn_and_knn_passed"])

    def test_incorrect_memory_identity_cannot_impersonate_a_cnn(self):
        self.png("OK", "FALTANDO", marker=55)

        class KNNImpostor(FakePredictor):
            def inspect(self, reference, test, mode):
                return {
                    "verdict": "FALHA FALSA",
                    "detail": {
                        "recognition_route": "KNOWN_KNN",
                        "model_kind": "cnn_memoria",
                        "checkpoint_verified": True,
                        "checkpoint_sha256": "0" * 64,
                    },
                }

        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: KNNImpostor(MODEL_MEMORY),
            },
        )
        self.assertEqual(report["models"][MODEL_MEMORY]["invalid"], 1)
        self.assertFalse(report["cnn_and_knn_passed"])

    def test_corrupted_png_invalidates_all_applicable_models(self):
        self.png("OK", "FALTANDO", marker=55).write_bytes(b"bad image")
        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: FakePredictor(MODEL_MEMORY),
            },
        )
        self.assertEqual(report["models"][MODEL_FALTANDO]["invalid"], 1)
        self.assertEqual(report["models"][MODEL_MEMORY]["invalid"], 1)

    def test_nonrecognized_category_blocks_v2_coverage(self):
        self.png("OK", "CATEGORY_NEW", marker=55)
        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: FakePredictor(MODEL_MEMORY),
            },
        )
        self.assertEqual(report["models"][MODEL_FALTANDO]["invalid"], 1)
        self.assertEqual(report["models"][MODEL_MEMORY]["passed"], 1)

    def test_reports_stay_outside_archive_and_do_not_write_new_photos(self):
        self.png("OK", "FALTANDO", marker=55)
        report = validate_archive_cnns(
            self.root, extractor=self.fake_extractor,
            predictors={
                MODEL_FALTANDO: FakePredictor(MODEL_FALTANDO),
                MODEL_MEMORY: FakePredictor(MODEL_MEMORY),
            },
        )
        json_path, txt_path = write_cnn_report(
            report, self.root / "reports" / "startup_regression"
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], report["schema"])
        self.assertIn("MEMORIA_KNN", txt_path.read_text(encoding="utf-8"))
        self.assertEqual(len(list(self.ok.glob("*.png"))), 1)
        with self.assertRaises(ValueError):
            write_cnn_report(report, self.ok)


if __name__ == "__main__":
    unittest.main()
