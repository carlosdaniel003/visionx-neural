"""Testes do sincronizador de diagnóstico, sem retreinamento ou migração."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_reconciler import (
    reconcile_archive, scan_memory_dataset, write_reconciliation_report,
)


class ArchiveReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for sub in (
            "public/ok_archive", "public/ng_archive",
            "public/dataset/nao_anomalia", "public/dataset/anomalia",
        ):
            (self.root / sub).mkdir(parents=True)
        self.reference = np.full((40, 50, 3), 22, dtype=np.uint8)

    @staticmethod
    def _write_png(path, array):
        success, contents = cv2.imencode(".png", array)
        if not success:
            raise AssertionError("PNG sintético não foi codificado")
        path.write_bytes(contents.tobytes())

    def archive(self, name, label, marker):
        out = (
            self.root / "public" /
            ("ok_archive" if label == "OK" else "ng_archive") / name
        )
        self._write_png(out, np.full((40, 50, 3), marker, dtype=np.uint8))
        return out

    def record(
        self, name, label, marker, *, source="button", signature=True,
        paired=True, screenshot=False, board="B1", category="FALTANDO",
        override_label=None,
    ):
        folder = "nao_anomalia" if label == "OK" else "anomalia"
        parent = self.root / "public" / "dataset" / folder
        test = np.full((40, 50, 3), marker, dtype=np.uint8)
        if paired:
            self._write_png(parent / f"{name}_reference.png", self.reference)
            self._write_png(parent / f"{name}_test.png", test)
        if screenshot:
            self._write_png(parent / f"{name}_source.png", test)
        data = {
            "schema": "visionx.memory.v3",
            "label": override_label or label,
            "decision": {"operator_label": label, "source": source},
            "aoi_info": {
                "board": board, "parts": "U2~5",
                "category": category, "value": "FALTANDO",
                "lighting_mode": "SIDE",
            },
            "storage": {
                "reference_image_file": f"{name}_reference.png" if paired else "",
                "test_image_file": f"{name}_test.png" if paired else "",
                "source_image_file": f"{name}_source.png" if screenshot else "",
                "test_image_fingerprint": image_fingerprint(test),
            },
            "analysis": {
                "anomaly_memory": {"vector": [0.0] * 224} if signature else {}
            },
        }
        path = parent / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def extract(self, frame):
        return self.reference, frame, {
            "board": "B1", "parts": "U2~5",
            "value": "0 <= 10 <= 20 FALTANDO",
        }

    # Make the fixture value match the OCR value used by the extractor.
    def _sync_values(self):
        for folder in ("nao_anomalia", "anomalia"):
            for path in (self.root / "public" / "dataset" / folder).glob("*.json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                data["aoi_info"]["value"] = "0 <= 10 <= 20 FALTANDO"
                path.write_text(json.dumps(data), encoding="utf-8")

    def test_finds_verified_exact_pair_and_no_training_or_new_files(self):
        original = self.archive("2026-10-09_0800_FALTANDO.png", "OK", 60)
        self.record("memory1", "OK", 60)
        self._sync_values()
        originals = {
            p: p.read_bytes()
            for p in self.root.rglob("*") if p.is_file()
        }
        with patch(
            "src.core.experts.knn_expert.KNNExpert.__init__",
            side_effect=AssertionError("Não instanciar KNN de produção"),
        ):
            report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {"PAR_VERIFICADO": 1})
        self.assertEqual(report["memory_record_reasons"], {"VERIFICADO_KNN": 1})
        self.assertFalse(report["writes_to_dataset"])
        self.assertFalse(report["startup_gate_enabled"])
        self.assertFalse(report["trains_models"])
        self.assertEqual(originals, {
            p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()
        })
        paths = write_reconciliation_report(
            report, self.root / "reports" / "startup_regression"
        )
        self.assertTrue(all(p.is_file() for p in paths))
        with self.assertRaises(ValueError):
            write_reconciliation_report(report, self.root / "public" / "dataset")

    def test_reports_source_png_with_missing_audit_pair_but_never_approves(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 63)
        self.record("record", "OK", 63, paired=False, screenshot=True)
        self._sync_values()
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {
            "SCREENSHOT_FONTE_ENCONTRADO": 1
        })
        row = report["cases"][0]
        self.assertEqual(row["candidates"][0]["eligibility"], "PAR_PNG_AUDITORIA_AUSENTE")
        self.assertEqual(report["memory_record_reasons"], {
            "PAR_PNG_AUDITORIA_AUSENTE": 1,
        })

    def test_legacy_signature_cannot_be_upgraded_to_verified(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 64)
        self.record("legacy", "OK", 64, signature=False)
        self._sync_values()
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {
            "REGISTRO_COMPATIVEL_INELEGIVEL": 1
        })
        self.assertEqual(report["cases"][0]["candidates"][0]["eligibility"],
                         "SEM_ASSINATURA_ANOMALIA")

    def test_human_label_conflict_is_not_an_approved_match(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "NG", 72)
        self.record("record", "OK", 72)
        self._sync_values()
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {
            "ROTULO_DIVERGENTE": 1
        })

    def test_automatic_labels_never_count_as_verified(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 66)
        self.record("auto", "OK", 66, source="auto")
        self._sync_values()
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["memory_record_reasons"],
                         {"SEM_CONFIRMACAO_HUMANA": 1})
        self.assertEqual(report["case_status_counts"],
                         {"REGISTRO_COMPATIVEL_INELEGIVEL": 1})

    def test_same_visual_pair_but_different_board_is_diagnostic_not_exact(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 67)
        self.record("different_board", "OK", 67, board="B2")
        self._sync_values()
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {
            "PAR_EXATO_METADADOS_DIFERENTES": 1
        })

    def test_unseen_archive_does_not_fabricate_memory(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 70)
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["case_status_counts"], {
            "SEM_REGISTROS_NA_CATEGORIA_LUZ": 1
        })

    def test_ocr_error_is_invalid_even_when_filename_gives_category(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 70)
        def bad_ocr(frame):
            return self.reference, frame, {
                "board": "", "parts": "", "value": "FALTANDO"
            }
        report = reconcile_archive(self.root, extractor=bad_ocr)
        self.assertEqual(report["case_status_counts"], {"OCR_INVALIDO": 1})
        self.assertEqual(report["cases"][0]["category_hint"], "FALTANDO")

    def test_both_labels_and_adaptive_light_are_kept_separate(self):
        self.archive("2026-10-09_0800_FALTANDO.png", "OK", 60)
        self.archive("2026-10-09_0801_FALTANDO_TOP.png", "NG", 90)
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["archive_png_count"], 2)
        self.assertEqual(report["by_label"]["OK"],
                         {"SEM_REGISTROS_NA_CATEGORIA_LUZ": 1})
        self.assertEqual(report["by_label"]["NG"],
                         {"SEM_REGISTROS_NA_CATEGORIA_LUZ": 1})
        self.assertEqual(
            [item["lighting_mode"] for item in report["cases"]],
            ["SIDE", "TOP"],
        )


if __name__ == "__main__":
    unittest.main()
