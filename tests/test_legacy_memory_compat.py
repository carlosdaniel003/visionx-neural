"""Contratos de compatibilidade do formato legado em modo somente leitura."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_reconciler import (
    reconcile_archive, write_reconciliation_report,
)
from src.services.startup_regression.legacy_memory_compat import (
    COMPATIBLE, inspect_legacy_record, summarize_legacy_profiles,
)


class LegacyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.ok_archive = self.root / "public" / "ok_archive"
        self.ng_archive = self.root / "public" / "ng_archive"
        self.ok_memory = self.root / "public" / "dataset" / "nao_anomalia"
        self.ng_memory = self.root / "public" / "dataset" / "anomalia"
        for path in (self.ok_archive, self.ng_archive, self.ok_memory, self.ng_memory):
            path.mkdir(parents=True)
        self.reference = np.full((35, 55, 3), 20, dtype=np.uint8)
        self.test = np.full((35, 55, 3), 120, dtype=np.uint8)
        self.info = {
            "board": "P22-22200 L13",
            "parts": "U2~5",
            "category": "FALTANDO",
            "value": "0 <= 54.872 <= 10 FALTANDO",
            "lighting_mode": "SIDE",
        }

    @staticmethod
    def png(target: Path, frame: np.ndarray):
        ok, blob = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError("Falha na criação PNG sintético")
        target.write_bytes(blob.tobytes())

    def archive(self, label="OK"):
        path = self.ok_archive if label == "OK" else self.ng_archive
        target = path / "2026-10-09_1225_FALTANDO.png"
        self.png(target, self.test)
        return target

    def memory(
        self, name="old", label="OK", schema="visionx.memory.v2",
        operator="button", with_pair=True, signature=True,
        declared_hash=None, dedup=False,
    ):
        folder = self.ok_memory if label == "OK" else self.ng_memory
        payload = {
            "label": label,
            "decision": {
                "operator_label": label,
                "source": operator,
            },
            "aoi_info": self.info,
            "storage": {
                "reference_image_file": f"{name}_reference.png" if with_pair else "",
                "test_image_file": f"{name}_test.png" if with_pair else "",
                "test_image_fingerprint": (
                    declared_hash if declared_hash is not None
                    else image_fingerprint(self.test)
                ),
                "duplicate_visual_of_json": "older.json" if dedup else "",
            },
            "analysis": {
                "anomaly_memory": {"vector": [0.1] * 224} if signature else None
            },
        }
        if schema is not None:
            payload["schema"] = schema
        if with_pair:
            self.png(folder / f"{name}_reference.png", self.reference)
            self.png(folder / f"{name}_test.png", self.test)
        target = folder / f"{name}.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def extract(self, frame):
        return self.reference, frame, dict(self.info)

    def test_old_schema_with_two_real_pngs_human_and_signature_is_candidate(self):
        self.archive()
        self.memory()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["memory_record_reasons"],
                         {"SCHEMA_NAO_SUPORTADO": 1})
        self.assertEqual(report["legacy_memory_profiles"]["schema_distribution"],
                         {"visionx.memory.v2": 1})
        self.assertEqual(report["legacy_memory_profiles"]["compatible_in_simulation"], 1)
        self.assertEqual(report["case_status_counts"],
                         {"REGISTRO_COMPATIVEL_INELEGIVEL": 1})
        self.assertEqual(
            report["cases"][0]["legacy_compatibility"]["status"],
            "LEGADO_PAR_SIMULADO_CONCORDA",
        )
        self.assertFalse(report["cases"][0]["legacy_compatibility"][
            "verified_in_production"
        ])
        self.assertFalse(report["legacy_memory_profiles"]["production_knn_coverage_changed"])
        self.assertEqual(before, {
            p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()
        })
        paths = write_reconciliation_report(
            report, self.root / "reports" / "startup_regression"
        )
        self.assertTrue(paths[0].exists() and paths[1].exists())
        txt = paths[1].read_text(encoding="utf-8")
        self.assertIn("LEGADO_PAR_SIMULADO_CONCORDA", txt)
        self.assertIn("visionx.memory.v2", txt)

    def test_unknown_schema_is_recorded_explicitly_and_does_not_become_knn(self):
        self.archive()
        self.memory(schema=None)
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["legacy_memory_profiles"]["schema_distribution"],
                         {"SEM_SCHEMA": 1})
        self.assertEqual(report["case_status_counts"],
                         {"REGISTRO_COMPATIVEL_INELEGIVEL": 1})

    def test_json_only_with_dedup_pointer_never_satisfies_exact_pair(self):
        self.archive()
        path = self.memory(with_pair=False, dedup=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        detail = inspect_legacy_record(path, "OK", data)
        self.assertEqual(detail["status"], "LEGADO_DEDUP_SEM_PAR")
        self.assertIsNone(detail["would_be_key"])
        report = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(report["legacy_memory_profiles"]["compatible_in_simulation"], 0)
        self.assertEqual(report["cases"][0]["legacy_compatibility"]["candidate_count"], 0)

    def test_old_embedding_alone_is_not_inferred_as_a_verified_signature(self):
        self.archive()
        path = self.memory(signature=False)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["analysis"] = {"embedding": [0.9, 0.4]}
        path.write_text(json.dumps(data), encoding="utf-8")
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(result["legacy_memory_profiles"]["eligibility_reasons"],
                         {"LEGADO_SEM_ASSINATURA_ANOMALIA": 1})

    def test_automatic_source_is_never_silently_promoted_to_human(self):
        self.archive()
        self.memory(operator="automatic")
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(
            result["legacy_memory_profiles"]["eligibility_reasons"],
            {"LEGADO_SEM_CONFIRMACAO_HUMANA": 1},
        )
        self.assertEqual(result["legacy_memory_profiles"]["compatible_in_simulation"], 0)

    def test_divergent_declared_image_hash_is_rejected(self):
        self.archive()
        self.memory(declared_hash="0" * 64)
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(
            result["legacy_memory_profiles"]["eligibility_reasons"],
            {"LEGADO_HASH_TESTE_DIVERGENTE": 1},
        )
        self.assertEqual(result["cases"][0]["legacy_compatibility"]["candidate_count"], 0)

    def test_legacy_label_disagrees_with_archive_without_becoming_pass(self):
        self.archive("OK")
        self.memory(label="NG")
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(
            result["cases"][0]["legacy_compatibility"]["status"],
            "LEGADO_PAR_SIMULADO_DIVERGE",
        )
        self.assertEqual(result["case_status_counts"],
                         {"REGISTRO_COMPATIVEL_INELEGIVEL": 1})

    def test_conflicting_legacy_labels_for_identical_pair_require_review(self):
        self.archive()
        self.memory(name="ok", label="OK")
        self.memory(name="ng", label="NG")
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(result["case_status_counts"],
                         {"CONFLITO_COM_MEMORIA_LEGADA": 1})
        self.assertEqual(
            result["cases"][0]["legacy_compatibility"]["status"],
            "LEGADO_CONFLITO_EXATO",
        )

    def test_different_board_prevents_legacy_exact_match(self):
        self.archive()
        path = self.memory()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["aoi_info"]["board"] = "OUTRA PLACA"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = reconcile_archive(self.root, extractor=self.extract)
        self.assertEqual(result["legacy_memory_profiles"]["compatible_in_simulation"], 1)
        self.assertEqual(
            result["cases"][0]["legacy_compatibility"]["status"],
            "SEM_PAR_LEGADO_AUDITAVEL",
        )

    def test_current_v3_exact_does_not_use_legacy_simulator(self):
        path = self.memory(schema="visionx.memory.v3")
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            inspect_legacy_record(path, "OK", payload)

    def test_ocr_missing_does_not_use_archive_filename_to_fabricate_match(self):
        self.archive()
        self.memory()
        def empty_ocr(frame):
            return self.reference, frame, {"board": "", "parts": "", "value": "FALTANDO"}
        result = reconcile_archive(self.root, extractor=empty_ocr)
        self.assertEqual(result["case_status_counts"], {"OCR_INVALIDO": 1})
        self.assertEqual(result["cases"][0]["legacy_compatibility"]["candidate_count"], 0)

    def test_schema_statistics_are_deterministic(self):
        summaries = summarize_legacy_profiles([
            {
                "path": "dataset/old.json",
                "legacy_profile": {
                    "schema": "visionx.memory.v2",
                    "status": COMPATIBLE,
                },
            },
            {
                "path": "dataset/no_schema.json",
                "legacy_profile": {
                    "schema": "SEM_SCHEMA",
                    "status": "LEGADO_SEM_PAR_PNG",
                },
            },
        ])
        self.assertEqual(summaries["records_examined"], 2)
        self.assertEqual(summaries["compatible_in_simulation"], 1)
        self.assertEqual(summaries["schema_distribution"], {
            "SEM_SCHEMA": 1, "visionx.memory.v2": 1,
        })


if __name__ == "__main__":
    unittest.main()
