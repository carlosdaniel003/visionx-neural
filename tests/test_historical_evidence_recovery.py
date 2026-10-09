"""Recuperação read-only: todo reconhecimento depende de evidência, nunca nome."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.historical_evidence_recovery import (
    EvidenceIndex, inspect_historical_evidence, write_evidence_report,
)


class HistoricalEvidenceTests(unittest.TestCase):
    def setUp(self):
        scope = tempfile.TemporaryDirectory()
        self.addCleanup(scope.cleanup)
        self.root = Path(scope.name)
        self.ok = self.root / "public" / "dataset" / "nao_anomalia" / "FALTANDO"
        self.ng = self.root / "public" / "dataset" / "anomalia" / "FALTANDO"
        self.archive = self.root / "public" / "ok_archive"
        for p in (self.ok, self.ng, self.archive):
            p.mkdir(parents=True)
        self.source = np.full((60, 80, 3), 100, dtype=np.uint8)
        self.reference = np.full((50, 70, 3), 50, dtype=np.uint8)
        self.test = np.full((50, 70, 3), 125, dtype=np.uint8)
        self.info = {
            "category": "FALTANDO", "board": "B1", "parts": "U2~5",
            "value": "0 <= 10 <= 20 FALTANDO", "lighting_mode": "SIDE",
        }

    @staticmethod
    def png(path, pixels):
        success, content = cv2.imencode(".png", pixels)
        if not success:
            raise ValueError("Não codificou PNG sintético")
        path.write_bytes(content.tobytes())
        return path

    def record(self, *, label="OK", name="memory",
               source_hash=None, test_hash=None, ref_hash=None,
               source_file="source.png", test_file="", ref_file="",
               operator="button", schema="visionx.memory.v2",
               with_signature=True):
        parent = self.ok if label == "OK" else self.ng
        payload = {
            "schema": schema, "label": label,
            "decision": {"operator_label": label, "source": operator},
            "aoi_info": dict(self.info),
            "analysis": {
                "anomaly_memory": (
                    {"vector": [0.1] * 224} if with_signature else None
                )
            },
            "storage": {
                "source_image_file": source_file,
                "test_image_file": test_file,
                "reference_image_file": ref_file,
                "source_image_fingerprint": source_hash,
                "test_image_fingerprint": test_hash,
                "reference_image_fingerprint": ref_hash,
            },
        }
        dest = parent / f"{name}.json"
        dest.write_text(json.dumps(payload), encoding="utf-8")
        return dest

    def extract(self, image):
        return self.reference, self.test, dict(self.info)

    def run_recovery(self):
        return inspect_historical_evidence(self.root, extractor=self.extract)

    def test_reconstructed_with_source_and_two_declared_crop_hashes_stays_preview(self):
        self.png(self.archive / "real_source.png", self.source)
        memory = self.record(
            source_file="real_source.png",
            source_hash=image_fingerprint(self.source),
            test_hash=image_fingerprint(self.test),
            ref_hash=image_fingerprint(self.reference),
        )
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.run_recovery()
        self.assertEqual(report["status_counts"], {"PAR_RECONSTRUIDO_PARA_REVISAO": 1})
        self.assertEqual(report["preview_for_manual_review"], 1)
        self.assertEqual(report["ready_for_migration"], 0)
        self.assertFalse(report["production_knn_modified"])
        self.assertFalse(report["dataset_modified"])
        self.assertFalse(report["startup_gate_enabled"])
        self.assertTrue(report["cases"][0]["human_confirmed"])
        self.assertTrue(report["cases"][0]["signature_present"])
        self.assertEqual(
            report["cases"][0]["reconstruction"]["status"],
            "RECONSTRUCAO_COM_HASHES_CONFERIDOS",
        )
        self.assertEqual(before, {
            p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()
        })
        json_path, txt_path = write_evidence_report(
            report, self.root / "reports" / "startup_regression"
        )
        self.assertEqual(
            json.loads(json_path.read_text(encoding="utf-8"))["schema"],
            report["schema"],
        )
        self.assertIn("PAR_RECONSTRUIDO_PARA_REVISAO",
                      txt_path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            write_evidence_report(report, self.ok)

    def test_hash_found_under_different_name_is_identity_not_timestamp(self):
        self.png(self.archive / "different_name.png", self.test)
        self.record(source_file="", test_file="missing_test.png",
                    test_hash=image_fingerprint(self.test))
        report = self.run_recovery()
        case = report["cases"][0]
        self.assertEqual(case["status"], "TESTE_POR_HASH_SEM_GABARITO")
        self.assertTrue(case["test"]["hash_verified"])
        self.assertFalse(case["test"]["same_folder_filename_found"])
        self.assertEqual(case["test"]["name_candidates"], 0)
        self.assertFalse(report["dataset_modified"])

    def test_filename_only_is_never_treated_as_verified_identity(self):
        self.png(self.archive / "same_name.png", self.test)
        self.record(source_file="", test_file="same_name.png")
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH": 1})
        self.assertFalse(report["cases"][0]["test"]["hash_verified"])

    def test_absent_image_hash_never_fabricates_match(self):
        self.png(self.archive / "same_name.png", self.test)
        self.record(source_file="", test_file="same_name.png",
                    test_hash="0" * 64)
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH": 1})
        self.assertEqual(report["cases"][0]["test"]["hash_candidates"], 0)

    def test_two_images_found_by_hash_are_review_only_when_original_pair_missing(self):
        self.png(self.archive / "ref_renamed.png", self.reference)
        self.png(self.archive / "test_renamed.png", self.test)
        self.record(source_file="",
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test))
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"DOIS_HASHES_DE_PARES_LOCALIZADOS_PARA_REVISAO": 1})
        self.assertEqual(report["ready_for_migration"], 0)

    def test_automatic_record_never_promoted_even_with_complete_images(self):
        self.png(self.archive / "source.png", self.source)
        self.record(source_hash=image_fingerprint(self.source),
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test),
                    operator="automatic")
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"ORIGEM_HUMANA_NAO_COMPROVADA": 1})
        self.assertEqual(report["preview_for_manual_review"], 0)

    def test_source_verified_but_ocr_incorrect_never_qualifies(self):
        self.png(self.archive / "source.png", self.source)
        self.record(source_hash=image_fingerprint(self.source),
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test))

        def wrong_ocr(_frame):
            info = dict(self.info)
            info["parts"] = "R100"
            return self.reference, self.test, info

        report = inspect_historical_evidence(self.root, extractor=wrong_ocr)
        self.assertEqual(report["status_counts"],
                         {"RECONSTRUCAO_OCR_DIVERGENTE": 1})
        self.assertEqual(report["ready_for_migration"], 0)

    def test_source_verified_but_test_hash_incorrect_is_rejected(self):
        self.png(self.archive / "source.png", self.source)
        self.record(source_hash=image_fingerprint(self.source),
                    ref_hash=image_fingerprint(self.reference),
                    test_hash="b" * 64)
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"RECONSTRUCAO_TESTE_HASH_DIVERGENTE": 1})

    def test_source_without_declared_fingerprint_does_not_claim_verified(self):
        self.png(self.archive / "source.png", self.source)
        self.record(source_hash=None,
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test))
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH": 1})
        self.assertFalse(report["cases"][0]["source"]["hash_verified"])

    def test_source_linked_in_same_record_folder_without_hash_is_only_candidate(self):
        self.png(self.ok / "source.png", self.source)
        self.record(source_hash=None,
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test))
        result = self.run_recovery()
        self.assertEqual(result["status_counts"], {
            "ORIGEM_JSON_LOCAL_SEM_HASH_PARA_REVISAO": 1
        })
        case = result["cases"][0]
        self.assertTrue(case["source"]["same_folder_filename_found"])
        self.assertFalse(case["source"]["hash_verified"])
        self.assertEqual(case["reconstruction"]["status"],
                         "RECONSTRUCAO_COM_HASHES_CONFERIDOS")
        self.assertFalse(case["migration_ready"])
        self.assertEqual(result["ready_for_migration"], 0)

    def test_nonlocal_name_match_never_runs_extraction_without_hash(self):
        self.png(self.archive / "source.png", self.source)
        self.record(source_hash=None, test_hash=None, ref_hash=None)
        called = []

        def forbidden(_image):
            called.append(True)
            raise AssertionError("Não pode extrair sem vínculo local/hash")

        result = inspect_historical_evidence(self.root, extractor=forbidden)
        self.assertEqual(result["status_counts"], {
            "ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH": 1
        })
        self.assertEqual(called, [])

    def test_no_registered_images_are_retained_as_no_evidence(self):
        self.record(source_file="", ref_file="", test_file="")
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"SEM_EVIDENCIA_VISUAL_LOCALIZAVEL": 1})

    def test_existing_legacy_pair_is_not_registered_as_new_memory(self):
        self.png(self.ok / "ref.png", self.reference)
        self.png(self.ok / "test.png", self.test)
        self.record(source_file="", ref_file="ref.png", test_file="test.png",
                    ref_hash=image_fingerprint(self.reference),
                    test_hash=image_fingerprint(self.test))
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"PAR_LEGADO_JA_AUDITAVEL": 1})

    def test_v3_records_are_excluded_from_simulated_migration(self):
        self.record(schema="visionx.memory.v3")
        report = self.run_recovery()
        self.assertEqual(report["v3_jsons"], 1)
        self.assertEqual(report["legacy_jsons"], 0)
        self.assertEqual(report["status_counts"], {"ATUAL_V3_FORA_ESCOPO": 1})

    def test_does_not_follow_symlink_to_external_sensitive_png(self):
        external_dir = tempfile.TemporaryDirectory()
        self.addCleanup(external_dir.cleanup)
        secret = self.png(Path(external_dir.name) / "secret.png", self.test)
        link = self.archive / "leak.png"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            self.skipTest("Windows CI sem permissão de symlink")
        self.record(source_file="", test_file="leak.png",
                    test_hash=image_fingerprint(self.test))
        report = self.run_recovery()
        self.assertEqual(report["scanned_pngs"], 0)
        self.assertEqual(report["status_counts"],
                         {"HASH_DECLARADO_SEM_PNG_COMPATIVEL": 1})

    def test_refuses_png_path_traversal_strings(self):
        self.png(self.archive / "same_name.png", self.test)
        self.record(source_file="", test_file="../same_name.png")
        report = self.run_recovery()
        self.assertEqual(report["status_counts"],
                         {"SEM_EVIDENCIA_VISUAL_LOCALIZAVEL": 1})


if __name__ == "__main__":
    unittest.main()
