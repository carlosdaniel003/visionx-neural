"""Contratos offline: KNN assinatura leave-one-out vs índice visual exato."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.verified_memory_router import VerifiedKNNMemory
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_exact_memory_plan import (
    plan_archive_exact_memory,
)
from src.services.startup_regression.legacy_knn_signature_audit import (
    audit_signature_knn, load_signature_records,
)
from src.services.startup_regression.two_level_memory_diagnostic_cli import (
    write_dual_level_reports,
)


def record(name, category, mode, label, strength, *, event=None):
    vector = [float(strength)] * 224
    import hashlib
    return {
        "path": name, "schema": "visionx.memory.v2",
        "category": category, "lighting_mode": mode,
        "label": label, "status": "ELEGIVEL",
        "event_id": event, "signature": {"vector": vector},
        "signature_hash": hashlib.sha256(
            np.asarray(vector, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


class SignatureSimulationTests(unittest.TestCase):
    def test_leave_one_out_never_matches_itself(self):
        rows = [record("one", "FALTANDO", "SIDE", "OK", .4)]
        out = audit_signature_knn(Path("."), records=rows)
        self.assertEqual(out["status_counts"], {"SEM_VIZINHOS": 1})
        self.assertFalse(out["archive_212_accuracy_measured"])

    def test_two_classes_have_leave_one_out_neighbors_of_same_label(self):
        rows = [
            record("ok1", "FALTANDO", "SIDE", "OK", .2),
            record("ok2", "FALTANDO", "SIDE", "OK", .2),
            record("ng1", "FALTANDO", "SIDE", "NG", .95),
            record("ng2", "FALTANDO", "SIDE", "NG", .95),
        ]
        out = audit_signature_knn(Path("."), records=rows, top_k=1)
        self.assertEqual(out["status_counts"], {"CONCORDA": 4})
        self.assertEqual(out["eligible_signatures"], 4)
        self.assertTrue(all(c["neighbors_used"] == 1 for c in out["cases"]))
        self.assertTrue(all(c["path"] != c["neighbors"][0]["path"] for c in out["cases"]))
        self.assertFalse(out["knn_runtime_modified"])

    def test_exact_same_signature_opposite_label_is_conflict(self):
        rows = [
            record("one", "INVERTIDO", "TOP", "OK", .35),
            record("two", "INVERTIDO", "TOP", "NG", .35),
        ]
        out = audit_signature_knn(Path("."), records=rows, top_k=1)
        self.assertEqual(out["status_counts"], {"CONFLITO_DE_ASSINATURA": 2})
        self.assertEqual(out["same_signature_conflict_records"], 2)

    def test_strict_category_and_lighting_prevent_global_leakage(self):
        rows = [
            record("ok1", "FALTANDO", "SIDE", "OK", .25),
            record("ngtop", "FALTANDO", "TOP", "NG", .25),
            record("ngother", "EMBORCADO", "SIDE", "NG", .25),
        ]
        out = audit_signature_knn(Path("."), records=rows)
        self.assertEqual(out["status_counts"], {"SEM_VIZINHOS": 3})

    def test_explicit_multilight_event_excluded_from_neighbor_pool(self):
        rows = [
            record("a", "FALTANDO", "SIDE", "OK", .2, event="event1"),
            record("b", "FALTANDO", "SIDE", "OK", .2, event="event1"),
        ]
        out = audit_signature_knn(Path("."), records=rows)
        self.assertEqual(out["status_counts"], {"SEM_VIZINHOS": 2})

    def test_rejects_invalid_signatures_and_forged_operator_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "public" / "dataset" / "nao_anomalia"
            folder.mkdir(parents=True)
            payload = {
                "schema": "visionx.memory.v2", "label": "OK",
                "decision": {"operator_label": "OK", "source": "automatic"},
                "aoi_info": {
                    "category": "FALTANDO", "lighting_mode": "SIDE"
                },
                "analysis": {
                    "anomaly_memory": {"vector": [0.2]*224}
                },
            }
            (folder / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
            data = load_signature_records(root)
            self.assertEqual(data[0]["status"], "SEM_ROTULO_HUMANO_VERIFICADO")
            out = audit_signature_knn(root)
            self.assertEqual(out["eligible_signatures"], 0)
            self.assertEqual(out["rejected_records"],
                             {"SEM_ROTULO_HUMANO_VERIFICADO": 1})
            payload["decision"]["source"] = "button"
            payload["analysis"]["anomaly_memory"] = {"vector": [0.1]*20}
            (folder / "bad.json").write_text(json.dumps(payload), encoding="utf-8")
            data = load_signature_records(root)
            self.assertEqual(data[0]["status"], "ASSINATURA_INVALIDA")

    def test_no_model_download_or_cnn_called(self):
        from unittest.mock import patch
        rows = [
            record("a", "MUITOADESIVO", "MID", "OK", .2),
            record("b", "MUITOADESIVO", "MID", "OK", .2),
        ]
        with patch(
            "src.core.experts.knn_expert.KNNExpert.__init__",
            side_effect=AssertionError("Não instanciar KNN nem baixar modelos"),
        ):
            out = audit_signature_knn(Path("."), records=rows)
        self.assertEqual(out["status_counts"], {"CONCORDA": 2})


class ExactArchivePlanTests(unittest.TestCase):
    def setUp(self):
        area = tempfile.TemporaryDirectory()
        self.addCleanup(area.cleanup)
        self.root = Path(area.name)
        self.ok = self.root / "public" / "ok_archive"
        self.ng = self.root / "public" / "ng_archive"
        self.ok.mkdir(parents=True)
        self.ng.mkdir(parents=True)
        self.ref = np.full((45, 50, 3), 40, dtype=np.uint8)
        self.test = np.full((45, 50, 3), 90, dtype=np.uint8)
        self.info = {
            "board": "B1", "parts": "U2~5",
            "value": "0 <= 10 <= 20 FALTANDO",
        }

    def png(self, name, label, pixels=None):
        img = pixels if pixels is not None else self.test
        dest = (self.ok if label == "OK" else self.ng) / name
        success, png_data = cv2.imencode(".png", img)
        assert success
        dest.write_bytes(png_data.tobytes())
        return dest

    def extract(self, frame):
        return self.ref.copy(), self.test.copy(), dict(self.info)

    def key(self):
        return VerifiedKNNMemory._key({
            **self.info, "category": "FALTANDO", "lighting_mode": "SIDE",
        }, image_fingerprint(self.ref), image_fingerprint(self.test))

    def test_new_png_has_plan_but_not_human_confirmation_or_import(self):
        png = self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        before = png.read_bytes()
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=[]
        )
        self.assertEqual(out["status_counts"], {"PENDENTE_ORIGEM_HUMANA": 1})
        self.assertEqual(out["ready_for_import"], 0)
        self.assertEqual(out["pending_human_evidence"], 1)
        self.assertFalse(out["archive_label_used_as_human_approval"])
        self.assertFalse(out["writes_to_dataset"])
        self.assertFalse(out["startup_gate_enabled"])
        self.assertEqual(png.read_bytes(), before)
        self.assertEqual(len(out["cases"][0]["reference_pixel_sha256"]), 64)

    def test_legacy_side_without_manifest_remains_uncertain(self):
        self.png("2026-10-09_1700_FALTANDO.png", "OK")
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=[]
        )
        self.assertEqual(out["status_counts"], {
            "PENDENTE_ORIGEM_HUMANA_E_LUZ": 1
        })

    def test_verified_pair_from_v3_is_counted_existing_not_imported(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        key = self.key()
        rows = [{
            "path": "public/dataset/nao_anomalia/FALTANDO/memory_v3.json",
            "label": "OK", "reason": "VERIFICADO_KNN",
            "key": key, "category": "FALTANDO",
            "lighting_mode": "SIDE", "board": "B1", "parts": "U2~5",
        }]
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=rows
        )
        self.assertEqual(out["status_counts"], {"JA_VERIFICADO_EXATO_V3": 1})
        self.assertEqual(out["verified_existing"], 1)
        self.assertEqual(out["ready_for_import"], 0)

    def test_legacy_pair_remains_pending_not_v3(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        key = self.key()
        rows = [{
            "path": "legacy.json", "label": "OK",
            "reason": "SCHEMA_NAO_SUPORTADO",
            "category": "FALTANDO",
            "lighting_mode": "SIDE", "board": "B1", "parts": "U2~5",
            "legacy_profile": {
                "status": "LEGADO_PAR_AUDITAVEL_SIMULADO",
                "would_be_key": key,
            },
        }]
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=rows
        )
        self.assertEqual(out["status_counts"], {"PAR_LEGADO_EXATO_PENDENTE": 1})
        self.assertFalse(out["cases"][0]["migration_ready"])

    def test_ng_conflict_does_not_get_approved(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "NG")
        rows = [{
            "path": "memory.json", "label": "OK",
            "reason": "VERIFICADO_KNN", "key": self.key(),
            "category": "FALTANDO", "lighting_mode": "SIDE",
            "board": "B1", "parts": "U2~5",
        }]
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=rows
        )
        self.assertEqual(out["status_counts"],
                         {"CONFLITO_COM_MEMORIA_EXISTENTE": 1})

    def test_ocr_invalid_stops_before_hash_pair_plan(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        def fail_ocr(frame):
            return self.ref, self.test, {
                "board": "", "parts": "", "value": "FALTANDO"
            }
        out = plan_archive_exact_memory(
            self.root, extractor=fail_ocr, memory_rows=[]
        )
        self.assertEqual(out["status_counts"], {"OCR_INVALIDO": 1})
        self.assertIsNone(out["cases"][0]["test_pixel_sha256"])

    def test_cross_label_identical_screenshot_is_invalid(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        self.png("2026-10-09_1701_FALTANDO_SIDE.png", "NG")
        out = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=[]
        )
        self.assertEqual(out["status_counts"],
                         {"PNG_OU_ROTULOS_INVALIDOS": 2})

    def test_output_is_separate_file_under_reports_and_never_dataset(self):
        self.png("2026-10-09_1700_FALTANDO_SIDE.png", "OK")
        signature = audit_signature_knn(
            self.root, records=[record("one", "FALTANDO", "SIDE", "OK", .3)]
        )
        archive = plan_archive_exact_memory(
            self.root, extractor=self.extract, memory_rows=[]
        )
        target = self.root / "reports" / "startup_regression"
        j, t = write_dual_level_reports(signature, archive, target)
        self.assertTrue(j.is_file())
        self.assertTrue(t.is_file())
        saved = json.loads(j.read_text(encoding="utf-8"))
        self.assertTrue(saved["two_results_must_not_be_merged"])
        self.assertFalse(saved["modifies_production"])
        self.assertEqual(len(list(self.ok.glob("*.png"))), 1)
        with self.assertRaises(ValueError):
            write_dual_level_reports(
                signature, archive, self.root / "public" / "dataset"
            )


if __name__ == "__main__":
    unittest.main()
