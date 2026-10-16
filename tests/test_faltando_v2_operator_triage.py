"""Record human defect categories, not fabricated FALTANDO NG/OK labels."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.scripts.faltando_v2_record_operator_triage import (
    record_triage, verify_operator_triage,
)
from src.services.faltando_neural_qualification import QualificationStore
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


class OperatorTriageTests(CNNV2Fixture):
    def _fixture(self):
        store = QualificationStore(self.manifest_path)
        ng = [s for s in store.by_path if store.expected_labels[s] == "NG"]
        ok = [s for s in store.by_path if store.expected_labels[s] == "OK"]
        sources = [
            (ng[0], "FALTANDO"), (ng[1], "INVERTIDO"),
            (ok[0], "IGNORAR"),
        ]
        digest = sha256(self.manifest_path.read_bytes()).hexdigest()
        review = {
            "schema": "visionx.faltando_v2_operator_contact_sheet_triage.v1",
            "source": "TEST_OPERATOR_3_PAIRS",
            "manifest_sha256": digest,
            "audit_candidate_sha256": "fake_candidate_sha",
            "operator_review_scope": "TEST_SUBSET",
            "individual_qualification_file_created": False,
            "production_training_approved": False,
            "bulk_archive_labels_modified": False,
            "cases": [
                {
                    "source": path, "source_sha256":
                        store.by_path[path]["source_sha256"],
                    "visual_category": label, "note": "Operador conferiu",
                } for path, label in sources
            ],
        }
        docs = self.root/"docs"
        docs.mkdir()
        (docs/"faltando_v2_operator_triage_20261016.json").write_text(
            json.dumps(review), encoding="utf-8",
        )
        folder = (self.root/"reports"/"faltando_neural"
                  /"failed_pair_audits"/"test")
        folder.mkdir(parents=True)
        cases = {}
        for section, items in (
            ("historico", [sources[2]]),
            ("treino", [sources[2], sources[0]]),
            ("holdout", [sources[1]]),
        ):
            cases[section] = [
                {
                    "source": name, "source_sha256":
                        store.by_path[name]["source_sha256"],
                    "label_from_folder": store.expected_labels[name],
                } for name, _ in items
            ]
        audit = {
            "schema": "visionx.faltando_v2_failed_pair_audit.v1",
            "manifest_sha256": digest,
            "source_candidate_sha256": "fake_candidate_sha",
            "read_only": True,
            "unique_source_count": 3,
            "cases": cases,
        }
        index = folder/"audit_index.json"
        index.write_text(json.dumps(audit), encoding="utf-8")
        return index, sources, review, audit

    def test_preserves_source_and_original_labels_and_writes_only_report(self):
        index, sources, review, audit = self._fixture()
        before = {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*.png")
        }
        output = record_triage(self.root, self.manifest_path, index)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["source_count"], 3)
        self.assertEqual(
            result["visual_categories"],
            {"FALTANDO": 1, "INVERTIDO": 1, "IGNORAR": 1},
        )
        self.assertEqual(result["audit_occurrences"], {
            "historico": 1, "treino": 2, "holdout": 1,
        })
        self.assertEqual(set(result["quarantine_for_future_faltando_experiments"]),
                         {sources[1][0], sources[2][0]})
        self.assertEqual(result["reviewed_missing_from_contact_sheets"],
                         [sources[0][0]])
        self.assertEqual(
            result["reviewed_general_ng_from_contact_sheets"],
            sorted([sources[0][0], sources[1][0]]),
        )
        self.assertTrue(result["cross_category_quality_ng_keeps_inverted"])
        self.assertTrue(result["aoi_category_is_only_routing_hint"])
        self.assertTrue(result["general_ng_does_not_imply_missing_component"])
        self.assertTrue(result["inverted_not_relabelled_ok"])
        self.assertFalse(result["production_approval_eligible"])
        self.assertFalse(result["train_performed"])
        self.assertFalse(result["recompute_accuracy_on_filtered_cases"])
        self.assertEqual(before, {
            name: sha256(Path(name).read_bytes()).hexdigest()
            for name in before
        })
        self.assertFalse((self.run/"qualification.json").exists())
        self.assertFalse(
            (self.root/"reports/neural_online/live_active.json").exists()
        )

    def test_fails_closed_on_changed_source_sha(self):
        index, sources, review, audit = self._fixture()
        file = self.root/"docs"/"faltando_v2_operator_triage_20261016.json"
        review["cases"][0]["source_sha256"] = "0" * 64
        file.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA da fonte"):
            verify_operator_triage(self.root, self.manifest_path, index)

    def test_fails_closed_on_missing_review_or_report_forgery(self):
        index, sources, review, audit = self._fixture()
        file = self.root/"docs"/"faltando_v2_operator_triage_20261016.json"
        review["cases"] = review["cases"][:2]
        file.write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não existe na revisão"):
            verify_operator_triage(self.root, self.manifest_path, index)
        file.write_text(json.dumps({**review, "cases": self._fixture_cases(sources)}),
                        encoding="utf-8")
        audit["source_candidate_sha256"] = "different_checkpoint"
        index.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Auditoria de origem"):
            verify_operator_triage(self.root, self.manifest_path, index)

    def _fixture_cases(self, sources):
        store = QualificationStore(self.manifest_path)
        return [
            {"source": source,
             "source_sha256": store.by_path[source]["source_sha256"],
             "visual_category": category, "note": "Operador conferiu"}
            for source, category in sources
        ]

    def test_fails_closed_on_arbitrary_report_location(self):
        index, *_ = self._fixture()
        with self.assertRaisesRegex(ValueError, "localização permitida"):
            verify_operator_triage(
                self.root, self.manifest_path, self.manifest_path,
            )
