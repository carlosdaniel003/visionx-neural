"""Teste de NG seguro: KNN real, balanceamento, dedup e revisão sem produção."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.services.startup_regression.knn_ng_discrimination_audit import (
    MODES, audit_knn_ng_discrimination,
)
from src.services.startup_regression.knn_ng_discrimination_cli import (
    write_discrimination_reports,
)
from src.services.startup_regression.legacy_knn_signature_audit import (
    audit_signature_knn,
)


def entry(name, label, signal, *, category="FALTANDO", light="SIDE",
          event=None):
    vector = np.zeros(224, dtype=np.float32)
    vector[0:4] = np.asarray(signal, dtype=np.float32)
    signature = {"schema": "visionx.anomaly.v1", "vector": vector.tolist()}
    return {
        "path": name, "label": label,
        "category": category, "lighting_mode": light,
        "event_id": event, "schema": "visionx.memory.v2",
        "status": "ELEGIVEL", "signature": signature,
        "signature_hash": hashlib.sha256(vector.tobytes()).hexdigest(),
    }


class KNNNGDiscriminationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def sample(self):
        return [
            entry("ok_a", "OK", [.15, .12, .01, .0]),
            entry("ok_b", "OK", [.16, .1, .02, .0]),
            entry("ok_c", "OK", [.17, .1, .03, .0]),
            entry("ok_d", "OK", [.14, .11, .01, .0]),
            entry("ok_e", "OK", [.13, .1, .01, .0]),
            entry("ng_a", "NG", [.8, .1, .1, .0]),
            entry("ng_b", "NG", [.9, .12, .1, .0]),
            entry("ng_c", "NG", [.78, .09, .12, .0]),
        ]

    def test_reuses_baseline_vote_from_existing_diagnostic(self):
        rows = self.sample()
        previous = audit_signature_knn(self.root, records=rows)
        current = audit_knn_ng_discrimination(self.root, records=rows)
        old = {r["path"]: r for r in previous["cases"]}
        for case in current["cases"]:
            baseline = case["modes"]["BASELINE_TOP5"]
            self.assertEqual(
                baseline["prediction"],
                old[case["path"]]["predicted_label"]
            )
            self.assertAlmostEqual(
                baseline["vote_ng"], old[case["path"]]["vote_ng"], places=7
            )
        self.assertEqual(set(current["comparisons"]), set(MODES))
        self.assertFalse(current["dataset_modified"])
        self.assertFalse(current["startup_gate_enabled"])
        self.assertFalse(current["archive_212_accuracy_measured"])

    def test_duplicate_query_signature_is_excluded_entirely_in_dedup_modes(self):
        rows = [
            entry("q", "OK", [.1, .2, .3, .4]),
            entry("exact_copy", "OK", [.1, .2, .3, .4]),
            entry("different_ok", "OK", [.14, .1, .35, .45]),
            entry("different_ng", "NG", [.8, .2, .0, .1]),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        q = next(c for c in audit["cases"] if c["path"] == "q")
        self.assertEqual(q["modes"]["BASELINE_TOP5"]["neighbors_used"], 3)
        dedup = q["modes"]["SEM_DUPLICATAS_TOP5"]
        self.assertEqual(dedup["excluded_query_signature_copies"], 1)
        self.assertTrue(all(
            n["path"] not in {"q", "exact_copy"}
            for n in dedup["neighbors"]
        ))

    def test_neighbor_duplicates_count_once_in_dedup_mode(self):
        rows = [
            entry("q", "OK", [.4, .2, .1, .0]),
            entry("copy1", "OK", [.45, .2, .1, .0]),
            entry("copy2", "OK", [.45, .2, .1, .0]),
            entry("ng", "NG", [.9, .1, .0, .0]),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        q = next(c for c in audit["cases"] if c["path"] == "q")
        dedup = q["modes"]["SEM_DUPLICATAS_TOP5"]
        self.assertEqual(dedup["collapsed_neighbor_signature_copies"], 1)
        self.assertEqual(len(dedup["neighbors"]), 2)

    def test_balanced_mode_includes_both_classes_when_present(self):
        rows = self.sample()
        audit = audit_knn_ng_discrimination(
            self.root, records=rows, per_label=2
        )
        q = next(c for c in audit["cases"] if c["path"] == "ok_a")
        neighbors = q["modes"]["BALANCEADO_3_POR_CLASSE"]["neighbors"]
        self.assertEqual(sum(x["label"] == "OK" for x in neighbors), 2)
        self.assertEqual(sum(x["label"] == "NG" for x in neighbors), 2)
        self.assertEqual(q["modes"]["BALANCEADO_3_POR_CLASSE"]["neighbors_used"], 4)

    def test_class_absent_is_review_not_auto_ok(self):
        rows = [
            entry("q", "OK", [.1, .2, .3, .4]),
            entry("other", "OK", [.3, .2, .1, .4]),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        for row in audit["cases"]:
            for mode in ("BALANCEADO_SEM_DUPLICATAS", "BALANCEADO_COM_REVISAO"):
                self.assertEqual(row["modes"][mode]["status"],
                                 "REVISAR_CLASSE_AUSENTE")
        self.assertEqual(
            audit["comparisons"]["BALANCEADO_COM_REVISAO"]["review_OK"], 2
        )

    def test_no_other_events_not_a_neighbor(self):
        rows = [
            entry("q", "NG", [.3, .3, .1, .1], event="A"),
            entry("same_event", "OK", [.3, .3, .1, .1], event="A"),
            entry("other", "NG", [.8, .3, .1, .1], event="B"),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        q = next(c for c in audit["cases"] if c["path"] == "q")
        self.assertEqual(q["other_records_in_same_scope"], 1)
        self.assertTrue(all(
            n["path"] != "same_event"
            for n in q["modes"]["BASELINE_TOP5"]["neighbors"]
        ))

    def test_opposite_labels_same_signature_cannot_be_approved(self):
        rows = [
            entry("q", "NG", [.3, .2, .2, .2]),
            entry("same_ok", "OK", [.3, .2, .2, .2]),
            entry("other_ng", "NG", [.4, .7, .2, .2]),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        q = next(c for c in audit["cases"] if c["path"] == "q")
        self.assertEqual(
            q["modes"]["BALANCEADO_COM_REVISAO"]["status"],
            "REVISAR_CONFLITO_ASSINATURA"
        )
        self.assertEqual(
            q["modes"]["BALANCEADO_COM_REVISAO"]["prediction"], None
        )

    def test_review_does_not_improve_auto_recall_by_faking_ng(self):
        rows = self.sample()
        audit = audit_knn_ng_discrimination(
            self.root, records=rows, min_similarity=1
        )
        m = audit["comparisons"]["BALANCEADO_COM_REVISAO"]
        self.assertEqual(
            m["expected_NG"],
            m["correct_NG"] + m["missed_NG_as_OK"] + m["review_NG"]
        )
        self.assertEqual(
            m["expected_OK"],
            m["correct_OK"] + m["false_NG_on_OK"] + m["review_OK"]
        )
        self.assertEqual(
            m["automatic_decisions"],
            m["correct_NG"] + m["correct_OK"]
            + m["missed_NG_as_OK"] + m["false_NG_on_OK"]
        )

    def test_isolated_categories_and_lighting_never_mix(self):
        rows = [
            entry("a", "OK", [.5, .2, .1, .1], category="FALTANDO"),
            entry("b", "NG", [.5, .2, .1, .1], category="EMBORCADO"),
            entry("c", "NG", [.5, .2, .1, .1], category="FALTANDO",
                  light="MID"),
        ]
        audit = audit_knn_ng_discrimination(self.root, records=rows)
        self.assertTrue(all(
            row["modes"]["BASELINE_TOP5"]["status"] == "REVISAR_SEM_VIZINHOS"
            for row in audit["cases"]
        ))

    def test_input_validation_without_model_or_network(self):
        rows = self.sample()
        with patch(
            "src.core.experts.knn_expert.KNNExpert.__init__",
            side_effect=AssertionError("Não instanciar KNN nem baixar modelos"),
        ):
            audit_knn_ng_discrimination(self.root, records=rows)
        with self.assertRaises(ValueError):
            audit_knn_ng_discrimination(self.root, records=rows, review_margin=.5)
        with self.assertRaises(ValueError):
            audit_knn_ng_discrimination(self.root, records=rows, per_label=0)
        with self.assertRaises(ValueError):
            audit_knn_ng_discrimination(
                self.root, records=[rows[0], rows[0]]
            )

    def test_produces_separate_reports_without_touching_dataset(self):
        audit = audit_knn_ng_discrimination(
            self.root, records=self.sample()
        )
        target = self.root / "reports" / "startup_regression"
        json_path, txt_path = write_discrimination_reports(audit, target)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], audit["schema"])
        self.assertFalse(payload["knn_runtime_modified"])
        self.assertIn("NG liberados incorretamente",
                      txt_path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            write_discrimination_reports(audit, self.root / "public" / "dataset")


if __name__ == "__main__":
    unittest.main()
