"""Gate: candidatos que recuperam NG mas rejeitam OK devem ser REPROVADOS."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.services.faltando_cnn_regression_triage import (
    SCHEMA, triage_comparison, triage_report,
)


def case(path, category, light, label, before, after):
    return {
        "source_path": f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/{path}",
        "category": category, "lighting_mode": light,
        "expected_label": label, "baseline_status": before,
        "candidate_status": after,
        "baseline_ng_score": 0.001 if label == "OK" else 0.01,
        "candidate_ng_score": 0.98 if after == "ERRO_CLASSIFICACAO" else 0.66,
    }


def model(*, passed, incorrect, review, ng_as_ok, ok_as_ng, eligible=211):
    return {
        "eligible": eligible, "passed": passed, "incorrect": incorrect,
        "review": review, "invalid": 0, "no_coverage": 0,
        "not_evaluated": 0, "ng_as_ok": ng_as_ok,
        "ok_as_ng": ok_as_ng,
    }


def failing_report():
    changes = []
    for ix in range(3):
        changes.append(case(
            f"ng_{ix}_FALTANDO_MID.png", "FALTANDO", "MID", "NG",
            "ERRO_CLASSIFICACAO", "PASSOU",
        ))
    for ix in range(2):
        changes.append(case(
            f"ng_review_{ix}_FALTANDO_MID.png", "FALTANDO", "MID", "NG",
            "ERRO_CLASSIFICACAO", "REVISAO",
        ))
    for ix in range(6):
        changes.append(case(
            f"ok_{ix}_DESLOCADO_SIDE.png", "DESLOCADO", "SIDE", "OK",
            "PASSOU", "ERRO_CLASSIFICACAO",
        ))
    for ix in range(2):
        changes.append(case(
            f"ok_review_{ix}_DESLOCADO_SIDE.png", "DESLOCADO", "SIDE", "OK",
            "PASSOU", "REVISAO",
        ))
    changes.append(case(
        "ok_legacy_review_DESLOCADO_TOP.png", "DESLOCADO", "TOP", "OK",
        "REVISAO", "ERRO_CLASSIFICACAO",
    ))
    return {
        "schema": "visionx.faltando_candidate_archive_comparison.v1",
        "mode": "READ_ONLY_FULL_ARCHIVE_REPLAY_COMPARISON",
        "candidate_only": True, "active_model_changed": False,
        "production_approved": False, "holdout_independent": False,
        "comparison": {
            "baseline": model(passed=205, incorrect=5, review=1,
                              ng_as_ok=5, ok_as_ng=0),
            "candidate": model(passed=200, incorrect=7, review=4,
                               ng_as_ok=0, ok_as_ng=7),
            "ng_false_ok_baseline": 5,
            "ng_false_ok_candidate": 0,
            "ok_false_ng_baseline": 0,
            "ok_false_ng_candidate": 7,
            "reviews_baseline": 1,
            "reviews_candidate": 4,
            "case_changes": changes,
        },
    }


class TriageTests(unittest.TestCase):
    def test_recovers_five_ng_but_blocks_seven_new_ok_false_ng(self):
        result = triage_comparison(failing_report())
        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(result["candidate_rejected_by_archive_gate"])
        self.assertEqual(result["new_ok_false_ng"], 7)
        self.assertEqual(result["new_ng_false_ok"], 0)
        self.assertEqual(result["new_reviews_from_previously_passed"], 2)
        self.assertEqual(result["recovered_ng_false_ok_as_ng_or_review"], 5)
        self.assertEqual(result["required_human_review_count"], 9)
        self.assertEqual(set(result["changes_by_category"]), {"FALTANDO", "DESLOCADO"})
        self.assertFalse(result["regression_checks"]["no_new_ok_false_ng"])
        self.assertFalse(result["regression_checks"]["no_new_review_for_previously_passed"])
        simulated = result["category_routing_counterfactual"]
        self.assertEqual(simulated["mode"], "COUNTERFACTUAL_ONLY_NO_INFERENCE")
        self.assertEqual(simulated["candidate_categories"], ["FALTANDO", "INVERTIDO"])
        self.assertEqual(simulated["projected_counts"]["passed"], 208)
        self.assertEqual(simulated["projected_counts"]["incorrect"], 0)
        self.assertEqual(simulated["projected_counts"]["review"], 3)
        self.assertEqual(simulated["projected_counts"]["ng_as_ok"], 0)
        self.assertEqual(simulated["projected_counts"]["ok_as_ng"], 0)
        self.assertFalse(simulated["operationally_evaluated"])
        self.assertFalse(simulated["routing_deployed"])
        self.assertTrue(simulated["requires_category_ocr_and_routing_validation"])
        self.assertFalse(result["independent_holdout_evaluated"])
        self.assertFalse(result["training_uses_archive_labels"])
        for row in result["required_human_review"]:
            self.assertEqual(row["human_verification"], "PENDING")
            self.assertFalse(row["training_authorized"])
            self.assertIsNone(row["verified_physical_event_id"])

    def test_unmodified_good_checkpoint_can_only_go_to_independent_testing(self):
        report = failing_report()
        report["comparison"]["baseline"] = model(
            passed=211, incorrect=0, review=0, ng_as_ok=0, ok_as_ng=0
        )
        report["comparison"]["candidate"] = copy.deepcopy(
            report["comparison"]["baseline"]
        )
        report["comparison"].update({
            "ng_false_ok_baseline": 0, "ng_false_ok_candidate": 0,
            "ok_false_ng_baseline": 0, "ok_false_ng_candidate": 0,
            "reviews_baseline": 0, "reviews_candidate": 0,
            "case_changes": [],
        })
        result = triage_comparison(report)
        self.assertTrue(result["regression_gate_passed"])
        self.assertTrue(result["can_consider_for_independent_testing"])
        self.assertFalse(result["production_approved"])

    def test_detects_tampered_case_source_and_counter(self):
        report = failing_report()
        report["comparison"]["case_changes"][0]["expected_label"] = "OK"
        with self.assertRaisesRegex(ValueError, "Pasta"):
            triage_comparison(report)
        report = failing_report()
        report["comparison"]["candidate"]["ok_as_ng"] = 6
        with self.assertRaisesRegex(ValueError, "divergem"):
            triage_comparison(report)

    def test_existing_report_creates_only_a_readonly_triage_in_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            folder = root / "reports/startup_regression/cnn_candidate_validation"
            folder.mkdir(parents=True)
            original = folder / "comparacao_acervo_cnn_peso050_test.json"
            original.write_text(json.dumps(failing_report()), encoding="utf-8")
            before = original.read_bytes()
            output, result = triage_report(root, original)
            self.assertTrue(output.is_file())
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(result["required_human_review_count"], 9)
            self.assertFalse(json.loads(output.read_text())["production_approved"])
            self.assertEqual(
                sorted(x.name for x in folder.iterdir()),
                sorted((original.name, output.name)),
            )


if __name__ == "__main__":
    unittest.main()
