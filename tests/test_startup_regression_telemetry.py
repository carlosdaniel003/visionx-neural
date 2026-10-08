"""Telemetria do replay SIDE: somente observação, sem ajustar a decisão."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.services.startup_regression.replay_telemetry import (
    aggregate_replay_cases,
    decision_snapshot,
    geometry_snapshot,
)
from src.services.startup_regression.side_replay import human_summary


def _analysis():
    return {
        "verdict": "DEFEITO REAL",
        "reason": "Divergência estrutural 19% | Presença suspeita",
        "bounding_box": (25, 30, 40, 35),
        "all_boxes": {"silk": (20, 22, 60, 50)},
        "detail": {
            "missing_active": True,
            "missing_score": 0.67,
            "missing_tolerance": 0.42,
            "missing_is_defect": True,
            "silk_error_pct": np.float64(0.19),
            "semantic_loss": np.float32(0.27),
            "local_score": 0.5,
            "ctx_score": 0.3,
            "ssim": 0.76,
            "missing_bounding_box": (12, 15, 70, 68),
            "anomaly_signature": {"not_permitted": np.zeros((600, 800))},
            "component_missing_mask": np.zeros((600, 800), dtype=np.uint8),
            "decision_trace": {
                "physical_score": 0.88,
                "final_score": 0.88,
                "cutoff": 0.45,
                "physical_defect": True,
                "dominant_engine": "missing",
                "fusion_rule": "physical_only",
                "weights": {"physical": 1.0, "knn": 0.0},
                "engines": [
                    {
                        "id": "structural",
                        "label": "Comparador estrutural",
                        "active": True,
                        "triggered": True,
                        "selected": False,
                        "raw_score": 0.19,
                        "effective_score": 0.85,
                        "threshold": 0.08,
                        "final_influence": 0.0,
                        "summary": "Divergência estrutural 19%",
                    },
                    {
                        "id": "missing",
                        "label": "Presença do componente",
                        "active": True,
                        "triggered": True,
                        "selected": True,
                        "raw_score": 0.67,
                        "effective_score": 0.88,
                        "threshold": 0.42,
                        "final_influence": 0.88,
                        "summary": "Presença suspeita",
                    },
                    {
                        "id": "knn",
                        "active": False,
                        "triggered": False,
                        "effective_score": 0.0,
                    },
                ],
            },
        },
    }


class ReplayTelemetryTests(unittest.TestCase):
    def test_trace_preserves_exact_raw_threshold_effective_and_influence(self):
        analysis = _analysis()
        result = decision_snapshot(analysis)
        self.assertEqual(result["schema"], "visionx.side_replay_telemetry.v1")
        self.assertEqual(result["dominant_engine"], "missing")
        self.assertEqual(result["cutoff"], 0.45)
        self.assertEqual(len(result["engines"]), 2)
        self.assertEqual(result["engines"][0]["raw_score"], 0.19)
        self.assertEqual(result["engines"][0]["threshold"], 0.08)
        self.assertEqual(result["engines"][0]["effective_score"], 0.85)
        self.assertEqual(result["engines"][1]["final_influence"], 0.88)
        self.assertEqual(result["physical_readings"]["missing_score"], 0.67)
        self.assertEqual(result["physical_readings"]["silk_error_pct"], 0.19)

    def test_does_not_serialize_knn_memory_or_full_image_arrays(self):
        result = decision_snapshot(_analysis())
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("anomaly_signature", encoded)
        self.assertNotIn("component_missing_mask", encoded)
        self.assertNotIn("not_permitted", encoded)
        self.assertNotIn('"id": "knn"', encoded)
        self.assertFalse(result["memory_consulted"])

    def test_geometry_keeps_full_aoi_and_focus_without_cropping(self):
        analysis = _analysis()
        reference = np.zeros((300, 400, 3), dtype=np.uint8)
        test = np.zeros((299, 400, 3), dtype=np.uint8)
        context = {
            "raw_anomalies": [(100, 110, 12, 14)],
            "old_epicenters": [(10, 15, 30, 35)],
            "real_epicenters": [(12, 17, 25, 29)],
            "global_box_info": {"x": 4, "y": 6, "w": 270, "h": 220, "detected": True},
        }
        before_ref = reference.copy()
        result = geometry_snapshot(context, analysis, reference, test)
        self.assertEqual(result["reference_width"], 400)
        self.assertEqual(result["reference_height"], 300)
        self.assertEqual(result["test_height"], 299)
        self.assertFalse(result["input_dimensions_equal"])
        self.assertEqual(result["raw_anomaly_boxes"], [[100, 110, 12, 14]])
        self.assertEqual(result["selected_epicenter_boxes"], [[12, 17, 25, 29]])
        self.assertEqual(result["global_box"]["w"], 270)
        self.assertEqual(result["specialist_boxes"]["silk"], [20, 22, 60, 50])
        self.assertEqual(result["final_bounding_box"], [25, 30, 40, 35])
        np.testing.assert_array_equal(reference, before_ref)

    def test_invalid_values_are_json_safe_without_inventing_score(self):
        ana = _analysis()
        ana["detail"]["semantic_loss"] = float("nan")
        ana["detail"]["decision_trace"]["engines"][0]["raw_score"] = float("inf")
        report = decision_snapshot(ana)
        self.assertIsNone(report["engines"][0]["raw_score"])
        self.assertNotIn("semantic_loss", report["physical_readings"])
        json.dumps(report, allow_nan=False)

    def test_summary_compares_ok_ng_by_category_and_triggered_engines(self):
        cases = [
            {
                "expected_label": "OK", "category": "FALTANDO",
                "status": "REGRESSAO", "telemetry": decision_snapshot(_analysis()),
            },
            {
                "expected_label": "NG", "category": "FALTANDO",
                "status": "PASSOU", "telemetry": decision_snapshot(_analysis()),
            },
            {
                "expected_label": "OK", "category_hint": "INVERTIDO",
                "status": "INVALIDO", "telemetry": None,
            },
        ]
        aggregate = aggregate_replay_cases(cases)
        self.assertEqual(len(aggregate["by_category_and_label"]), 3)
        self.assertEqual(
            aggregate["by_label_status_fusion_rule"]["OK|REGRESSAO|physical_only"], 1
        )
        self.assertEqual(
            aggregate["by_label_status_fusion_rule"]["NG|PASSOU|physical_only"], 1
        )
        self.assertEqual(
            aggregate["triggered_engines_in_regressions_by_category"][
                "FALTANDO|missing"
            ],
            1,
        )
        self.assertNotIn(
            "INVERTIDO|missing",
            aggregate["triggered_engines_in_regressions_by_category"],
        )

    def test_txt_includes_failed_and_passed_specialist_evidence(self):
        template = {
            "source_path": "public/ok_archive/legacy.png",
            "expected_label": "OK", "expected_verdict": "FALHA FALSA",
            "category": "FALTANDO", "status": "REGRESSAO",
            "verdict": "DEFEITO REAL", "error": None,
            "telemetry": {
                **decision_snapshot(_analysis()),
                "geometry": geometry_snapshot(
                    {}, _analysis(),
                    np.zeros((120, 180, 3), dtype=np.uint8),
                    np.zeros((120, 180, 3), dtype=np.uint8),
                ),
            },
        }
        ok = dict(template)
        ok.update({
            "source_path": "public/ng_archive/legacy_ng.png",
            "expected_label": "NG", "expected_verdict": "DEFEITO REAL",
            "status": "PASSOU",
        })
        cases = [template, ok]
        report = {
            "root": "test",
            "multilight_explicit_deferred": 0,
            "summary": {
                "total": 2, "passed": 1, "regressions": 1, "invalid": 0,
                "by_label": {
                    "OK": {"total": 1, "passed": 0, "regressions": 1, "invalid": 0},
                    "NG": {"total": 1, "passed": 1, "regressions": 0, "invalid": 0},
                },
            },
            "diagnostics": aggregate_replay_cases(cases),
            "cases": cases,
        }
        txt = human_summary(report)
        self.assertIn("RESUMO POR RÓTULO / CATEGORIA", txt)
        self.assertIn("TELEMETRIA INDIVIDUAL — TODOS OS CASOS", txt)
        self.assertIn("bruto=0.1900", txt)
        self.assertIn("limite=0.0800", txt)
        self.assertIn("influencia=0.8800", txt)
        self.assertIn("epicentro=NENHUM", txt)
        self.assertIn("[PASSOU] public/ng_archive/legacy_ng.png", txt)
        self.assertIn("[REGRESSAO] public/ok_archive/legacy.png", txt)

    def test_existing_verdict_remains_untouched_by_telemetry(self):
        analysis = _analysis()
        previous = analysis["verdict"]
        decision_snapshot(analysis)
        self.assertEqual(analysis["verdict"], previous)
        self.assertEqual(analysis["detail"]["decision_trace"]["final_score"], 0.88)


if __name__ == "__main__":
    unittest.main()
