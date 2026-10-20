"""KNN PNG-only: testa exclusão, contagens por classe e não vazamento."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.services.startup_regression.knn_archive_visual_replay import (
    replay_knn_archive_visual, _excluded,
)
from src.services.startup_regression.knn_archive_visual_replay_cli import (
    write_knn_archive_reports,
)


def item(date: str, time: str, label: str, kind: str) -> dict:
    name = f"{date}_{time}_FALTANDO_SIDE.png"
    return {
        "path": f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/{name}",
        "expected_label": label, "category_hint": "FALTANDO",
        "lighting_mode": "SIDE", "lighting_source": "EXPLICIT_SUFFIX",
        "status": "VALID_PNG",
        "file_sha256": f"file-{date}-{time}-{kind}",
        "pixel_sha256": f"pixel-{date}-{time}-{kind}",
        "event_id": None,
    }


def inventory(rows, *, conflicts=()):
    return {
        "schema": "visionx.archive_inventory.v1",
        "summary": {"png_count": len(rows)},
        "images": rows,
        "cross_label_conflicts": list(conflicts),
    }


def extract(kind: str):
    reference = np.zeros((64, 64, 3), dtype=np.uint8)
    test = reference.copy()
    if kind == "OK":
        test[3:12, 4:13, :] = 90
    elif kind == "NG":
        test[37:55, 36:53, :] = 200
    else:
        test[18:28, 12:19, :] = 140
    info = {
        "value": "0 <= 20 <= 80 FALTANDO", "category": "FALTANDO",
        "board": "B1", "parts": "U2",
    }
    return reference, test, info


def runner(rows, *, protocol="visual_holdout", k=1, conflicts=(),
           extractor=extract, cancelled=None, progress=None,
           geometry_experiment=False):
    return replay_knn_archive_visual(
        Path("/not-a-real-odin-dataset"),
        inventory=inventory(rows, conflicts=conflicts),
        frame_loader=lambda _root, row: row["expected_label"],
        extractor=extractor,
        protocol=protocol, k=k, cancelled=cancelled, progress=progress,
        geometry_experiment=geometry_experiment,
    )


class KnnArchiveReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            item("2026-10-02", "0900", "OK", "OK"),
            item("2026-10-02", "1000", "NG", "NG"),
            item("2026-10-03", "0900", "OK", "OK"),
            item("2026-10-03", "1000", "NG", "NG"),
        ]

    def test_optional_geometry_ab_preserves_original_visual_results(self):
        visual = runner(self.rows)
        dual = runner(self.rows, geometry_experiment=True)
        self.assertEqual(visual["results"], dual["results"])
        self.assertEqual(visual["shadow_candidate"], dual["shadow_candidate"])
        self.assertIsNone(visual["geometry_experiment"])
        experiment = dual["geometry_experiment"]
        self.assertTrue(experiment["experimental_only"])
        self.assertFalse(experiment["production_approved"])
        self.assertEqual(experiment["case_count"], 4)
        self.assertEqual(experiment["geometry_failures"], 0)
        self.assertEqual(experiment["results"]["total"], 4)
        self.assertEqual(experiment["shadow_candidate"]["total"], 4)
        self.assertEqual(experiment["consensus_candidate"]["total"], 4)
        self.assertFalse(experiment["consensus_candidate"]["production_approved"])
        self.assertEqual(
            sum(experiment["consensus_source_counts"].values()), 4,
        )
        self.assertIn("FALTANDO", experiment["consensus_by_category"])
        self.assertEqual(experiment["consensus_by_lighting"]["SIDE"]["total"], 4)
        self.assertEqual(len(dual["cases"]), 4)
        self.assertFalse(dual["writes_memory_or_dataset"])
        for case in dual["cases"]:
            self.assertIn("geometry_experiment", case)
            self.assertIn("shadow_candidate", case["geometry_experiment"])
            self.assertIn("geometry_consensus_candidate", case)
            self.assertFalse(case["geometry_consensus_candidate"]["auto_decision_authorized"])
            self.assertFalse(case["geometry_consensus_candidate"]["uses_expected_label"])
            self.assertIsNone(case["geometry_consensus_candidate"]["xp_command"])

    def test_cancelled_geometry_ab_cannot_meet_97pct(self):
        calls = []
        result = runner(
            self.rows, geometry_experiment=True,
            progress=lambda *_: calls.append(True),
            cancelled=lambda: len(calls) >= 2,
        )
        self.assertFalse(result["geometry_experiment"]["complete"])
        self.assertFalse(
            result["geometry_experiment"]["shadow_candidate"]["target_97pct_met"]
        )
        self.assertFalse(
            result["geometry_experiment"]["consensus_candidate"]["target_97pct_met"]
        )
        self.assertEqual(result["geometry_experiment"]["results"]["total"], 4)

    def test_shadow_candidate_is_separate_from_baseline_and_never_commands_xp(self):
        report = runner(self.rows)
        self.assertIn("shadow_candidate", report)
        self.assertTrue(report["shadow_candidate"]["diagnostic_only"])
        self.assertFalse(report["shadow_candidate"]["production_approved"])
        self.assertEqual(report["results"]["correct"], 4)
        for case in report["cases"]:
            self.assertIn(
                case["shadow_candidate"]["status"],
                ("SUGERE_OK", "SUGERE_NG", "REVISAO"),
            )
            self.assertFalse(case["shadow_candidate"]["auto_decision_authorized"])

    def test_two_classes_match_by_visual_appearance_and_never_self(self):
        report = runner(self.rows)
        self.assertEqual(report["inventory_png_count"], 4)
        self.assertEqual(report["results"]["by_label"]["OK"]["correct"], 2)
        self.assertEqual(report["results"]["by_label"]["NG"]["correct"], 2)
        self.assertTrue(report["results"]["target_98pct_met"])
        self.assertFalse(report["uses_legacy_memory_json"])
        self.assertFalse(report["is_production_verified_exact_knn_test"])
        for case in report["cases"]:
            self.assertEqual(case["status"], "PASSOU")
            self.assertEqual(case["neighbors_count"], 1)
            self.assertTrue(case["neighbors"])
            self.assertNotEqual(
                case["neighbors"][0]["source_path"], case["source_path"]
            )
            self.assertFalse(report["date_used_for_neighbor_selection"])
            self.assertEqual(report["similarity_basis"],
                             "reference_vs_capture_global_outer_inner_epicenter")


    def test_same_day_independent_visual_examples_are_valid_neighbors(self):
        # Duas capturas de mesmo dia, mas não da mesma peça/evento.
        report = runner(self.rows[:2])
        self.assertEqual(report["results"]["status_counts"], {"REGRESSAO": 2})
        for case in report["cases"]:
            self.assertEqual(case["neighbors_count"], 1)
            self.assertEqual(case["source_path"].split("/")[-1][:10],
                case["neighbors"][0]["source_path"].split("/")[-1][:10])

    def test_visual_holdout_excludes_only_self_and_same_event(self):
        from src.services.startup_regression.knn_archive_visual_replay import _row
        a = item("2026-10-02", "0900", "OK", "A")
        b = item("2026-10-02", "0901", "OK", "B")
        self.assertFalse(_excluded(_row(a), _row(b)))
        self.assertTrue(_excluded(_row(a), _row(a)))
        b["event_id"] = "real-event-1"
        a["event_id"] = "real-event-1"
        self.assertTrue(_excluded(_row(a), _row(b)))

    def test_one_valid_image_has_no_coverage(self):
        report = runner([self.rows[0]])
        self.assertEqual(report["results"]["status_counts"],
                         {"SEM_COBERTURA": 1})
        self.assertEqual(report["results"]["by_label"]["OK"]["correct"], 0)

    def test_exact_duplicate_and_same_prefix_never_become_neighbors(self):
        q = self.rows[0]
        same = dict(self.rows[2], pixel_sha256=q["pixel_sha256"])
        from src.services.startup_regression.knn_archive_visual_replay import _row
        self.assertTrue(_excluded(_row(q), _row(same), "visual_holdout"))
        second_light = dict(
            self.rows[2],
            path="public/ok_archive/2026-10-02_0900_FALTANDO_TOP.png",
            lighting_mode="TOP",
            pixel_sha256="different-pixels", file_sha256="different-file",
        )
        self.assertTrue(_excluded(_row(q), _row(second_light), "visual_holdout"))

    def test_current_category_only_ocr_no_longer_marks_archive_invalid(self):
        def category_only(frame):
            reference, test, info = extract(frame)
            info["value"] = ""
            return reference, test, info

        report = runner(self.rows, extractor=category_only)
        self.assertEqual(report["results"]["by_label"]["OK"]["correct"], 2)
        self.assertEqual(report["results"]["by_label"]["NG"]["correct"], 2)
        self.assertTrue(report["complete"])
        self.assertEqual(report["by_capture_cohort"]["side_only"]["capture_groups"], 4)
        self.assertEqual(report["by_capture_cohort"]["three_lights"]["capture_groups"], 0)

    def test_three_lights_and_one_side_only_reported_separately(self):
        rows = []
        for date in ("2026-10-02", "2026-10-03"):
            for label in ("OK", "NG"):
                triple = item(date, "0900", label, label)
                triple["event_id"] = date + "-" + label
                rows.append(triple)
                for light in ("TOP", "MID"):
                    other = dict(triple)
                    other["path"] = triple["path"].replace("_SIDE.png", "_" + light + ".png")
                    other["lighting_mode"] = light
                    other["file_sha256"] += light
                    other["pixel_sha256"] += light
                    rows.append(other)
                # Outro evento com somente SIDE no mesmo dia.
                single = item(date, "1430", label, label)
                single["event_id"] = date + "-" + label + "-single"
                rows.append(single)

        report = runner(rows)
        triples = report["by_capture_cohort"]["three_lights"]
        singles = report["by_capture_cohort"]["side_only"]
        self.assertEqual(triples["capture_groups"], 4)
        self.assertEqual(singles["capture_groups"], 4)
        self.assertEqual(triples["results"]["total"], 12)
        self.assertEqual(singles["results"]["total"], 4)
        self.assertEqual(report["capture_grouping"]["manifest_verified_groups"], 8)
        self.assertTrue(report["results"]["target_98pct_met"])
        for case in report["cases"]:
            for neighbor in case["neighbors"]:
                self.assertNotEqual(neighbor["source_path"], case["source_path"])


    def test_cancellation_keeps_all_denominators_and_does_not_pass(self):
        called = []
        report = runner(
            self.rows,
            progress=lambda i, total, path: called.append(i),
            cancelled=lambda: len(called) >= 2,
        )
        self.assertTrue(report["canceled"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["results"]["total"], 4)
        self.assertFalse(report["results"]["target_98pct_met"])
        self.assertEqual(report["results"]["status_counts"]["NAO_ANALISADO"], 4)

    def test_conflicts_are_invalid_not_fake_accuracy(self):
        bad = [self.rows[0]["path"], self.rows[1]["path"]]
        report = runner(self.rows, conflicts=[{"paths": bad}])
        self.assertFalse(report["results"]["target_98pct_met"])
        self.assertEqual(report["results"]["status_counts"]["INVALIDO"], 2)
        self.assertEqual(report["results"]["by_label"]["NG"]["no_decision"], 1)

    def test_category_ocr_mismatch_is_invalid(self):
        def wrong_ocr(frame):
            ref, test, info = extract(frame)
            info["value"] = "0 <= 20 <= 80 INVERTIDO"
            return ref, test, info

        report = runner(self.rows, extractor=wrong_ocr)
        self.assertEqual(report["results"]["status_counts"]["INVALIDO"], 4)
        self.assertFalse(report["results"]["target_98pct_met"])

    def test_extra_ng_error_is_counted_separately(self):
        # Consulta NG com característica OK: nunca rescrever gabarito para passar.
        def misclassified(frame):
            if frame == "NG":
                return extract("OK")
            return extract(frame)

        # NG em dia sem nenhum NG de referência: vizinhos são somente OK.
        rows = [
            item("2026-10-02", "1000", "NG", "NG"),
            item("2026-10-03", "0900", "OK", "OK"),
            item("2026-10-04", "0900", "OK", "OK"),
        ]
        report = runner(rows, extractor=misclassified)
        self.assertFalse(report["results"]["target_98pct_met"])
        self.assertEqual(report["results"]["NG_as_OK"], 1)

    def test_writer_does_not_touch_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = runner(self.rows)
            report["root"] = str(root)
            json_path, txt_path = write_knn_archive_reports(
                report, root / "reports" / "startup_regression"
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(txt_path.is_file())
            self.assertIn("NG perigosamente como OK", txt_path.read_text("utf-8"))
            with self.assertRaisesRegex(ValueError, "reports"):
                write_knn_archive_reports(report, root / "public" / "ok_archive")

    def test_no_empty_dataset_or_invalid_k(self):
        with self.assertRaisesRegex(ValueError, "Inventário"):
            runner([])
        with self.assertRaisesRegex(ValueError, "k precisa"):
            runner(self.rows, k=0)


if __name__ == "__main__":
    unittest.main()
