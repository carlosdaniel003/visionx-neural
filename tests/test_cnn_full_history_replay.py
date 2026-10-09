"""Regressão CNN de 100% dos PNGs; nem revisão nem ausência vira acerto."""
import tempfile
import unittest
from pathlib import Path
import json

import cv2
import numpy as np

from src.services.startup_regression.cnn_full_history_replay import (
    replay_full_cnn_history, UNSUPPORTED,
)
from src.services.startup_regression.cnn_full_history_replay_cli import (
    write_full_history_cnn_report,
)
from src.services.startup_regression.archive_inventory import inventory_archives


class SimulatedCNN:
    """Apenas teste de contrato; nunca uma rede treinada."""
    def __init__(self, *, wrong=None, review=None, unverified=None):
        self.wrong = set(wrong or ())
        self.review = set(review or ())
        self.unverified = set(unverified or ())
        self.calls = []

    def inspect(self, reference, test, mode):
        pixel = int(test[0, 0, 0])
        self.calls.append((pixel, mode))
        defect = pixel >= 100
        if pixel in self.wrong:
            defect = not defect
        score = .98 if defect else .02
        verdict = (
            "REVISÃO OBRIGATÓRIA" if pixel in self.review
            else "DEFEITO REAL" if defect else "FALHA FALSA"
        )
        return {
            "verdict": verdict,
            "production_review_required": pixel in self.review,
            "detail": {
                "cnn_v2_active": True,
                "cnn_v2_checkpoint_verified": pixel not in self.unverified,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_checkpoint_sha256": "c" * 64,
                "cnn_v2_ng_score_uncalibrated": score,
            },
        }


class FullHistoryReplayTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.ok = self.root / "public" / "ok_archive"
        self.ng = self.root / "public" / "ng_archive"
        self.ok.mkdir(parents=True)
        self.ng.mkdir(parents=True)
        self.model = SimulatedCNN()

    def png(self, label, name, marker):
        success, blob = cv2.imencode(
            ".png", np.full((60, 70, 3), marker, dtype=np.uint8)
        )
        self.assertTrue(success)
        path = (self.ok if label == "OK" else self.ng) / name
        path.write_bytes(blob.tobytes())
        return path

    def extractor(self, image):
        p = int(image[0, 0, 0])
        categories = {
            51: "FALTANDO", 151: "FALTANDO",
            55: "DESLOCADO", 56: "INVERTIDO",
            58: "MUITO ADESIVO", 159: "EMBORCADO",
            61: "FALTANDO",
        }
        category = categories.get(p, "FALTANDO")
        return image.copy(), image.copy(), {
            "board": "B1", "parts": "U2~5", "value": category,
        }

    def run_replay(self, **kwargs):
        return replay_full_cnn_history(
            self.root, extractor=self.extractor,
            predictors={"CNN_FALTANDO_V2": self.model}, **kwargs,
        )

    def test_all_pngs_counted_including_category_without_cnn(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        self.png("NG", "2026-10-09_1001_FALTANDO_SIDE.png", 151)
        self.png("OK", "2026-10-09_1002_MUITO_ADESIVO_SIDE.png", 58)
        report = self.run_replay()
        s = report["overall"]
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["cnn_scoped"], 2)
        self.assertEqual(s["passed"], 2)
        self.assertEqual(s["unsupported"], 1)
        self.assertAlmostEqual(s["historical_full_archive_match_rate"], 2 / 3, 6)
        self.assertEqual(s["NG_as_NG"], 1)
        self.assertFalse(s["historical_98pct_target_met"])
        self.assertFalse(report["knn_used"])
        self.assertFalse(report["cnn_training_performed"])
        self.assertFalse(report["production_approved"])
        self.assertEqual(len(self.model.calls), 2)

    def test_review_with_correct_raw_binary_is_not_considered_passed(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        self.model = SimulatedCNN(review={51})
        report = self.run_replay()
        s = report["overall"]
        self.assertEqual(s["review"], 1)
        self.assertEqual(s["passed"], 0)
        self.assertEqual(s["raw_binary_correct"], 1)
        self.assertFalse(s["historical_98pct_target_met"])

    def test_missed_ng_and_false_alarm_both_fail(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        self.png("NG", "2026-10-09_1001_FALTANDO_SIDE.png", 151)
        self.model = SimulatedCNN(wrong={51, 151})
        report = self.run_replay()
        s = report["overall"]
        self.assertEqual(s["NG_as_OK"], 1)
        self.assertEqual(s["OK_as_NG"], 1)
        self.assertEqual(s["passed"], 0)
        self.assertEqual(s["regression"], 2)

    def test_complete_archive_cnn_at_98_percent_satisfies_historical_target(self):
        for i in range(50):
            marker = 51 if i == 0 else 61
            self.png(
                "OK", f"2026-10-09_{1000+i:04d}_FALTANDO_SIDE.png",
                marker if i % 2 else 51,
            )
        # Usar casos com o mesmo pixel é permitido apenas na mesma classe.
        report = self.run_replay()
        self.assertEqual(report["overall"]["total"], 50)
        self.assertEqual(report["overall"]["passed"], 50)
        self.assertTrue(report["overall"]["historical_98pct_target_met"])
        self.assertFalse(report["production_approved"])

    def test_unverified_checkpoint_never_passes_or_produces_fake_ok(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        self.model = SimulatedCNN(unverified={51})
        result = self.run_replay()
        self.assertEqual(result["overall"]["invalid"], 1)
        self.assertEqual(result["overall"]["passed"], 0)
        self.assertIsNone(result["cases"][0]["verdict"])

    def test_ocr_category_mismatch_blocks_false_success(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 55)
        result = self.run_replay()
        self.assertEqual(result["overall"]["invalid"], 1)
        self.assertEqual(self.model.calls, [])

    def test_ng_adesivo_without_specialized_cnn_is_not_approved(self):
        self.png("NG", "2026-10-09_1000_MUITO_ADESIVO_SIDE.png", 58)
        result = self.run_replay()
        self.assertEqual(result["overall"]["unsupported"], 1)
        self.assertEqual(result["cases"][0]["status"], UNSUPPORTED)
        self.assertFalse(result["overall"]["historical_98pct_target_met"])

    def test_no_relabeling_from_json_memory_or_knn(self):
        image = self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        original = image.read_bytes()
        report = self.run_replay()
        self.assertEqual(image.read_bytes(), original)
        self.assertEqual(report["cases"][0]["expected_label"], "OK")
        self.assertEqual(report["overall"]["passed"], 1)

    def test_explicit_three_light_manifest_is_audited_but_not_fabricated(self):
        filenames = [
            "2026-10-09_1000_FALTANDO_SIDE.png",
            "2026-10-09_1000_FALTANDO_TOP.png",
            "2026-10-09_1000_FALTANDO_MID.png",
        ]
        for name in filenames:
            self.png("OK", name, 51)
        inventory = inventory_archives(self.root)
        for row in inventory["images"]:
            row["event_id"] = "event-confirmed"
            row["manifest_links"] = [{
                "event_id": "event-confirmed",
                "lighting_mode": row["lighting_mode"],
                "manifest_path": "synthetic_fixture_manifest.json",
            }]
        report = self.run_replay(inventory=inventory)
        events = report["multilight_explicit_events"]
        self.assertEqual(events["events_with_explicit_manifest"], 1)
        self.assertEqual(events["status_counts"], {"PASSOU_3_LUZES": 1})

    def test_shared_png_does_not_hide_two_real_manifest_events(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        for minute in ("1000", "1001"):
            for light in ("TOP", "MID"):
                self.png("OK", f"2026-10-09_{minute}_FALTANDO_{light}.png", 51)
        inventory = inventory_archives(self.root)
        for row in inventory["images"]:
            filename = Path(row["path"]).name
            if filename.endswith("_SIDE.png"):
                events = ("e1", "e2")
                row["event_id"] = None
            else:
                events = ("e1",) if "_1000_" in filename else ("e2",)
                row["event_id"] = events[0]
            row["manifest_links"] = [{
                "event_id": event,
                "lighting_mode": row["lighting_mode"],
                "manifest_path": f"{event}.json",
            } for event in events]
        report = self.run_replay(inventory=inventory)
        events = report["multilight_explicit_events"]
        self.assertEqual(events["events_with_explicit_manifest"], 2)
        self.assertEqual(events["status_counts"], {"PASSOU_3_LUZES": 2})
        self.assertTrue(report["overall"]["explicit_event_integrity_passed"])
        self.assertEqual(report["by_model"]["CNN_FALTANDO_V2"]["total"], 5)

    def test_shared_png_assigned_to_another_light_is_not_false_three_light_pass(self):
        for minute in ("1000", "1001", "1002"):
            self.png("OK", f"2026-10-09_{minute}_FALTANDO_SIDE.png", 51)
        inventory = inventory_archives(self.root)
        for row, light in zip(inventory["images"], ("SIDE", "TOP", "MID")):
            row["event_id"] = "e1"
            row["manifest_links"] = [{
                "event_id": "e1",
                "lighting_mode": light,
                "manifest_path": "e1.json",
            }]
        report = self.run_replay(inventory=inventory)
        self.assertEqual(report["overall"]["passed"], 3)
        self.assertEqual(
            report["multilight_explicit_events"]["status_counts"],
            {"ILUMINACAO_CNN_NAO_VERIFICADA": 1},
        )
        self.assertFalse(report["overall"]["historical_98pct_target_met"])

    def test_unqualified_manifest_blocks_historical_target(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        inventory = inventory_archives(self.root)
        inventory["manifests"].append({
            "path": "public/ok_archive/bad.json",
            "status": "NEEDS_QUALIFICATION",
        })
        report = self.run_replay(inventory=inventory)
        self.assertEqual(report["overall"]["passed"], 1)
        self.assertFalse(report["overall"]["manifest_integrity_passed"])
        self.assertFalse(report["overall"]["historical_98pct_target_met"])

    def test_duplicate_manifest_event_ids_block_historical_target(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        inventory = inventory_archives(self.root)
        inventory["issues"].append({"code": "DUPLICATE_EVENT_ID"})
        report = self.run_replay(inventory=inventory)
        self.assertEqual(report["overall"]["passed"], 1)
        self.assertFalse(report["overall"]["manifest_integrity_passed"])
        self.assertFalse(report["overall"]["historical_98pct_target_met"])

    def test_three_light_event_with_different_checkpoints_is_not_passed(self):
        for light in ("SIDE", "TOP", "MID"):
            self.png("OK", f"2026-10-09_1000_FALTANDO_{light}.png", 51)
        inventory = inventory_archives(self.root)
        for row in inventory["images"]:
            row["event_id"] = "event-confirmed"
            row["manifest_links"] = [{
                "event_id": "event-confirmed",
                "lighting_mode": row["lighting_mode"],
                "manifest_path": "valid_manifest.json",
            }]
        baseline = self.model.inspect
        def different_model(ref, test, mode):
            output = baseline(ref, test, mode)
            output["detail"]["cnn_v2_checkpoint_sha256"] = (
                "d" * 64 if mode == "MID" else "c" * 64
            )
            return output
        self.model.inspect = different_model
        report = self.run_replay(inventory=inventory)
        self.assertEqual(
            report["multilight_explicit_events"]["status_counts"],
            {"CHECKPOINTS_DIFERENTES_NO_EVENTO": 1}
        )
        self.assertEqual(report["overall"]["passed"], 3)
        self.assertFalse(report["overall"]["explicit_event_integrity_passed"])
        self.assertFalse(report["overall"]["historical_98pct_target_met"])
        self.assertFalse(report["production_approved"])

    def test_98_percent_boundary_and_two_errors_in_fifty(self):
        for i in range(50):
            self.png("OK", f"2026-10-09_{1000+i:04d}_FALTANDO_SIDE.png", 51)
        inventory = inventory_archives(self.root)
        original = self.model.inspect
        calls = [0]
        def one_bad(ref, test, light):
            calls[0] += 1
            result = original(ref, test, light)
            if calls[0] in {1, 2}:
                result["verdict"] = "DEFEITO REAL"
                result["detail"]["cnn_v2_ng_score_uncalibrated"] = .98
            return result
        self.model.inspect = one_bad
        replay = self.run_replay(inventory=inventory)
        self.assertEqual(replay["overall"]["passed"], 48)
        self.assertFalse(replay["overall"]["historical_98pct_target_met"])
        calls[0] = 0
        def one_error(ref, test, light):
            calls[0] += 1
            result = original(ref, test, light)
            if calls[0] == 1:
                result["verdict"] = "DEFEITO REAL"
                result["detail"]["cnn_v2_ng_score_uncalibrated"] = .98
            return result
        self.model.inspect = one_error
        replay = self.run_replay(inventory=inventory)
        self.assertEqual(replay["overall"]["passed"], 49)
        self.assertTrue(replay["overall"]["historical_98pct_target_met"])

    def test_unlinked_three_similar_images_not_assumed_one_event(self):
        for light in ("SIDE", "TOP", "MID"):
            self.png("OK", f"2026-10-09_1000_FALTANDO_{light}.png", 51)
        report = self.run_replay()
        self.assertEqual(
            report["multilight_explicit_events"]["events_with_explicit_manifest"], 0
        )
        self.assertEqual(report["overall"]["total"], 3)

    def test_corrupt_file_remains_denominator_and_not_success(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        bad = self.ok / "2026-10-09_1001_INVERTIDO_SIDE.png"
        bad.write_bytes(b"not a png")
        report = self.run_replay()
        self.assertEqual(report["overall"]["total"], 2)
        self.assertEqual(report["overall"]["passed"], 1)
        self.assertEqual(report["overall"]["invalid"], 1)

    def test_reports_only_under_reports_folder(self):
        self.png("OK", "2026-10-09_1000_FALTANDO_SIDE.png", 51)
        report = self.run_replay()
        output = self.root / "reports" / "startup_regression"
        json_path, txt_path = write_full_history_cnn_report(report, output)
        self.assertEqual(
            json.loads(json_path.read_text(encoding="utf-8"))["schema"],
            report["schema"],
        )
        self.assertIn("NG históricos", txt_path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            write_full_history_cnn_report(report, self.ok)


if __name__ == "__main__":
    unittest.main()
