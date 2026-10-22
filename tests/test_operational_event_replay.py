"""Testes para replay por evento: vínculos, rotas e segurança sem XP."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.multilight_fusion import fuse_multilight
from src.core.side_early_decision import build_side_early_decision
from src.services.startup_regression.operational_event_replay import (
    SCHEMA, _verified_frame, replay, save_report, verified_event_manifests,
)


def case(label="NG", event_id="piece-A", valid=True):
    names = {
        mode: f"public/{'ng' if label == 'NG' else 'ok'}_archive/"
              f"2026-10-22_0800_FALTANDO_{mode}.png"
        for mode in ("SIDE", "TOP", "MID")
    }
    fake_sha = "a" * 64
    manifest = {
        "event_id": event_id, "status": "LINKED" if valid else "INVALID_MANIFEST",
        "expected_label": label,
        "path": f"public/{label.lower()}_archive/event.json",
        "frame_paths": names,
    }
    imgs = [
        {
            "path": path, "expected_label": label,
            "status": "VALID_PNG", "file_sha256": fake_sha,
            "manifest_links": [
                {"event_id": event_id, "lighting_mode": mode}
            ],
        }
        for mode, path in names.items()
    ]
    return {
        "schema": "visionx.archive_inventory.v1", "images": imgs,
        "manifests": [manifest],
    }


def local(verdict="DEFEITO REAL", *, early=False):
    defect = verdict == "DEFEITO REAL"
    detail = {
        "final_score": .99 if defect else .01,
        "physical_score": 0.0,
        "decision_trace": {"operator_review_required": not early},
    }
    if early:
        detail.update({
            "recognition_route": "KNOWN_KNN",
            "recognition_match": "EXACT_PAIR",
            "recognition_memory_verified": True,
            "recognition_known_label": "NG" if defect else "OK",
        })
    return {
        "verdict": verdict, "is_defect": defect, "confidence": .99,
        "lighting_mode": "SIDE",
        "production_review_required": not early,
        "detail": detail,
    }


class FakeRuntime:
    def __init__(self, root, *, early=False, verdict="DEFEITO REAL"):
        self.early, self.verdict = early, verdict
        self.called = []
        self.strategy_c_warmup = {"status": "READY"}

    def inspect(self, image, light, category):
        self.called.append(light)
        obj = local(self.verdict, early=self.early)
        obj["lighting_mode"] = light
        return obj, "FALTANDO", {
            "lighting_mode": light,
            "sample_image": image, "test_image": image,
            "analysis": obj,
        }

    def finalize(self, analyses, samples, category, *, side_only):
        if side_only:
            return build_side_early_decision(analyses["SIDE"], category)
        return fuse_multilight(analyses, category)


class ManifestQualificationTests(unittest.TestCase):
    def test_only_manifest_linked_event_counts_as_a_piece(self):
        inv = case()
        linked, coverage = verified_event_manifests(inv)
        self.assertEqual(len(linked), 1)
        self.assertEqual(coverage["qualified_event_count"], 1)
        self.assertEqual(coverage["unlinked_png_count"], 0)
        self.assertEqual(linked[0]["frame_paths"]["SIDE"],
                         inv["manifests"][0]["frame_paths"]["SIDE"])

    def test_no_manifest_does_not_invent_events_from_minute(self):
        inv = case()
        inv["manifests"] = []
        linked, coverage = verified_event_manifests(inv)
        self.assertEqual(linked, [])
        self.assertEqual(coverage["unlinked_png_count"], 3)

    def test_invalid_or_duplicated_event_id_excluded(self):
        inv = case(valid=False)
        self.assertEqual(verified_event_manifests(inv)[0], [])
        inv = case()
        inv["manifests"].append(dict(inv["manifests"][0]))
        linked, coverage = verified_event_manifests(inv)
        self.assertEqual(linked, [])
        self.assertEqual(coverage["unusable_manifests"]["DUPLICATE_EVENT_ID"], 2)

    def test_missing_or_mismatched_hash_and_light_rejected(self):
        for mutator in (
            lambda inv: inv["images"][0].update(file_sha256=""),
            lambda inv: inv["images"][0].update(expected_label="OK"),
            lambda inv: inv["images"][0].update(manifest_links=[]),
        ):
            inv = case()
            mutator(inv)
            linked, _ = verified_event_manifests(inv)
            self.assertEqual(linked, [])

    def test_rejects_invalid_inventory_schema(self):
        with self.assertRaises(ValueError):
            verified_event_manifests({"schema": "other", "images": []})


class OperationalReplayTests(unittest.TestCase):
    def _execute(self, inv, *, early, verdict="DEFEITO REAL", cancel=None):
        holder = []
        def factory(root):
            obj = FakeRuntime(root, early=early, verdict=verdict)
            holder.append(obj)
            return obj
        with tempfile.TemporaryDirectory() as temp:
            report = replay(
                Path(temp), inventory=inv, runtime_factory=factory,
                frame_loader=lambda *args: np.zeros((32, 32, 3), dtype=np.uint8),
                cancelled=cancel,
            )
            self.assertEqual(report["schema"], SCHEMA)
            self.assertFalse(report["is_end_to_end_production_certification"])
            self.assertFalse(report["is_unseen_generalization_test"])
            return report, holder

    def test_side_early_reuses_real_side_decision_and_skips_top_mid(self):
        report, runtime = self._execute(case("NG"), early=True)
        row = report["events"][0]
        self.assertEqual(row["route"], "SIDE_EARLY")
        self.assertEqual(row["analyzed_lights"], ["SIDE"])
        self.assertEqual(row["skipped_lights"], ["TOP", "MID"])
        self.assertEqual(row["predicted_label"], "NG")
        self.assertEqual(row["status"], "PASSOU")
        self.assertEqual(runtime[0].called, ["SIDE"])
        self.assertEqual(report["counts"]["ng_as_ok"], 0)

    def test_full_multilight_counts_one_event_and_preserves_ng(self):
        report, runtime = self._execute(case("NG"), early=False)
        self.assertEqual(report["coverage"]["archive_png_count"], 3)
        self.assertEqual(report["counts"]["replayed"], 1)
        self.assertEqual(report["events"][0]["route"], "MULTILIGHT")
        self.assertEqual(runtime[0].called, ["SIDE", "TOP", "MID"])
        self.assertEqual(report["counts"]["ng_as_ok"], 0)

    def test_false_ng_is_reported_without_masking_by_image_count(self):
        report, _ = self._execute(case("OK"), early=True)
        self.assertEqual(report["counts"]["ok_as_ng"], 1)
        self.assertEqual(report["counts"]["passed"], 0)
        self.assertEqual(report["counts"]["incorrect"], 1)

    def test_without_verified_events_no_runtime_is_constructed(self):
        inv = case("NG")
        inv["manifests"] = []
        report, runtime = self._execute(inv, early=True)
        self.assertEqual(report["counts"]["qualified_events"], 0)
        self.assertEqual(report["counts"]["passed"], 0)
        self.assertEqual(runtime, [])

    def test_cancel_does_not_count_missing_events_as_correct(self):
        report, runtime = self._execute(case(), early=True,
                                         cancel=lambda: True)
        self.assertTrue(report["canceled"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["counts"]["not_evaluated"], 1)
        self.assertEqual(report["counts"]["passed"], 0)

    def test_report_json_and_txt_are_outside_archive(self):
        report, _ = self._execute(case(), early=True)
        with tempfile.TemporaryDirectory() as tmp:
            jp, tp = save_report(report, Path(tmp))
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["counts"]["replayed"], 1)
            self.assertIn("NG classificados como OK", tp.read_text(encoding="utf-8"))

    def test_png_modified_since_inventory_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "public/ng_archive"
            archive.mkdir(parents=True)
            path = archive / "2026-10-22_0800_FALTANDO_SIDE.png"
            image = np.zeros((50, 80, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(_verified_frame(
                root, path.relative_to(root).as_posix(), expected
            ).shape, image.shape)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "alterado"):
                _verified_frame(root, path.relative_to(root).as_posix(), expected)


class WiringTests(unittest.TestCase):
    def test_cli_and_ui_use_isolated_module_without_xp_commands(self):
        ui = Path("src/ui/training_validation_panel.py").read_text(encoding="utf-8")
        cli = Path(
            "src/services/startup_regression/operational_event_replay_cli.py"
        ).read_text(encoding="utf-8")
        runner = Path(
            "src/services/startup_regression/operational_event_replay.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Replay operacional por evento", ui)
        self.assertIn("operational_event_replay_cli", ui)
        self.assertIn("event_id", runner)
        self.assertIn("build_side_early_decision(", runner)
        self.assertIn("fuse_multilight(", runner)
        self.assertIn("apply_physical_missing_veto(", runner)
        self.assertIn("apply_knn_visual_corroboration(", runner)
        self.assertNotIn("send_xp_command", runner + cli)
        self.assertNotIn("save_label(", runner + cli)


if __name__ == "__main__":
    unittest.main()
