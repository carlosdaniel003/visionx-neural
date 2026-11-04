"""V3 época 3: ponteiro reversível, inferência e veto NG em Produção."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3, MODEL_SCHEMA_V3
from src.core.neural.aoi_quality_cnn_v3_production_trial import (
    AOIQualityV3ProductionTrial, install_aoi_quality_v3_production_trial,
    POINTER_SCHEMA, SNAPSHOT_RELATIVE_PATH, SOURCE_MANIFEST_SHA256,
    INITIAL_V3_SHA256, TRIAL_POINTER, trial_pointer_payload,
    validate_trial_snapshot,
)
from src.core.neural.aoi_quality_cnn_v3_trial_consensus import (
    attach_v3_multilight_evidence, active_v3_veto,
)
from src.services.aoi_v3_epoch_diagnostics import EPOCH_SCHEMA
from src.scripts.activate_aoi_quality_v3_production_trial import (
    activate, deactivate, REPORT_RELATIVE_PATH,
)


class TrialFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / SNAPSHOT_RELATIVE_PATH
        self.snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(9)
        model = AOIQualityCNNV3().eval()
        self.weights = {
            "schema": EPOCH_SCHEMA, "architecture_schema": MODEL_SCHEMA_V3,
            "epoch": 3, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "previous_candidate_sha256": INITIAL_V3_SHA256,
            "image_size": 96, "focus_fraction": .70,
            "lights": ("SIDE", "TOP", "MID"),
            "diagnostics_only": True, "candidate_only": True,
            "production_approved": False, "automatic_xp_authorized": False,
            "test_evaluated_for_this_epoch": False,
            "selected_by_safe_gate": False,
            "validation": {"NG_false_OK": 3, "OK_false_NG": 0},
            "operator_training_metrics": {
                "NG_correct": 8, "OK_correct": 5, "correct": 13,
            },
            "model_weights_not_production_state_dict": model.state_dict(),
        }
        torch.save(self.weights, self.snapshot)
        self.digest = sha256(self.snapshot.read_bytes()).hexdigest()

    def report_fixture(self):
        history = [{"epoch": n, "validation": {},
                    "operator_training_replay": {"metrics": {}}}
                   for n in range(7)]
        history[0]["validation"] = {"NG_false_OK": 3, "OK_false_NG": 0}
        history[0]["operator_training_replay"]["metrics"] = {"correct": 0}
        history[3].update({
            "safe_gate_passed": True,
            "validation": {"NG_false_OK": 3, "OK_false_NG": 0},
            "operator_training_replay": {"metrics": {
                "NG_correct": 8, "OK_correct": 5, "correct": 13,
            }},
        })
        snapshots = [
            {"epoch": n, "path": SNAPSHOT_RELATIVE_PATH if n == 3
             else f"epoch_{n:02d}.pt",
             "sha256": self.digest if n == 3 else "a"*64,
             "production_approved": False,
             "automatic_xp_authorized": False}
            for n in range(1, 7)
        ]
        report = {
            "schema": "visionx.aoi_quality_cnn_v3_training_report.v1",
            "candidate_only": True,
            "production_approved": False,
            "automatic_xp_authorized": False,
            "training": {"selected_epoch_using_validation_only": 0},
            "operator_image_review_verified": 14,
            "aoi_category_used_for_training_selection_evaluation": False,
            "provenance": {"manifest_sha256": SOURCE_MANIFEST_SHA256},
            "initial_v3_candidate": {"candidate_sha256": INITIAL_V3_SHA256},
            "epoch_history": history,
            "epoch_diagnostic_checkpoints": snapshots,
        }
        path = self.root / REPORT_RELATIVE_PATH
        path.write_text(json.dumps(report), encoding="utf-8")
        return path


class SnapshotAndPointerTests(TrialFixture):
    def test_pinned_snapshot_is_only_loaded_after_full_sha_contract(self):
        with patch(
            "src.core.neural.aoi_quality_cnn_v3_production_trial.SNAPSHOT_SHA256",
            self.digest,
        ):
            raw, path = validate_trial_snapshot(self.root)
            self.assertEqual(path, self.snapshot)
            self.assertEqual(raw["operator_training_metrics"]["correct"], 13)
            self.snapshot.write_bytes(self.snapshot.read_bytes()+b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA"):
                validate_trial_snapshot(self.root)

    def test_cli_enables_and_removes_only_exact_trial_pointer(self):
        self.report_fixture()
        with patch(
            "src.core.neural.aoi_quality_cnn_v3_production_trial.SNAPSHOT_SHA256",
            self.digest,
        ), patch(
            "src.scripts.activate_aoi_quality_v3_production_trial.SNAPSHOT_SHA256",
            self.digest,
        ):
            self.assertEqual(activate(self.root), trial_pointer_payload())
            self.assertEqual(activate(self.root), trial_pointer_payload())
            self.assertTrue((self.root/TRIAL_POINTER).is_file())
            self.assertTrue(deactivate(self.root))
            self.assertFalse(deactivate(self.root))
            self.assertFalse((self.root/TRIAL_POINTER).exists())

    def test_snapshot_metadata_tamper_denied_even_with_recomputed_hash(self):
        self.weights["operator_training_metrics"]["correct"] = 14
        torch.save(self.weights, self.snapshot)
        digest = sha256(self.snapshot.read_bytes()).hexdigest()
        with patch(
            "src.core.neural.aoi_quality_cnn_v3_production_trial.SNAPSHOT_SHA256",
            digest,
        ):
            with self.assertRaisesRegex(ValueError, "evidência"):
                validate_trial_snapshot(self.root)

    def test_live_inference_is_off_without_pointer_and_reversible(self):
        engine = AOIQualityV3ProductionTrial(self.root)
        image = np.full((80, 120, 3), 127, dtype=np.uint8)
        self.assertIsNone(engine.predict(image, image, "SIDE"))
        (self.root/TRIAL_POINTER).parent.mkdir(parents=True, exist_ok=True)
        with patch(
            "src.core.neural.aoi_quality_cnn_v3_production_trial.SNAPSHOT_SHA256",
            self.digest,
        ):
            (self.root/TRIAL_POINTER).write_text(
                json.dumps(trial_pointer_payload()), encoding="utf-8"
            )
            outcome = engine.predict(image, image, "SIDE")
            self.assertEqual(outcome["schema"], POINTER_SCHEMA)
            self.assertIn(outcome["predicted"], {"OK", "NG"})
            self.assertFalse(outcome["automatic_ok_from_v3_authorized"])
        (self.root/TRIAL_POINTER).unlink()
        self.assertIsNone(engine.predict(image, image, "SIDE"))
        self.assertIsNone(engine._model)

    def test_corrupt_pointer_is_not_silently_downgraded(self):
        pointer = self.root/TRIAL_POINTER
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"enabled":true}', encoding="utf-8")
        engine = AOIQualityV3ProductionTrial(self.root)
        with self.assertRaisesRegex(ValueError, "divergente"):
            engine.enabled()


def _vote(light, score):
    return {
        "schema": POINTER_SCHEMA, "status": "READY",
        "epoch": 3, "light": light,
        "checkpoint_sha256": (
            "fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce"
        ),
        "ng_score_uncalibrated": score,
        "predicted": "NG" if score >= .5 else "OK",
        "ng_veto": score >= .5,
        "automatic_ok_from_v3_authorized": False,
    }


class FusionTests(unittest.TestCase):
    def test_three_lights_ng_veto_survives_non_dominant_mode(self):
        source = {light: {
            "detail": {"aoi_v3_trial": _vote(light, score)}
        } for light, score in [
            ("SIDE", .2), ("TOP", .99), ("MID", .3),
        ]}
        fused = attach_v3_multilight_evidence(
            {"verdict": "FALHA FALSA", "detail": {}}, source,
        )
        self.assertEqual(active_v3_veto(fused), ("READY", True))
        self.assertEqual(fused["verdict"], "FALHA FALSA")
        # Política de produção decide; fusão NÃO transmita 0 nem 1.
        self.assertFalse(fused["detail"]["aoi_v3_multilight"][
            "automatic_ok_from_v3_authorized"])

    def test_missing_or_corrupted_one_light_blocks_automatic_release(self):
        source = {light: {"detail": {"aoi_v3_trial": _vote(light, .1)}}
                  for light in ("SIDE", "TOP")}
        source["MID"] = {"detail": {}}
        fused = attach_v3_multilight_evidence({"detail": {}}, source)
        self.assertEqual(active_v3_veto(fused), ("ERROR", False))
        source["MID"]["detail"]["aoi_v3_trial"] = _vote("MID", float("nan"))
        fused = attach_v3_multilight_evidence({"detail": {}}, source)
        self.assertEqual(active_v3_veto(fused), ("ERROR", False))

    def test_one_light_veto_and_unchanged_path_when_disabled(self):
        analysis = {"lighting_mode": "SIDE",
                    "detail": {"aoi_v3_trial": _vote("SIDE", .98)}}
        self.assertEqual(active_v3_veto(analysis), ("READY", True))
        self.assertEqual(active_v3_veto({"detail": {}}), ("ABSENT", False))
        self.assertEqual(
            attach_v3_multilight_evidence({"detail": {}}, {
                light: {"detail": {}} for light in ("SIDE","TOP","MID")
            }), {"detail": {}}
        )


class RoutingTests(unittest.TestCase):
    def test_new_event_without_aoi_category_reaches_v3_and_known_preserved(self):
        seen = []
        class Dummy:
            def inspect(self, full_gab, full_test, raw_anomalies,
                        aoi_info, global_box_info, aoi_epicenters):
                return {
                    "verdict": "FALHA FALSA", "lighting_mode": "SIDE",
                    "detail": {"recognition_route": self.route},
                }
        class Predictor:
            def predict(self, a, b, light):
                seen.append(light)
                return _vote(light, .99)
        install_aoi_quality_v3_production_trial(Dummy, predictor=Predictor())
        dummy = Dummy()
        dummy.route = "NEW_EXPERTS"
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = dummy.inspect(image, image, [], {"lighting_mode":"SIDE",
                                    "category":""}, None, None)
        self.assertEqual(seen, ["SIDE"])
        self.assertEqual(result["detail"]["aoi_v3_trial"]["predicted"], "NG")
        self.assertEqual(result["verdict"], "FALHA FALSA")
        dummy.route = "KNOWN_KNN"
        known = dummy.inspect(image, image, [], {}, None, None)
        self.assertNotIn("aoi_v3_trial", known["detail"])
        self.assertEqual(seen, ["SIDE"])


if __name__ == "__main__":
    unittest.main()
