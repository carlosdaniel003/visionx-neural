"""Promotion V3 epoch 5: SHA, local opt-in, all light modes and rollback.

Uses an explicit synthetic fixture; real epoch 5 weights are local to the
operator's machine, never included in the repository or CI.
"""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3, MODEL_SCHEMA_V3
import src.core.neural.aoi_v3_epoch5_contract as epoch5
import src.core.neural.aoi_quality_cnn_v3_production_trial as runtime
import src.core.neural.aoi_quality_cnn_v3_trial_consensus as consensus
from src.core.neural.aoi_quality_cnn_v3_primary import AOIQualityV3Primary
from src.services.aoi_v3_epoch_diagnostics import EPOCH_SCHEMA
from src.scripts.activate_aoi_quality_v3_epoch5 import activate, deactivate


class Epoch5ActivationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / epoch5.SNAPSHOT_PATH
        self.snapshot.parent.mkdir(parents=True)
        torch.manual_seed(105)
        checkpoint = {
            "schema": EPOCH_SCHEMA,
            "architecture_schema": MODEL_SCHEMA_V3,
            "epoch": 5,
            "source_manifest_sha256": epoch5.MANIFEST_SHA256,
            "previous_candidate_sha256": epoch5.PREVIOUS_EPOCH3_SHA256,
            "image_size": 96,
            "focus_fraction": .70,
            "lights": ("SIDE", "TOP", "MID"),
            "diagnostics_only": True,
            "candidate_only": True,
            "production_approved": False,
            "automatic_xp_authorized": False,
            "selected_by_safe_gate": False,
            "test_evaluated_for_this_epoch": False,
            "validation": {"NG_false_OK": 0, "OK_false_NG": 2},
            "operator_training_metrics": {
                "correct": 14, "NG_correct": 9, "OK_correct": 5,
            },
            "model_weights_not_production_state_dict": AOIQualityCNNV3().state_dict(),
        }
        torch.save(checkpoint, self.snapshot)
        self.snapshot_sha = sha256(self.snapshot.read_bytes()).hexdigest()
        self.report_file = self.root / epoch5.REPORT_PATH
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        history = [{"epoch": i} for i in range(7)]
        history[5].update({
            "safe_gate_passed": False, "selected": False,
            "validation": {"NG_false_OK": 0, "OK_false_NG": 2},
            "operator_training_replay": {"metrics": {
                "correct": 14, "NG_correct": 9, "OK_correct": 5,
            }},
            "operator_ng_escape_training_replay": {
                "NG_total": 24, "NG_correct": 24,
            },
        })
        snapshots = [{"epoch": i, "path": "not_selected",
                      "sha256": "a"*64} for i in range(1, 7)]
        snapshots[4] = {
            "epoch": 5, "path": epoch5.SNAPSHOT_PATH.as_posix(),
            "sha256": self.snapshot_sha,
            "safe_gate_selected": False,
            "test_evaluated_for_this_epoch": False,
            "production_approved": False,
            "automatic_xp_authorized": False,
        }
        self.report = {
            "schema": "visionx.aoi_quality_cnn_v3_training_report.v1",
            "candidate_only": True,
            "production_approved": False,
            "automatic_xp_authorized": False,
            "operator_ng_escape_review_verified": 25,
            "training": {
                "selected_epoch_using_validation_only": 0,
                "v3_initialized_from_verified_epoch3": True,
            },
            "provenance": {
                "manifest_sha256": epoch5.MANIFEST_SHA256,
                "v3_ng_escape_sources_in_dataset": 25,
                "v3_ng_escape_unique_pairs_in_dataset": 24,
            },
            "initial_v3_candidate": {
                "checkpoint_sha256": epoch5.PREVIOUS_EPOCH3_SHA256,
            },
            "epoch_history": history,
            "epoch_diagnostic_checkpoints": snapshots,
        }
        self._write_report()

    def _write_report(self):
        self.report_file.write_text(json.dumps(self.report, indent=2)+"\n", encoding="utf-8")
        self.report_sha = sha256(self.report_file.read_bytes()).hexdigest()

    def _patch_hashes(self):
        stack = ExitStack()
        stack.enter_context(patch.object(epoch5, "SNAPSHOT_SHA256", self.snapshot_sha))
        stack.enter_context(patch.object(epoch5, "REPORT_SHA256", self.report_sha))
        stack.enter_context(patch.object(runtime, "EPOCH5_SNAPSHOT_SHA256", self.snapshot_sha))
        stack.enter_context(patch.object(consensus, "EPOCH5_SHA256", self.snapshot_sha))
        return stack

    def test_epoch5_runs_all_lights_as_same_primary_and_is_reversible(self):
        with self._patch_hashes():
            self.assertFalse(epoch5.epoch5_pointer_enabled(self.root))
            payload = activate(self.root)
            self.assertEqual(payload["epoch"], 5)
            self.assertFalse(payload["automatic_ok_from_v3_authorized"])
            self.assertEqual(payload["checkpoint_sha256"], self.snapshot_sha)
            self.assertEqual(activate(self.root), payload)
            engine = runtime.AOIQualityV3ProductionTrial(self.root)
            primary = AOIQualityV3Primary(engine=engine)
            self.assertEqual(primary.warmup()["state"], "ready")
            sample = np.full((80, 120, 3), 100, dtype=np.uint8)
            votes = {}
            for light in ("SIDE", "TOP", "MID"):
                prediction = engine.predict(sample, sample, light)
                self.assertEqual(prediction["schema"], epoch5.EPOCH5_VOTE_SCHEMA)
                self.assertEqual(prediction["epoch"], 5)
                self.assertEqual(prediction["checkpoint_sha256"], self.snapshot_sha)
                self.assertFalse(prediction["automatic_ok_from_v3_authorized"])
                self.assertTrue(consensus._valid_vote(prediction, light))
                result = primary.inspect(sample, sample, light)
                self.assertTrue(result["detail"]["cnn_v3_primary"])
                self.assertEqual(result["detail"]["cnn_v3_checkpoint_epoch"], 5)
                self.assertIn("época 5", result["reason"])
                self.assertNotIn("cnn_v2_active", result["detail"])
                votes[light] = {"detail": {"aoi_v3_trial": prediction}}
            combined = consensus.attach_v3_multilight_evidence(
                {"detail": {}}, votes,
            )
            self.assertEqual(combined["detail"]["aoi_v3_multilight"]["epoch"], 5)
            self.assertEqual(consensus.active_v3_veto(combined)[0], "READY")
            self.assertTrue(deactivate(self.root))
            self.assertFalse(deactivate(self.root))
            self.assertFalse(engine.enabled())
            self.assertIsNone(engine.predict(sample, sample, "SIDE"))

    def test_corrupt_local_snapshot_prevents_activation(self):
        with self._patch_hashes():
            self.snapshot.write_bytes(self.snapshot.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                activate(self.root)
            self.assertFalse((self.root / epoch5.EPOCH5_POINTER).exists())

    def test_unselected_epoch_is_never_pretended_to_have_passed_gate(self):
        self.report["epoch_history"][5]["safe_gate_passed"] = True
        self._write_report()
        with self._patch_hashes():
            with self.assertRaisesRegex(ValueError, "Métricas"):
                activate(self.root)
        self.assertFalse((self.root / epoch5.EPOCH5_POINTER).exists())

    def test_invalid_epoch5_pointer_never_falls_back_to_epoch3(self):
        with self._patch_hashes():
            p = self.root / epoch5.EPOCH5_POINTER
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"epoch":5}', encoding="utf-8")
            # Even if the old epoch 3 pointer exists, integrity failure on
            # the epoch5 pointer must stop all inference.
            old = self.root / runtime.TRIAL_POINTER
            old.write_text(json.dumps(runtime.trial_pointer_payload()), encoding="utf-8")
            engine = runtime.AOIQualityV3ProductionTrial(self.root)
            with self.assertRaisesRegex(ValueError, "diverge"):
                engine.enabled()
            with self.assertRaisesRegex(ValueError, "diverge"):
                activate(self.root)

    def test_mixing_epoch3_and_epoch5_votes_is_rejected(self):
        with self._patch_hashes():
            sample = {
                "schema": epoch5.EPOCH5_VOTE_SCHEMA,
                "status": "READY", "epoch": 5,
                "checkpoint_sha256": self.snapshot_sha,
                "ng_score_uncalibrated": .95,
                "predicted": "NG", "ng_veto": True,
                "automatic_ok_from_v3_authorized": False,
            }
            source = {light: {"detail": {"aoi_v3_trial": dict(sample, light=light)}}
                      for light in ("SIDE", "TOP", "MID")}
            source["MID"]["detail"]["aoi_v3_trial"].update({
                "schema": runtime.POINTER_SCHEMA,
                "epoch": 3,
                "checkpoint_sha256": runtime.SNAPSHOT_SHA256,
            })
            mixed = consensus.attach_v3_multilight_evidence({"detail": {}}, source)
            self.assertEqual(consensus.active_v3_veto(mixed), ("ERROR", False))
            self.assertEqual(
                mixed["detail"]["aoi_v3_multilight"]["reason"], "MIXED_V3_EPOCHS"
            )

    def test_epoch5_primary_ok_is_not_an_automatic_xp_release(self):
        from src.ui.production_confidence_gate import production_decision_policy

        class FakeEpoch5:
            def predict(self, reference, test, light):
                return {
                    "schema": epoch5.EPOCH5_VOTE_SCHEMA,
                    "status": "READY", "epoch": 5, "light": light,
                    "checkpoint_sha256": epoch5.SNAPSHOT_SHA256,
                    "ng_score_uncalibrated": .01,
                    "predicted": "OK", "ng_veto": False,
                    "automatic_ok_from_v3_authorized": False,
                }

        image = np.full((80, 120, 3), 100, dtype=np.uint8)
        analysis = AOIQualityV3Primary(engine=FakeEpoch5()).inspect(
            image, image, "SIDE",
        )
        analysis["multilight_final"] = True
        analysis["eligible_for_final_decision"] = True
        analysis["detail"]["multilight_final"] = True
        analysis["detail"]["eligible_for_final_decision"] = True
        analysis["detail"]["production_continuous_auto"] = True
        analysis["detail"]["production_binary_auto"] = True
        analysis["detail"]["recognition_route"] = "NEW_CNN"
        policy = production_decision_policy(analysis)
        self.assertFalse(policy["auto_allowed"])
        self.assertNotEqual(policy["proposed_decision"], "OK")
        self.assertEqual(
            policy["reason"],
            "cnn_v3_primary_ok_requires_independent_confirmation",
        )

    def test_malformed_votes_fail_closed(self):
        with self._patch_hashes():
            for bad in ([], {}, ["bad"], float("nan")):
                vote = {
                    "schema": epoch5.EPOCH5_VOTE_SCHEMA,
                    "status": "READY", "epoch": bad,
                    "checkpoint_sha256": self.snapshot_sha,
                    "light": "SIDE", "ng_score_uncalibrated": .8,
                    "predicted": "NG", "ng_veto": True,
                    "automatic_ok_from_v3_authorized": False,
                }
                self.assertFalse(consensus._valid_vote(vote, "SIDE"))


if __name__ == "__main__":
    unittest.main()
