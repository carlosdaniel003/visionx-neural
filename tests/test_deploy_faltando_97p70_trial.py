"""Controlled active 97.70% CNN with reversible operator-only review."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.core.neural.faltando_checkpoint_contract import (
    POINTER_SCHEMA, TRIAL_97P70_CHECKPOINT, TRIAL_97P70_MODE,
)
from src.core.neural.faltando_live import FaltandoCNNLive
from src.scripts import deploy_faltando_97p70_trial as deployment
from src.ui.production_confidence_gate import production_decision_policy


class ControlledDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root/"reports/faltando_neural/archive_experiments/test"
        source.mkdir(parents=True)
        self.source = source/"faltando_v2_archive_experimental_candidate.pt"
        torch.manual_seed(42)
        model = FaltandoCNNV2()
        torch.save({
            "schema": MODEL_SCHEMA_V2,
            "state_dict": model.state_dict(),
            "lights": ("SIDE", "TOP", "MID"),
            "experimental": True,
            "production_approved": False,
            "image_size": 64,
            "focus_fraction": .7,
            "individual_pair_review_complete": False,
            "archive_labels_experimental": True,
            "source_champion_sha256": "a"*64,
            "provisional_archive_manifest_sha256": "b"*64,
        }, self.source)
        self.sha = sha256(self.source.read_bytes()).hexdigest()
        data = {
            "qualification_mode": "ARCHIVE_FOLDER_BULK_EXPERIMENT",
            "candidate_only": True,
            "promoted": False,
            "production_approval_eligible": False,
            "candidate_relative_path": self.source.relative_to(self.root).as_posix(),
            "candidate_sha256": self.sha,
            "base_checkpoint_sha256": "a"*64,
            "provenance": {"manifest_sha256": "b"*64},
            "new_holdout": {"images": 87, "correct_images": 85, "ng_false_ok": 0},
            "new_historical": {
                "images": 117, "correct_images": 98,
                "cases": (
                    [{"label": "OK", "predicted": "OK"} for _ in range(98)]
                    + [{"label": "OK", "predicted": "NG"} for _ in range(19)]
                ),
            },
        }
        report = self.root/deployment.ORIGINAL_REPORT
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(data), encoding="utf-8")

    def _install_patches(self):
        return (
            patch.object(deployment, "TRIAL_97P70_SHA256", self.sha),
            patch("src.core.neural.faltando_live.TRIAL_97P70_SHA256", self.sha),
            patch.object(deployment, "_validate_original_station"),
            patch.object(deployment, "_read_benchmark", return_value={}),
        )

    def _live_fixture(self):
        ptr = self.root/deployment.POINTER
        ptr.parent.mkdir(parents=True, exist_ok=True)
        target = self.root/TRIAL_97P70_CHECKPOINT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.source.read_bytes())
        pointer = {
            "schema": POINTER_SCHEMA, "category": "FALTANDO",
            "checkpoint_relative_path": TRIAL_97P70_CHECKPOINT,
            "checkpoint_sha256": self.sha,
            "deployment_mode": TRIAL_97P70_MODE,
            "manual_review_required": True,
            "automatic_ok_authorized": False,
            "historical_ok_regression_acknowledged": True,
            "experimental": True, "production_approved": False,
        }
        ptr.write_text(json.dumps(pointer), encoding="utf-8")
        return ptr

    def test_install_refuses_without_ack_and_never_changes_weights(self):
        before = sha256(self.source.read_bytes()).hexdigest()
        with patch.object(deployment, "_install", side_effect=AssertionError("called")):
            with self.assertRaisesRegex(ValueError, "confirmação explícita"):
                deployment.main(["install", "--root", str(self.root)])
        self.assertEqual(before, sha256(self.source.read_bytes()).hexdigest())
        self.assertFalse((self.root/deployment.POINTER).exists())

    def test_real_checkpoint_warmup_installs_and_rolls_back(self):
        pre, runtime_sha, baseline, bench = self._install_patches()
        with pre, runtime_sha, baseline, bench:
            answer = deployment._install(self.root)
            self.assertEqual(answer["status"], "trial_active")
            self.assertEqual(self.sha, sha256(
                (self.root/TRIAL_97P70_CHECKPOINT).read_bytes()
            ).hexdigest())
            self.assertEqual(deployment._read_trial_pointer(self.root)[
                "deployment_mode"], TRIAL_97P70_MODE)
            with self.assertRaisesRegex(ValueError, "já existe"):
                deployment._install(self.root)
            live = FaltandoCNNLive(online_root=self.root)
            reference = np.full((88, 99, 3), 100, dtype=np.uint8)
            target = reference.copy()
            target[20:65, 30:75] = 160
            result = live.inspect(reference, target, "SIDE")
            self.assertEqual(result["detail"]["cnn_v2_status"], "INFERENCE_OK")
            self.assertTrue(result["detail"]["cnn_v2_trial_manual_only"])
            self.assertTrue(result["production_review_required"])
            self.assertFalse(production_decision_policy(result)["auto_allowed"])
            rollback = deployment._rollback(self.root)
            self.assertEqual(rollback["status"], "baseline_restored_after_restart")
            self.assertFalse((self.root/deployment.POINTER).exists())
            self.assertEqual(self.sha, sha256(self.source.read_bytes()).hexdigest())

    def test_tampered_pointer_does_not_switch_or_rollback(self):
        ptr = self._live_fixture()
        data = json.loads(ptr.read_text(encoding="utf-8"))
        data["manual_review_required"] = False
        ptr.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(deployment, "TRIAL_97P70_SHA256", self.sha):
            with self.assertRaisesRegex(ValueError, "outra implantação"):
                deployment._rollback(self.root)
        self.assertTrue(ptr.exists())
        with patch("src.core.neural.faltando_live.TRIAL_97P70_SHA256", self.sha):
            live = FaltandoCNNLive(online_root=self.root)
            res = live.inspect(np.ones((65, 65, 3), np.uint8),
                               np.ones((65, 65, 3), np.uint8), "SIDE")
            self.assertEqual(res["verdict"], "REVISÃO OBRIGATÓRIA")
            self.assertTrue(res["production_review_required"])

    def test_production_policy_blocks_supervised_auto_even_fused(self):
        # The fusion can overwrite operator_review_required of a light,
        # but its SHA is preserved per illumination.
        a = {
            "verdict": "FALHA FALSA",
            "confidence": .999,
            "multilight_final": True,
            "eligible_for_final_decision": True,
            "detail": {
                "cnn_v2_active": True, "cnn_v2_experimental": True,
                "multilight_final": True,
                "eligible_for_final_decision": True,
                "multilight_category": "FALTANDO",
                "recognition_route": "NEW_CNN",
                "cnn_v2_supervised_auto_eligible": True,
                "cnn_v2_consensus": "OK",
                "cnn_v2_light_votes": {x: "OK" for x in ("SIDE", "TOP", "MID")},
                "cnn_v2_verified_lights": ["SIDE", "TOP", "MID"],
                "cnn_v2_light_diagnostics": {
                    x: {"checkpoint_sha256": "OTHER"} for x in ("SIDE", "TOP", "MID")
                },
                "physical_missing_veto": {"status": "NO_STRONG_EVIDENCE"},
            },
        }
        self.assertTrue(production_decision_policy(a)["auto_allowed"])
        from src.core.neural import faltando_checkpoint_contract as contract
        a["detail"]["cnn_v2_light_diagnostics"]["TOP"]["checkpoint_sha256"] = (
            contract.TRIAL_97P70_SHA256
        )
        verdict = production_decision_policy(a)
        self.assertFalse(verdict["auto_allowed"])
        self.assertTrue(verdict["operator_review_required"])
        self.assertEqual(
            verdict["reason"], "cnn_97p70_unqualified_operator_review_only"
        )
        a["detail"]["cnn_v2_light_diagnostics"]["TOP"]["checkpoint_sha256"] = "OTHER"
        a["detail"]["cnn_v2_checkpoint_sha256"] = contract.TRIAL_97P70_SHA256
        self.assertFalse(production_decision_policy(a)["auto_allowed"])
        a["detail"]["cnn_v2_checkpoint_sha256"] = "OTHER"
        a["detail"]["cnn_v2_trial_manual_only"] = True
        self.assertFalse(production_decision_policy(a)["auto_allowed"])

    def test_refuse_existing_unrelated_pointer_without_overwriting(self):
        ptr = self.root/deployment.POINTER
        ptr.parent.mkdir(parents=True)
        before = b'{"schema":"third_party_or_old"}'
        ptr.write_bytes(before)
        with patch.object(deployment, "TRIAL_97P70_SHA256", self.sha):
            with self.assertRaisesRegex(ValueError, "já existe"):
                deployment._install(self.root)
        self.assertEqual(ptr.read_bytes(), before)
