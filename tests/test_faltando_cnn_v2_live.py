"""Verificações do motor CNN FALTANDO v2 integrado ao ODIN normal.

Não envolve arquivos da fábrica. Testa rotas, checkpoint, multilight e
proibição de AUTO-OK na produção enquanto o modelo for experimental.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.core.neural.faltando_live import (
    FaltandoCNNLive, install_faltando_cnn_live,
    PINNED_CHECKPOINT_SHA256,
)
from src.core.multilight_fusion import fuse_multilight
from src.ui.production_confidence_gate import production_decision_policy


class FakeOrchestrator:
    def __init__(self):
        self.old_inspections = 0

    def inspect(self, *args):
        self.old_inspections += 1
        return {"old_engine": True, "verdict": "FALHA FALSA"}


class FakePredictor:
    def __init__(self):
        self.calls = []

    def inspect(self, reference, test, mode):
        self.calls.append(mode)
        return {"verdict": "DEFEITO REAL", "active_engines": ["faltando_cnn_v2.py"]}


def pair(v=30):
    ref = np.full((112, 139, 3), v, dtype=np.uint8)
    test = ref.copy()
    test[26:75, 40:90] = 250
    return ref, test


class RoutingTests(unittest.TestCase):
    def test_direct_faltando_route_bypasses_old_experts_for_all_lights(self):
        class Orchestrator(FakeOrchestrator):
            pass

        fake = FakePredictor()
        install_faltando_cnn_live(Orchestrator, predictor=fake)
        unit = Orchestrator()
        r, t = pair()
        for light in ("SIDE", "TOP", "MID"):
            out = unit.inspect(r, t, [], {"category": "FALTANDO", "lighting_mode": light}, {}, [])
            self.assertEqual(out["active_engines"], ["faltando_cnn_v2.py"])
        self.assertEqual(fake.calls, ["SIDE", "TOP", "MID"])
        self.assertEqual(unit.old_inspections, 0)

    def test_nonadhesive_categories_route_to_cnn_and_replay_stays_physical(self):
        class Orchestrator(FakeOrchestrator):
            pass

        fake = FakePredictor()
        install_faltando_cnn_live(Orchestrator, predictor=fake)
        unit = Orchestrator()
        r, t = pair()
        for cat in ("DESLOCADO", "EMBORCADO", "INVERTIDO"):
            self.assertEqual(unit.inspect(r, t, [], {"category": cat}, {}, [])["active_engines"], ["faltando_cnn_v2.py"])
        for cat in ("MUITO ADESIVO", "Much Adhesive", "UNKNOWN"):
            self.assertTrue(unit.inspect(r, t, [], {"category": cat}, {}, [])["old_engine"])
        self.assertTrue(unit.inspect(
            r, t, [], {"category": "FALTANDO", "_replay_without_memory": True}, {}, []
        )["old_engine"])
        self.assertEqual(unit.old_inspections, 4)
        self.assertEqual(fake.calls, ["SIDE", "SIDE", "SIDE"])

    def test_double_installation_does_not_double_predict(self):
        class Orchestrator(FakeOrchestrator):
            pass

        p = FakePredictor()
        install_faltando_cnn_live(Orchestrator, predictor=p)
        install_faltando_cnn_live(Orchestrator, predictor=p)
        r, t = pair()
        Orchestrator().inspect(r, t, [], {"category": "FALTANDO"}, {}, [])
        self.assertEqual(p.calls, ["SIDE"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name)/"checkpoint.pt"
        model = FaltandoCNNV2()
        torch.save({
            "schema": MODEL_SCHEMA_V2,
            "state_dict": model.state_dict(),
            "lights": ("SIDE", "TOP", "MID"),
            "experimental": True,
            "production_approved": False,
            "image_size": 64,
            "focus_fraction": .70,
            "best_epoch": 18,
        }, self.checkpoint)
        self.hash = sha256(self.checkpoint.read_bytes()).hexdigest()

    def test_valid_checkpoint_runs_real_inference_and_never_calls_knn(self):
        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        r, t = pair()
        out = clf.inspect(r, t, "SIDE")
        self.assertTrue(out["detail"]["cnn_v2_active"])
        self.assertTrue(out["detail"]["cnn_v2_experimental"])
        self.assertTrue(out["detail"]["cnn_v2_checkpoint_verified"])
        self.assertEqual(out["detail"]["dominant_engine"], "faltando_cnn_v2")
        self.assertEqual(out["active_engines"], ["faltando_cnn_v2.py"])
        self.assertEqual(out["detail"]["decision_trace"]["weights"]["knn"], 0.0)
        self.assertEqual(out["detail"]["cnn_v2_checkpoint_best_epoch"], 18)
        self.assertTrue(0 <= out["detail"]["cnn_v2_ng_score_uncalibrated"] <= 1)
        self.assertEqual(production_decision_policy(out)["auto_allowed"], False)

    def test_corrupted_checkpoint_or_missing_file_fails_to_review(self):
        self.checkpoint.write_bytes(self.checkpoint.read_bytes()+b"bad")
        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        r, t = pair()
        out = clf.inspect(r, t, "SIDE")
        self.assertEqual(out["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(out["is_defect"])
        self.assertTrue(out["production_review_required"])
        self.assertFalse(out["detail"]["cnn_v2_checkpoint_verified"])
        self.checkpoint.unlink()
        other = FaltandoCNNLive(self.checkpoint, self.hash)
        self.assertEqual(other.inspect(r, t, "SIDE")["verdict"], "REVISÃO OBRIGATÓRIA")

    def test_invalid_inputs_and_light_require_review_without_loading_model(self):
        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        r, t = pair()
        cases = [(r, t, "UNKNOWN"), (None, t, "SIDE"), (r, np.zeros((0,0,3)), "SIDE")]
        for a, b, light in cases:
            self.assertEqual(clf.inspect(a, b, light)["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertIsNone(clf._model)


    def test_identical_cnn_inference_is_cached_in_session_not_self_trained(self):
        from unittest.mock import patch

        class DeterministicModel:
            calls = 0
            def __call__(self, *args):
                self.calls += 1
                return torch.tensor([-11.0]), torch.tensor([[-11.0, -11.0, -11.0]])

        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        clf._metadata = {"image_size": 64, "focus_fraction": .70}
        dummy = DeterministicModel()
        r, t = pair()
        with patch.object(clf, "_load", return_value=dummy):
            first = clf.inspect(r, t, "SIDE")
            second = clf.inspect(r, t, "SIDE")
            other_light = clf.inspect(r, t, "TOP")
            changed = t.copy()
            changed[1, 1, 0] ^= 1
            changed_result = clf.inspect(r, changed, "SIDE")
        self.assertEqual(dummy.calls, 3)
        self.assertFalse(first["detail"]["cnn_v2_session_inference_cache"])
        self.assertTrue(second["detail"]["cnn_v2_session_inference_cache"])
        self.assertFalse(other_light["detail"]["cnn_v2_session_inference_cache"])
        self.assertFalse(changed_result["detail"]["cnn_v2_session_inference_cache"])
        self.assertEqual(first["verdict"], second["verdict"])
        self.assertFalse(second["detail"]["cnn_v2_session_cache_is_training"])

    def test_internal_profile_separates_checkpoint_preprocess_forward_and_cache(self):
        from unittest.mock import patch

        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        r, t = pair()
        first = clf.inspect(r, t, "SIDE")
        self.assertEqual(first["detail"]["cnn_v2_status"], "INFERENCE_OK")
        first_profile = first["detail"]["cnn_v2_performance"]
        self.assertFalse(first_profile["cache_hit"])
        self.assertTrue(first_profile["checkpoint_loaded"])
        self.assertIn("checkpoint_hash_ms", first_profile["checkpoint_load_ms"])
        self.assertIn("checkpoint_deserialize_ms", first_profile["checkpoint_load_ms"])
        self.assertIn("model_construct_ms", first_profile["checkpoint_load_ms"])
        self.assertIn("fingerprints_ms", first_profile["stages_ms"])
        self.assertIn("input_prepare_ms", first_profile["stages_ms"])
        self.assertIn("torch_forward_ms", first_profile["stages_ms"])
        self.assertGreaterEqual(first_profile["total_ms"], 0)
        self.assertGreaterEqual(first_profile["torch_num_threads"], 1)

        second = clf.inspect(r, t, "SIDE")
        second_profile = second["detail"]["cnn_v2_performance"]
        self.assertTrue(second_profile["cache_hit"])
        self.assertEqual(second_profile["checkpoint_load_ms"], {})
        self.assertNotIn("torch_forward_ms", second_profile["stages_ms"])
        self.assertFalse(second_profile["checkpoint_loaded"])
        self.assertEqual(
            first["detail"]["cnn_v2_ng_score_uncalibrated"],
            second["detail"]["cnn_v2_ng_score_uncalibrated"],
        )
        self.assertFalse(
            first["detail"]["cnn_v2_performance"]["cache_hit"]
        )
        other = clf.inspect(r, t, "TOP")
        other_profile = other["detail"]["cnn_v2_performance"]
        self.assertFalse(other_profile["cache_hit"])
        self.assertFalse(other_profile["checkpoint_loaded"])
        self.assertIn("torch_forward_ms", other_profile["stages_ms"])
        self.assertIn("model_acquire_ms", other_profile["stages_ms"])

    def test_profile_failure_preserves_required_review(self):
        clf = FaltandoCNNLive(self.checkpoint, "0" * 64)
        r, t = pair()
        result = clf.inspect(r, t, "SIDE")
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["detail"]["cnn_v2_checkpoint_verified"])
        perf = result["detail"]["cnn_v2_performance"]
        self.assertEqual(perf["status"], "ValueError")
        self.assertIn("checkpoint_hash_ms", perf["checkpoint_load_ms"])
        self.assertNotIn("torch_forward_ms", perf["stages_ms"])

    def test_wrapper_profiles_persistent_memory_lookup_before_cnn(self):
        class MemoryPredictor:
            def __init__(self):
                self.calls = 0

            def recognize_confirmed(self, *args):
                return None

            def inspect(self, reference, test, mode):
                self.calls += 1
                return {
                    "verdict": "FALHA FALSA",
                    "active_engines": ["faltando_cnn_v2.py"],
                    "detail": {},
                }

        class Orchestrator(FakeOrchestrator):
            pass

        fake = MemoryPredictor()
        install_faltando_cnn_live(Orchestrator, predictor=fake)
        r, t = pair()
        result = Orchestrator().inspect(
            r, t, [], {"category": "FALTANDO", "lighting_mode": "SIDE"}, {}, []
        )
        self.assertEqual(fake.calls, 1)
        self.assertIn("cnn_v2_memory_lookup_ms", result["detail"])
        self.assertIn("cnn_v2_route_total_ms", result["detail"])
        self.assertGreaterEqual(
            result["detail"]["cnn_v2_route_total_ms"],
            result["detail"]["cnn_v2_memory_lookup_ms"],
        )

    def test_unexpected_model_exception_forces_review(self):
        class BrokenModel:
            def __call__(self, *inputs):
                raise AssertionError("falha inesperada no motor neural")

        clf = FaltandoCNNLive(self.checkpoint, self.hash)
        clf._model = BrokenModel()
        clf._metadata = {"image_size": 64, "focus_fraction": .70}
        r, t = pair()
        result = clf.inspect(r, t, "SIDE")
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(result["detail"]["cnn_v2_status"], "AssertionError")
        self.assertTrue(result["production_review_required"])

    def test_pin_uses_approved_archive_digest(self):
        self.assertEqual(
            PINNED_CHECKPOINT_SHA256,
            "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599"
        )


class ProductionTests(unittest.TestCase):
    @staticmethod
    def analysis(label="OK", light="SIDE"):
        score = .00002 if label == "OK" else .99993
        defect = label == "NG"
        return {
            "is_defect": defect,
            "verdict": "DEFEITO REAL" if defect else "FALHA FALSA",
            "confidence": .9999,
            "detail": {
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_lighting_mode": light,
                "final_score": score,
                "physical_score": 0.0,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "dominant_engine": "faltando_cnn_v2",
            },
        }

    def test_experimental_cnn_ok_requires_operator_not_auto_zero(self):
        outcome = production_decision_policy(self.analysis("OK"))
        self.assertFalse(outcome["auto_allowed"])
        self.assertEqual(outcome["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(outcome["proposed_decision"], "")
        self.assertTrue(outcome["operator_review_required"])
        self.assertEqual(outcome["reason"], "faltando_cnn_v2_experimental_requires_operator")

    def test_other_categories_auto_ok_unchanged(self):
        analysis = {"is_defect": False, "verdict": "FALHA FALSA", "detail": {}}
        self.assertTrue(production_decision_policy(analysis)["auto_allowed"])

    def test_multilight_fusion_preserves_cnn_engine_and_safe_operator_gate(self):
        info = {mode: self.analysis("OK", mode) for mode in ("SIDE", "TOP", "MID")}
        result = fuse_multilight(info, "FALTANDO")
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertTrue(result["detail"]["cnn_v2_active"])
        self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_multilight_single_cnn_ng_is_not_overridden_by_legacy_missing_rules(self):
        info = {mode: self.analysis("OK", mode) for mode in ("SIDE", "TOP", "MID")}
        info["TOP"] = self.analysis("NG", "TOP")
        result = fuse_multilight(info, "FALTANDO")
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertFalse(production_decision_policy(result)["auto_allowed"])


if __name__ == "__main__":
    unittest.main()
