"""Estratégia C: deploy ao vivo só como revisão, não como emissor de XP."""
from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from src.core.knn_c_live import LiveKNNStrategyC, install_live_knn_strategy_c
from src.ui.production_confidence_gate import production_decision_policy


def sample(path, label, num, light="SIDE"):
    return {
        "path": f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/{path}",
        "status": "VALID_PNG",
        "expected_label": label,
        "lighting_mode": light,
        "category_hint": "INCORRETA_AOI",
        "pixel_sha256": f"pixel-sha-{num}",
        "file_sha256": f"file-sha-{num}",
    }


def pair(label):
    ref = np.full((64, 64, 3), 45, dtype=np.uint8)
    obs = ref.copy()
    cv2.rectangle(obs, (5, 6), (22, 28), (100, 100, 100), -1)
    if label == "NG":
        cv2.rectangle(obs, (32, 34), (52, 54), (210, 210, 210), -1)
    return ref, obs, {"category": "WRONG_AOI"}


def make_bank():
    rows = [
        sample("ok-1.png", "OK", 1),
        sample("ok-2.png", "OK", 2),
        sample("ok-3.png", "OK", 3),
        sample("ng-1.png", "NG", 4),
        sample("ng-2.png", "NG", 5),
        sample("ng-3.png", "NG", 6),
    ]
    bank = LiveKNNStrategyC(
        root=Path("/mock-root"),
        inventory={
            "schema": "visionx.archive_inventory.v1",
            "images": rows,
            "summary": {"png_count": len(rows)},
            "cross_label_conflicts": [],
        },
        extractor=pair,
        frame_loader=lambda root, row: row["expected_label"],
    )
    return bank


def outcome(verdict="FALHA FALSA", route="NEW_CNN"):
    return {
        "is_defect": verdict == "DEFEITO REAL",
        "verdict": verdict,
        "confidence": .99,
        "reason": "julgamento original",
        "detail": {
            "recognition_route": route,
            "decision_trace": {"fusion_rule": "original"},
        },
    }


def pipeline_factory(original_result):
    class Pipeline:
        def inspect(self, _ref, _obs, _regions, _info, _box, _epicenters):
            return original_result()

    return Pipeline


class StubStrategy:
    def __init__(self, status="SUGERE_NG"):
        self.value = status
        self.calls = 0

    def diagnose(self, *args):
        self.calls += 1
        return {
            "status": self.value, "reason": "A_E_B_CONCORDAM",
            "source": "A_E_B", "auto_decision_authorized": False,
        }


class KNNCLiveTests(unittest.TestCase):
    def test_archive_bank_ready_without_category_selection(self):
        bank = make_bank()
        status = bank.warmup()
        self.assertEqual(status["status"], "READY")
        ref, obs, _ = pair("NG")
        result = bank.diagnose(ref, obs, "SIDE")
        self.assertEqual(result["bank_status"], "READY")
        self.assertFalse(result["uses_aoi_category"])
        self.assertFalse(result["auto_decision_authorized"])
        self.assertIn(result["status"], ("SUGERE_OK", "SUGERE_NG", "REVISAO"))
        self.assertIsNone(result["xp_command"])
        if result["status"] in ("SUGERE_OK", "SUGERE_NG", "REVISAO") and (
            "vote_A" in result and "vote_B" in result
        ):
            for method in ("A", "B"):
                self.assertIn("reason_" + method, result)
                self.assertIn("weighted_ng_" + method, result)
                neighbors = result["neighbors_" + method]
                self.assertLessEqual(len(neighbors), 5)
                self.assertGreater(len(neighbors), 0)
                for entry in neighbors:
                    self.assertIn(entry["label"], ("OK", "NG"))
                    self.assertIn("similarity", entry)
                    self.assertIn("path", entry)
            self.assertFalse(result["auto_decision_authorized"])

    def test_unknown_light_has_no_reference_bank(self):
        bank = make_bank()
        bank.warmup()
        ref, obs, _ = pair("NG")
        status = bank.diagnose(ref, obs, "TOP")
        self.assertEqual(status["status"], "REVISAO")
        self.assertEqual(status["reason"], "SEM_MEMORIA_NESTA_ILUMINACAO")

    def test_conflicted_archive_does_not_load(self):
        bank = make_bank()
        bank._inventory["cross_label_conflicts"] = [{"paths": ["a", "b"]}]
        self.assertEqual(bank.warmup()["status"], "UNAVAILABLE")
        ref, obs, _ = pair("OK")
        self.assertEqual(bank.diagnose(ref, obs, "SIDE")["status"], "REVISAO")

    def call(self, verdict, route="NEW_CNN", proposed="SUGERE_NG",
             replay=False):
        klass = pipeline_factory(lambda: outcome(verdict, route))
        c = StubStrategy(proposed)
        install_live_knn_strategy_c(klass, strategy=c)
        ref, obs, _ = pair("OK")
        result = klass().inspect(
            ref, obs, [], {"lighting_mode": "SIDE",
                           "_replay_without_memory": replay}, {}, [],
        )
        return result, c.calls

    def test_exact_memory_never_interfered(self):
        for route in ("KNOWN_KNN", "MEMORY_CONFLICT"):
            with self.subTest(route=route):
                res, calls = self.call("FALHA FALSA", route)
                self.assertEqual(calls, 0)
                self.assertNotIn("knn_c", res["detail"])
                self.assertNotIn("production_review_required", res)

    def test_ng_vs_original_ok_vetoes_auto_ok(self):
        res, calls = self.call("FALHA FALSA")
        self.assertEqual(calls, 1)
        self.assertEqual(res["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertTrue(res["production_review_required"])
        self.assertTrue(res["detail"]["operator_review_required"])
        self.assertTrue(res["detail"]["decision_trace"]["operator_review_required"])
        self.assertTrue(res["detail"]["knn_c"]["conflicts_with_existing_verdict"])
        self.assertEqual(res["detail"]["knn_c"]["proposed_label"], "NG")
        self.assertEqual(res["confidence"], .5)
        policy = production_decision_policy(res)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])

    def test_ok_cannot_override_original_ng(self):
        res, calls = self.call("DEFEITO REAL", proposed="SUGERE_OK")
        self.assertEqual(calls, 1)
        self.assertEqual(res["verdict"], "DEFEITO REAL")
        self.assertTrue(res["is_defect"])
        self.assertTrue(res["production_review_required"])

    def test_agreement_requires_manual_confirmation_not_xp(self):
        res, _ = self.call("FALHA FALSA", proposed="SUGERE_OK")
        self.assertEqual(res["verdict"], "FALHA FALSA")
        self.assertTrue(res["production_review_required"])
        self.assertEqual(res["detail"]["knn_c"]["proposed_label"], "OK")
        policy = production_decision_policy(res)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])

    def test_review_does_not_force_a_verdict(self):
        res, _ = self.call("FALHA FALSA", proposed="REVISAO")
        self.assertEqual(res["verdict"], "FALHA FALSA")
        self.assertNotIn("production_review_required", res)
        self.assertEqual(res["detail"]["knn_c"]["status"], "REVISAO")

    def test_training_replay_bypasses_runtime_c(self):
        res, calls = self.call("FALHA FALSA", replay=True)
        self.assertEqual(calls, 0)
        self.assertNotIn("knn_c", res["detail"])


if __name__ == "__main__":
    unittest.main()
