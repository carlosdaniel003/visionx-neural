import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.core.full_frame_memory import (
    attach_full_frame_signature,
    build_full_frame_signature,
)
from src.core.strict_category_memory import install_strict_category_memory
from src.services.anomaly_learning import _decision_task
from src.services.dataset_manager import DatasetManager
from src.services.decision_persistence import DecisionPersistenceQueue


class MultiLightLearningSnapshotTests(unittest.TestCase):
    class Panel:
        pass

    def test_decision_task_carries_three_complete_lighting_samples(self):
        panel = self.Panel()
        panel.current_ng = np.full((20, 20, 3), 10, dtype=np.uint8)
        panel.current_sample = np.zeros((20, 20, 3), dtype=np.uint8)
        panel.current_aoi_info = {
            "board": "BOARD-1",
            "parts": "U2~5",
            "category": "FALTANDO",
            "value": "0 <= 54.872 <= 10 FALTANDO",
            "lighting_mode": "SIDE",
        }
        panel.current_analysis = {
            "is_defect": True,
            "verdict": "DEFEITO REAL",
            "detail": {"fusion_rule": "multilight_strong_single"},
        }
        panel.adhesive_multilight_primary_event_id = "evt-learning-001"
        panel.adhesive_multilight_learning_samples = {}
        for index, mode in enumerate(("SIDE", "TOP", "MID"), start=1):
            panel.adhesive_multilight_learning_samples[mode] = {
                "lighting_mode": mode,
                "sample_image": np.full(
                    (30, 40, 3),
                    index,
                    dtype=np.uint8,
                ),
                "test_image": np.full(
                    (30, 40, 3),
                    index * 10,
                    dtype=np.uint8,
                ),
                "source_frame": np.full(
                    (60, 80, 3),
                    index * 20,
                    dtype=np.uint8,
                ),
                "analysis": {
                    "is_defect": mode == "TOP",
                    "verdict": (
                        "DEFEITO REAL"
                        if mode == "TOP"
                        else "FALHA FALSA"
                    ),
                    "detail": {},
                },
            }

        task = _decision_task(panel, "NG", "button", "NG")

        self.assertEqual(task["event_id"], "evt-learning-001")
        self.assertEqual(
            [item["lighting_mode"] for item in task["multilight_samples"]],
            ["SIDE", "TOP", "MID"],
        )
        self.assertEqual(
            task["aoi_info"]["parts"],
            "U2~5",
        )
        self.assertTrue(
            all(
                item["test_image"].shape == (30, 40, 3)
                for item in task["multilight_samples"]
            )
        )


class MultiLightPersistenceQueueTests(unittest.TestCase):
    class DatasetStub:
        calls = []

        @staticmethod
        def save_sample(**kwargs):
            MultiLightPersistenceQueueTests.DatasetStub.calls.append(kwargs)
            return f"memory-{kwargs.get('lighting_mode', '')}.json"

    class OrchestratorStub:
        def __init__(self):
            self.reloads = 0

        def reload_memory(self):
            self.reloads += 1

    def test_one_human_judgement_persists_side_top_mid_and_reloads_once(self):
        self.DatasetStub.calls = []
        orchestrator = self.OrchestratorStub()
        queue = DecisionPersistenceQueue(
            orchestrator,
            dataset_manager=self.DatasetStub,
        )
        task = {
            "ng_image": np.zeros((10, 10, 3), dtype=np.uint8),
            "sample_image": np.zeros((10, 10, 3), dtype=np.uint8),
            "label": "NG",
            "aoi_info": {
                "board": "B1",
                "parts": "R3~5",
                "category": "FALTANDO",
                "value": "10 <= 2 <= 80 FALTANDO",
            },
            "analysis": {
                "is_defect": True,
                "verdict": "DEFEITO REAL",
                "detail": {},
            },
            "save_images": False,
            "source": "button",
            "ai_decision": "NG",
            "event_id": "evt-3lights",
            "multilight_samples": [],
        }
        for index, mode in enumerate(("SIDE", "TOP", "MID"), start=1):
            task["multilight_samples"].append(
                {
                    "lighting_mode": mode,
                    "sample_image": np.full(
                        (20, 20, 3),
                        index,
                        dtype=np.uint8,
                    ),
                    "test_image": np.full(
                        (20, 20, 3),
                        index + 10,
                        dtype=np.uint8,
                    ),
                    "source_frame": np.full(
                        (30, 30, 3),
                        index + 20,
                        dtype=np.uint8,
                    ),
                    "analysis": {
                        "is_defect": mode == "TOP",
                        "verdict": (
                            "DEFEITO REAL"
                            if mode == "TOP"
                            else "FALHA FALSA"
                        ),
                        "detail": {},
                    },
                }
            )

        queue.submit(task)
        queue.wait_until_idle()

        self.assertEqual(len(self.DatasetStub.calls), 3)
        self.assertEqual(
            [call["lighting_mode"] for call in self.DatasetStub.calls],
            ["SIDE", "TOP", "MID"],
        )
        self.assertTrue(
            all(call["save_images"] for call in self.DatasetStub.calls)
        )
        self.assertTrue(
            all(
                call["event_id"] == "evt-3lights"
                for call in self.DatasetStub.calls
            )
        )
        self.assertTrue(
            all(call["label"] == "NG" for call in self.DatasetStub.calls)
        )
        self.assertTrue(
            all(
                call["aoi_info"]["board"] == "B1"
                and call["aoi_info"]["parts"] == "R3~5"
                and call["aoi_info"]["category"] == "FALTANDO"
                and call["aoi_info"]["value"] == "10 <= 2 <= 80 FALTANDO"
                for call in self.DatasetStub.calls
            )
        )
        self.assertEqual(orchestrator.reloads, 1)


class FullFrameMemoryTests(unittest.TestCase):
    def test_full_frame_preserves_defect_outside_small_focus(self):
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        test = reference.copy()
        # Defeito distante de um hipotético epicentro no canto superior esquerdo.
        test[70:115, 100:155] = 255

        local = build_anomaly_signature(
            reference,
            test,
            {},
            {"category": "FALTANDO"},
            (0, 0, 20, 20),
        )
        enriched = attach_full_frame_signature(local, reference, test)
        full = enriched.get("full_frame_signature", {})

        self.assertTrue(full)
        self.assertEqual(full.get("memory_role"), "full_frame")
        self.assertGreater(
            float((full.get("summary") or {}).get("map_peak", 0.0)),
            0.0,
        )
        self.assertIn("full_frame", enriched.get("memory_scales", []))


class DatasetDedupTests(unittest.TestCase):
    def setUp(self):
        DatasetManager._folder_fingerprint_cache.clear()

    def test_legacy_side_duplicate_is_not_saved_again_but_new_top_is(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            anomaly_dir = root / "anomaly"
            reference = np.zeros((64, 64, 3), dtype=np.uint8)
            side = reference.copy()
            side[10:20, 10:20] = 120
            top = reference.copy()
            top[30:45, 30:45] = 220
            info = {
                "board": "B1",
                "parts": "U1",
                "category": "FALTANDO",
                "value": "FALTANDO",
            }

            with (
                patch.object(settings, "NORMAL_DIR", normal_dir),
                patch.object(settings, "ANOMALY_DIR", anomaly_dir),
            ):
                first = DatasetManager.save_sample(
                    ng_image=side,
                    label="OK",
                    sample_image=reference,
                    aoi_info=info,
                    analysis={"detail": {}},
                    save_images=True,
                )
                duplicate = DatasetManager.save_sample(
                    ng_image=side.copy(),
                    label="OK",
                    sample_image=reference,
                    aoi_info={**info, "lighting_mode": "SIDE"},
                    analysis={"detail": {}},
                    save_images=True,
                    lighting_mode="SIDE",
                    event_id="evt-repeat",
                )
                top_path = DatasetManager.save_sample(
                    ng_image=top,
                    label="OK",
                    sample_image=reference,
                    aoi_info={**info, "lighting_mode": "TOP"},
                    analysis={"detail": {}},
                    save_images=True,
                    lighting_mode="TOP",
                    event_id="evt-repeat",
                )

            self.assertEqual(first, duplicate)
            self.assertNotEqual(first, top_path)
            category_dir = normal_dir / "FALTANDO"
            self.assertEqual(len(list(category_dir.glob("*.json"))), 2)
            self.assertEqual(len(list(category_dir.glob("*_test.png"))), 2)

            with open(first, "r", encoding="utf-8") as file:
                upgraded = json.load(file)
            self.assertEqual(
                upgraded["aoi_info"]["lighting_mode"],
                "SIDE",
            )
            self.assertGreaterEqual(
                upgraded["deduplication"]["duplicate_observations"],
                1,
            )

    def test_same_pixels_in_different_lighting_keep_distinct_memories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            normal_dir = root / "normal"
            anomaly_dir = root / "anomaly"
            reference = np.zeros((48, 48, 3), dtype=np.uint8)
            image = reference.copy()
            image[12:30, 12:30] = 180
            info = {
                "board": "B2",
                "parts": "C7",
                "category": "DESLOCADO",
                "value": "DESLOCADO",
            }

            with (
                patch.object(settings, "NORMAL_DIR", normal_dir),
                patch.object(settings, "ANOMALY_DIR", anomaly_dir),
            ):
                side_path = DatasetManager.save_sample(
                    ng_image=image,
                    label="OK",
                    sample_image=reference,
                    aoi_info={**info, "lighting_mode": "SIDE"},
                    analysis={"detail": {}},
                    save_images=True,
                    lighting_mode="SIDE",
                    event_id="evt-identical",
                )
                top_path = DatasetManager.save_sample(
                    ng_image=image.copy(),
                    label="OK",
                    sample_image=reference,
                    aoi_info={**info, "lighting_mode": "TOP"},
                    analysis={"detail": {}},
                    save_images=True,
                    lighting_mode="TOP",
                    event_id="evt-identical",
                )

            self.assertNotEqual(side_path, top_path)
            category_dir = normal_dir / "DESLOCADO"
            self.assertEqual(len(list(category_dir.glob("*.json"))), 2)


class StrictLightingMemoryTests(unittest.TestCase):
    class DummyKNN:
        def __init__(self):
            signature = build_anomaly_signature(
                np.zeros((20, 20, 3), dtype=np.uint8),
                np.ones((20, 20, 3), dtype=np.uint8),
                {},
                {"category": "FALTANDO"},
                None,
            )
            self.signatures_ok = [
                {
                    "mode": "anomaly",
                    "category": "FALTANDO",
                    "lighting_mode": "SIDE",
                    "anomaly_signature": signature,
                },
                {
                    "mode": "anomaly",
                    "category": "FALTANDO",
                    "lighting_mode": "TOP",
                    "anomaly_signature": signature,
                },
            ]
            self.signatures_ng = []

        @staticmethod
        def _empty_result(query_anomaly_signature=None):
            return {
                "has_memory": False,
                "query_anomaly_signature": query_anomaly_signature or {},
            }

        def _analyze_anomaly_memory(
            self,
            anomaly_signature,
            ok,
            ng,
            top_k,
            scope,
        ):
            return {
                "has_memory": True,
                "n_neighbors": len(ok) + len(ng),
            }

    def test_top_query_does_not_mix_side_memory(self):
        cls = self.DummyKNN
        install_strict_category_memory(cls)
        knn = cls()
        signature = knn.signatures_ok[0]["anomaly_signature"]

        result = knn.analyze(
            None,
            None,
            aoi_info={
                "category": "FALTANDO",
                "lighting_mode": "TOP",
            },
            anomaly_signature=signature,
        )

        self.assertEqual(result["memory_lighting"], "TOP")
        self.assertEqual(result["memory_candidate_count"], 1)
        self.assertEqual(result["n_neighbors"], 1)


if __name__ == "__main__":
    unittest.main()
