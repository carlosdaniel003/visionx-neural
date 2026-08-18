import json
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import (
    build_anomaly_signature,
    compare_anomaly_signatures as compare_local_signatures,
)
from src.core.dual_scale_memory import (
    CONTEXT_WEIGHT,
    EPICENTER_WEIGHT,
    attach_component_context,
    compare_component_context_signatures,
    valid_context_signature,
)
from src.core.prototype_memory import (
    OK_PROTOTYPE_MERGE_SIMILARITY,
    PROTOTYPE_SCHEMA,
    condense_ok_records,
    install_prototype_memory,
    protect_ng_records,
)
from src.services.dataset_manager import DatasetManager


ROOT = Path(__file__).resolve().parents[1]


def _scene(style="a", anomaly=True):
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (35, 35), (285, 205), (0, 255, 0), 4)
    if style == "a":
        image[55:185, 60:260] = (90, 90, 90)
        cv2.rectangle(image, (85, 75), (235, 165), (160, 160, 160), -1)
        cv2.line(image, (95, 120), (225, 120), (30, 30, 30), 6)
    elif style == "b":
        image[55:185, 60:260] = (40, 80, 170)
        for y in range(65, 180, 24):
            for x in range(70, 250, 30):
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + 18, y + 14),
                    (180, 45, 35) if (x + y) % 2 else (30, 180, 200),
                    -1,
                )
    else:
        image[55:185, 60:260] = (120, 75, 45)
        cv2.circle(image, (160, 120), 55, (210, 210, 210), -1)

    if anomaly:
        cv2.rectangle(image, (148, 106), (174, 136), (245, 245, 245), -1)
        cv2.line(image, (150, 108), (172, 134), (15, 15, 15), 3)
    return image


def _dual_signature(style="a", part="C120"):
    reference = _scene(style, anomaly=False)
    test = _scene(style, anomaly=True)
    local = build_anomaly_signature(
        reference,
        test,
        {},
        {"board": "BOARD-A", "parts": part, "category": "FALTANDO"},
        (140, 100, 45, 45),
    )
    return attach_component_context(local, reference, test), reference, test


def _dual_compare(first, second):
    local_similarity, local_breakdown = compare_local_signatures(first, second)
    first_context = first.get("context_signature", {})
    second_context = second.get("context_signature", {})
    if valid_context_signature(first_context) and valid_context_signature(second_context):
        context_similarity, context_breakdown = compare_component_context_signatures(
            first_context,
            second_context,
        )
        combined = (
            local_similarity * EPICENTER_WEIGHT
            + context_similarity * CONTEXT_WEIGHT
        )
        return float(combined), {
            "dual_scale": True,
            "epicenter_similarity": float(local_similarity),
            "context_similarity": float(context_similarity),
            "epicenter": local_breakdown,
            "component_context": context_breakdown,
        }
    return float(local_similarity), {
        "dual_scale": False,
        "epicenter_similarity": float(local_similarity),
        "context_similarity": None,
        "epicenter": local_breakdown,
    }


def _record(signature, index, *, label="OK", part="C120", board="BOARDA"):
    return {
        "category": "FALTANDO",
        "board": board,
        "part": part,
        "label": label,
        "path": f"{label}_{index}.json",
        "json_path": f"missing_{label}_{index}.json",
        "mode": "anomaly",
        "anomaly_signature": signature,
    }


class InMemoryPrototypeTests(unittest.TestCase):
    def test_fifty_equivalent_ok_records_become_one_prototype(self):
        signature, _, _ = _dual_signature("a")
        records = [_record(signature, index) for index in range(50)]

        prototypes = condense_ok_records(records, _dual_compare)

        self.assertEqual(len(prototypes), 1)
        self.assertEqual(prototypes[0]["prototype_occurrences"], 50)
        self.assertEqual(prototypes[0]["prototype_member_jsons"], 50)
        self.assertFalse(prototypes[0]["prototype_protected"])
        self.assertFalse(prototypes[0]["quantity_influence"])
        self.assertEqual(prototypes[0]["prototype_schema"], PROTOTYPE_SCHEMA)

    def test_identical_ng_records_are_never_condensed(self):
        signature, _, _ = _dual_signature("a")
        records = [
            _record(signature, index, label="NG")
            for index in range(3)
        ]

        protected = protect_ng_records(records)

        self.assertEqual(len(protected), 3)
        self.assertTrue(all(item["prototype_protected"] for item in protected))
        self.assertTrue(all(item["prototype_member_jsons"] == 1 for item in protected))
        self.assertTrue(all(item["quantity_influence"] is False for item in protected))

    def test_different_components_are_not_merged(self):
        signature, _, _ = _dual_signature("a")
        records = [
            _record(signature, 1, part="C120"),
            _record(signature, 2, part="C121"),
        ]

        prototypes = condense_ok_records(records, _dual_compare)

        self.assertEqual(len(prototypes), 2)

    def test_different_boards_are_not_merged(self):
        signature, _, _ = _dual_signature("a")
        records = [
            _record(signature, 1, board="BOARD-A"),
            _record(signature, 2, board="BOARD-B"),
        ]

        prototypes = condense_ok_records(records, _dual_compare)

        self.assertEqual(len(prototypes), 2)

    def test_visually_different_ok_patterns_remain_distinct(self):
        first, _, _ = _dual_signature("a")
        second, _, _ = _dual_signature("b")
        similarity, breakdown = _dual_compare(first, second)
        self.assertLess(similarity, OK_PROTOTYPE_MERGE_SIMILARITY)
        self.assertLess(breakdown["context_similarity"], 0.97)

        prototypes = condense_ok_records(
            [_record(first, 1), _record(second, 2)],
            _dual_compare,
        )
        self.assertEqual(len(prototypes), 2)


class FakeKNN:
    def _load_all(self):
        return None

    def analyze(self, *args, **kwargs):
        del args, kwargs
        return {"best_similarity": 0.99}


class PrototypeDatasetManager(DatasetManager):
    pass


class PersistencePrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.best_module = types.SimpleNamespace(
            compare_anomaly_signatures=_dual_compare,
        )
        cls.dataset_module = types.SimpleNamespace(
            build_anomaly_signature=build_anomaly_signature,
        )
        install_prototype_memory(
            FakeKNN,
            PrototypeDatasetManager,
            cls.dataset_module,
            cls.best_module,
        )

    def setUp(self):
        self.old_normal = settings.NORMAL_DIR
        self.old_anomaly = settings.ANOMALY_DIR
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        settings.NORMAL_DIR = root / "normal"
        settings.ANOMALY_DIR = root / "anomaly"
        settings.NORMAL_DIR.mkdir(parents=True, exist_ok=True)
        settings.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        settings.NORMAL_DIR = self.old_normal
        settings.ANOMALY_DIR = self.old_anomaly
        self.temporary.cleanup()

    @staticmethod
    def _analysis(signature, label):
        return {
            "is_defect": label == "NG",
            "confidence": 0.99,
            "detail": {"anomaly_signature": signature},
        }

    def _save(self, style, label, *, part="C120", ai_decision=None):
        signature, reference, test = _dual_signature(style, part=part)
        return PrototypeDatasetManager.save_sample(
            ng_image=test,
            label=label,
            sample_image=reference,
            aoi_info={
                "board": "BOARD-A",
                "parts": part,
                "category": "FALTANDO",
            },
            analysis=self._analysis(signature, label),
            save_images=False,
            source="button",
            ai_decision=ai_decision or label,
        )

    def test_redundant_ok_updates_occurrence_counter_without_new_json(self):
        first_path = self._save("a", "OK")
        second_path = self._save("a", "OK")

        self.assertEqual(first_path, second_path)
        files = list(settings.NORMAL_DIR.rglob("*.json"))
        self.assertEqual(len(files), 1)

        data = json.loads(files[0].read_text(encoding="utf-8"))
        prototype = data["prototype"]
        self.assertEqual(prototype["schema"], PROTOTYPE_SCHEMA)
        self.assertEqual(prototype["occurrences"], 2)
        self.assertFalse(prototype["protected"])
        self.assertFalse(prototype["quantity_influence"])

    def test_distinct_ok_creates_another_prototype(self):
        self._save("a", "OK")
        self._save("b", "OK")

        files = list(settings.NORMAL_DIR.rglob("*.json"))
        self.assertEqual(len(files), 2)
        occurrences = sorted(
            json.loads(path.read_text(encoding="utf-8"))["prototype"]["occurrences"]
            for path in files
        )
        self.assertEqual(occurrences, [1, 1])

    def test_same_visual_pattern_on_other_component_is_not_persistently_merged(self):
        self._save("a", "OK", part="C120")
        self._save("a", "OK", part="C121")

        self.assertEqual(len(list(settings.NORMAL_DIR.rglob("*.json"))), 2)

    def test_every_ng_is_saved_as_individual_protected_memory(self):
        first_path = self._save("a", "NG")
        second_path = self._save("a", "NG")

        self.assertNotEqual(first_path, second_path)
        files = list(settings.ANOMALY_DIR.rglob("*.json"))
        self.assertEqual(len(files), 2)
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            prototype = data["prototype"]
            self.assertTrue(prototype["protected"])
            self.assertEqual(prototype["occurrences"], 1)
            self.assertEqual(prototype["compaction_policy"], "never_merge_ng")
            self.assertFalse(prototype["quantity_influence"])

    def test_statistics_do_not_modify_knn_result(self):
        expert = FakeKNN()
        expert.memory_prototype_stats = {
            "raw_ok_jsons": 400,
            "ok_prototypes": 20,
            "raw_ng_jsons": 3,
            "protected_ng_prototypes": 3,
            "quantity_influence": False,
        }
        result = expert.analyze()

        self.assertEqual(result["best_similarity"], 0.99)
        self.assertEqual(result["memory_prototype_stats"]["ok_prototypes"], 20)
        self.assertFalse(result["memory_quantity_influence"])


class InstallationOrderTests(unittest.TestCase):
    def test_prototype_layer_is_after_dual_scale(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_dual_scale_memory("),
            source.index("install_prototype_memory("),
        )
        self.assertLess(
            source.index("install_prototype_memory("),
            source.index("panel = ControlPanel()"),
        )


if __name__ == "__main__":
    unittest.main()
