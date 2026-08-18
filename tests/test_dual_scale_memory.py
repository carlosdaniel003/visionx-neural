import json
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import build_anomaly_signature
from src.core.dual_scale_memory import (
    CONTEXT_WEIGHT,
    EPICENTER_WEIGHT,
    attach_component_context,
    build_component_context_signature,
    compare_component_context_signatures,
    install_dual_scale_memory,
    valid_context_signature,
)
from src.services.dataset_manager import DatasetManager


ROOT = Path(__file__).resolve().parents[1]


def _scene(style: str = "a", anomaly: bool = False) -> np.ndarray:
    image = np.full((240, 320, 3), 28, dtype=np.uint8)
    cv2.rectangle(image, (38, 38), (282, 202), (0, 255, 0), 4)

    if style == "a":
        image[55:185, 62:258] = (92, 92, 92)
        cv2.rectangle(image, (88, 78), (232, 164), (155, 155, 155), -1)
        cv2.line(image, (100, 120), (220, 120), (35, 35, 35), 7)
        cv2.circle(image, (130, 105), 13, (190, 190, 190), -1)
    elif style == "b":
        image[55:185, 62:258] = (35, 70, 155)
        for y in range(65, 180, 24):
            for x in range(72, 250, 32):
                color = (185, 55, 35) if ((x + y) // 24) % 2 else (25, 175, 190)
                cv2.rectangle(image, (x, y), (x + 22, y + 16), color, -1)
    else:
        image[55:185, 62:258] = (125, 80, 45)

    if anomaly:
        cv2.rectangle(image, (148, 108), (174, 136), (245, 245, 245), -1)
        cv2.line(image, (150, 110), (172, 134), (20, 20, 20), 3)
    return image


class DualScaleSignatureTests(unittest.TestCase):
    def test_context_signature_contains_both_context_appearances(self):
        reference = _scene("a", anomaly=False)
        test = _scene("a", anomaly=True)

        context = build_component_context_signature(reference, test)

        self.assertTrue(valid_context_signature(context))
        self.assertEqual(len(context["reference_embedding_128"]), 128)
        self.assertEqual(len(context["test_embedding_128"]), 128)
        self.assertEqual(len(context["semantic_delta_128"]), 128)
        self.assertEqual(np.asarray(context["spatial_delta_4x4"]).shape, (4, 4))
        self.assertEqual(np.asarray(context["difference_map_8x8"]).shape, (8, 8))
        self.assertGreater(context["context_box"][2], 200)
        self.assertGreater(context["context_box"][3], 120)
        json.dumps(context)

    def test_attaching_context_does_not_change_existing_epicenter_vector(self):
        reference = _scene("a", anomaly=False)
        test = _scene("a", anomaly=True)
        local = build_anomaly_signature(
            reference,
            test,
            {},
            {"category": "FALTANDO"},
            (140, 100, 42, 42),
        )
        original_vector = list(local["vector"])

        expanded = attach_component_context(local, reference, test)

        self.assertEqual(expanded["vector"], original_vector)
        self.assertEqual(expanded["vector_size"], local["vector_size"])
        self.assertEqual(expanded["memory_scales"], ["epicenter", "component_context"])
        self.assertTrue(valid_context_signature(expanded["context_signature"]))
        self.assertEqual(expanded["scale_weights"]["epicenter"], EPICENTER_WEIGHT)
        self.assertEqual(expanded["scale_weights"]["component_context"], CONTEXT_WEIGHT)

    def test_context_distinguishes_different_component_appearance(self):
        query = build_component_context_signature(_scene("a", False), _scene("a", True))
        same = build_component_context_signature(_scene("a", False), _scene("a", True))
        different = build_component_context_signature(_scene("b", False), _scene("b", True))

        same_score, _ = compare_component_context_signatures(query, same)
        different_score, _ = compare_component_context_signatures(query, different)

        self.assertGreater(same_score, 0.99)
        self.assertLess(different_score, same_score - 0.08)

    def test_dataset_json_persists_epicenter_and_component_context(self):
        reference = _scene("a", False)
        test = _scene("a", True)
        local = build_anomaly_signature(
            reference,
            test,
            {},
            {"category": "FALTANDO", "parts": "C100"},
            (140, 100, 42, 42),
        )
        expanded = attach_component_context(local, reference, test)

        old_normal = settings.NORMAL_DIR
        old_anomaly = settings.ANOMALY_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings.NORMAL_DIR = root / "normal"
            settings.ANOMALY_DIR = root / "anomaly"
            try:
                path = DatasetManager.save_sample(
                    ng_image=test,
                    label="NG",
                    sample_image=reference,
                    aoi_info={"category": "FALTANDO", "parts": "C100"},
                    analysis={
                        "is_defect": True,
                        "confidence": 0.99,
                        "detail": {"anomaly_signature": expanded},
                    },
                    save_images=False,
                    source="button",
                    ai_decision="NG",
                )
                self.assertTrue(path)
                saved = json.loads(Path(path).read_text(encoding="utf-8"))
            finally:
                settings.NORMAL_DIR = old_normal
                settings.ANOMALY_DIR = old_anomaly

        memory = saved["analysis"]["anomaly_memory"]
        self.assertEqual(memory["vector"], expanded["vector"])
        self.assertEqual(memory["memory_scales"], ["epicenter", "component_context"])
        self.assertTrue(valid_context_signature(memory["context_signature"]))
        self.assertEqual(memory["context_signature"]["schema"], "visionx.context.v1")


class DualScaleInstallerTests(unittest.TestCase):
    @staticmethod
    def _local_builder(reference, test, detail, aoi_info=None, focus_box=None):
        del reference, test, detail, aoi_info, focus_box
        return {
            "schema": "visionx.anomaly.v1",
            "vector_size": 224,
            "vector": [0.25] * 224,
        }

    def _modules(self, local_similarity: float):
        anomaly_module = types.SimpleNamespace(build_anomaly_signature=self._local_builder)
        best_module = types.SimpleNamespace(
            compare_anomaly_signatures=lambda query, stored: (
                float(local_similarity),
                {"schema": "local", "groups": {}},
            ),
        )
        dataset_module = types.SimpleNamespace(build_anomaly_signature=self._local_builder)
        install_dual_scale_memory(anomaly_module, best_module, dataset_module)
        return anomaly_module, best_module, dataset_module

    def test_dual_scale_similarity_uses_local_and_context(self):
        anomaly_module, best_module, dataset_module = self._modules(1.0)
        query = anomaly_module.build_anomaly_signature(_scene("a", False), _scene("a", True), {})
        stored = dataset_module.build_anomaly_signature(_scene("b", False), _scene("b", True), {})

        similarity, breakdown = best_module.compare_anomaly_signatures(query, stored)

        self.assertTrue(breakdown["dual_scale"])
        self.assertAlmostEqual(breakdown["epicenter_similarity"], 1.0, places=7)
        self.assertLess(breakdown["context_similarity"], 0.92)
        expected = (
            breakdown["epicenter_similarity"] * EPICENTER_WEIGHT
            + breakdown["context_similarity"] * CONTEXT_WEIGHT
        )
        self.assertAlmostEqual(similarity, expected, places=7)
        self.assertLess(similarity, 1.0)

    def test_legacy_memory_without_context_keeps_old_similarity(self):
        anomaly_module, best_module, _ = self._modules(0.83)
        query = anomaly_module.build_anomaly_signature(_scene("a", False), _scene("a", True), {})
        legacy = {
            "schema": "visionx.anomaly.v1",
            "vector_size": 224,
            "vector": [0.25] * 224,
        }

        similarity, breakdown = best_module.compare_anomaly_signatures(query, legacy)

        self.assertAlmostEqual(similarity, 0.83, places=7)
        self.assertFalse(breakdown["dual_scale"])
        self.assertEqual(breakdown["policy"], "legacy_epicenter_only")
        self.assertIsNone(breakdown["context_similarity"])

    def test_dataset_fallback_builder_also_receives_second_scale(self):
        _, _, dataset_module = self._modules(0.9)
        signature = dataset_module.build_anomaly_signature(_scene("a", False), _scene("a", True), {})
        self.assertIn("context_signature", signature)
        self.assertTrue(valid_context_signature(signature["context_signature"]))
        json.dumps(signature)


class InstallationOrderTests(unittest.TestCase):
    def test_dual_scale_is_installed_after_best_match(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_best_match_memory(KNNExpert, anomaly_memory_module)"),
            source.index("install_dual_scale_memory("),
        )


if __name__ == "__main__":
    unittest.main()
