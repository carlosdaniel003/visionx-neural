"""ODIN ao vivo: primeira decisão humana alimenta KNN e experiência CNN V2."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.neural.faltando_live import FaltandoCNNLive
from src.core.verified_memory_router import VerifiedKNNMemory
from src.services.live_session_learning import immediate_confirmed_memory


def pair(color=80):
    ref = np.zeros((72, 84, 3), dtype=np.uint8)
    ref[9:55, 15:70] = (15, 30, 50)
    test = ref.copy()
    test[25:42, 27:51] = (0, color, 5)
    return ref, test


def labeled_task(label="OK", *, category="FALTANDO",
                 source="xp_keyboard", light="SIDE", value=80):
    reference, test = pair(value)
    return {
        "source": source, "label": label,
        "event_id": "event-123",
        "sample_image": reference, "ng_image": test,
        "aoi_info": {
            "board": "TEST-01", "parts": "U2~5",
            "category": category, "lighting_mode": light,
            "value": "0 <= 80 <= 90 " + category,
        },
        "analysis": {
            "is_defect": False, "detail": {"recognition_route": "NEW_CNN"}
        },
        "multilight_samples": [], "shadow_partial_samples": [],
    }


class LiveSessionLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = PersistentArchiveKNN(self.root)

    def test_first_human_ok_then_nine_identical_recalled_same_session(self):
        first = labeled_task()
        response = immediate_confirmed_memory(first, store=self.store)
        self.assertTrue(response["committed"])
        self.assertEqual(response["new"], 1)
        lookup = VerifiedKNNMemory(archive_store=self.store)
        for _ in range(9):
            result = lookup.lookup(
                None, first["sample_image"], first["ng_image"],
                first["aoi_info"],
            )
            self.assertEqual(result["status"], "KNOWN")
            self.assertEqual(result["label"], "OK")
        self.assertEqual(len(self.store.records()), 1)

    def test_first_human_ng_then_nine_identical_recalled(self):
        first = labeled_task("NG")
        immediate_confirmed_memory(first, store=self.store)
        for _ in range(9):
            result = PersistentArchiveKNN(self.root).lookup(
                first["sample_image"], first["ng_image"],
                first["aoi_info"]
            )
            self.assertEqual((result["status"], result["label"]),
                             ("KNOWN", "NG"))

    def test_recall_survives_reboot_and_deleted_archives(self):
        task = labeled_task()
        immediate_confirmed_memory(task, store=self.store)
        renewed = PersistentArchiveKNN(self.root)
        result = renewed.lookup(
            task["sample_image"], task["ng_image"], task["aoi_info"]
        )
        self.assertEqual(result["label"], "OK")
        # Event provenance is operator_confirmed/, no archival PNG is required.
        self.assertTrue(
            self.store.records()[0]["sources"][0]["source_path"].startswith(
                "operator_confirmed/"
            )
        )

    def test_auto_predictions_never_self_label_or_train(self):
        for source in ("auto", "production_auto", "cnn", "knn", ""):
            response = immediate_confirmed_memory(
                labeled_task(source=source), store=self.store
            )
            self.assertFalse(response["committed"])
        self.assertFalse(self.store.path.exists())

    def test_opposite_operator_labels_same_pixels_are_conflict(self):
        immediate_confirmed_memory(labeled_task("OK"), store=self.store)
        immediate_confirmed_memory(labeled_task("NG"), store=self.store)
        task = labeled_task()
        self.assertEqual(self.store.lookup(
            task["sample_image"], task["ng_image"], task["aoi_info"]
        )["status"], "CONFLICT")

    def test_cnn_v2_experience_without_checkpoint_is_exact_only(self):
        task = labeled_task()
        immediate_confirmed_memory(task, store=self.store)
        cnn = FaltandoCNNLive(
            checkpoint_path=self.root / "nonexistent.pt",
            online_root=self.root,
        )
        recognized = cnn.recognize_confirmed(
            task["sample_image"], task["ng_image"], "FALTANDO", "SIDE"
        )
        self.assertEqual(recognized["verdict"], "FALHA FALSA")
        self.assertEqual(recognized["detail"]["recognition_route"], "KNOWN_KNN")
        self.assertTrue(recognized["detail"]["cnn_v2_experience_memory"])
        self.assertFalse(recognized["detail"]["cnn_v2_weight_update"])
        self.assertIsNone(cnn._model)
        self.assertIsNone(cnn.recognize_confirmed(
            *pair(100), "FALTANDO", "SIDE"
        ))
        self.assertIsNone(cnn.recognize_confirmed(
            task["sample_image"], task["ng_image"], "MUITO ADESIVO", "SIDE"
        ))

    def test_cnn_v2_scope_and_per_light_are_separate(self):
        for category in ("EMBORCADO", "INVERTIDO", "DESLOCADO"):
            task = labeled_task("OK", category=category)
            immediate_confirmed_memory(task, store=self.store)
            cnn = FaltandoCNNLive(online_root=self.root)
            self.assertEqual(cnn.recognize_confirmed(
                task["sample_image"], task["ng_image"], category, "SIDE"
            )["verdict"], "FALHA FALSA")
            self.assertIsNone(cnn.recognize_confirmed(
                task["sample_image"], task["ng_image"], category, "MID"
            ))

    def test_shadow_partial_two_lights_memorized(self):
        task = labeled_task("NG", source="xp_keyboard")
        ref, test = task["sample_image"], task["ng_image"]
        task["shadow_partial_samples"] = [
            {"lighting_mode": "SIDE", "sample_image": ref, "test_image": test},
            {"lighting_mode": "TOP", "sample_image": ref, "test_image": test},
        ]
        result = immediate_confirmed_memory(task, store=self.store)
        self.assertEqual(result["new"], 2)
        self.assertEqual(len(self.store.records()), 2)
        self.assertTrue(all(r["label"] == "NG"
                            for r in self.store.records()))

    def test_human_dedup_across_repeated_confirmations(self):
        task = labeled_task()
        self.assertEqual(immediate_confirmed_memory(
            task, store=self.store)["new"], 1)
        self.assertEqual(immediate_confirmed_memory(
            task, store=self.store)["new"], 0)
        self.assertEqual(len(self.store.records()), 1)


if __name__ == "__main__":
    unittest.main()
