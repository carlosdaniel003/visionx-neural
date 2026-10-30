"""ODIN ao vivo: primeira decisão humana alimenta KNN e experiência CNN V2."""
from __future__ import annotations

from pathlib import Path
import tempfile
from functools import partial
from unittest.mock import patch
import json
import unittest

import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.neural.faltando_live import FaltandoCNNLive
from src.core.verified_memory_router import VerifiedKNNMemory
from src.services.live_session_learning import immediate_confirmed_memory
from src.services.anomaly_learning import install_anomaly_learning
from src.services.neural_online_learning import (
    eligible_online_case, OnlineLearningQueue,
)


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

    def test_latest_human_label_overwrites_old_for_same_rgb_without_erasing_history(self):
        from unittest.mock import patch
        first = labeled_task("NG")
        first["event_id"] = "older-aoi-event"
        newer = labeled_task("OK")
        newer["event_id"] = "new-aoi-event"
        with patch("src.services.live_session_learning.time.time_ns",
                   side_effect=[100, 200]):
            immediate_confirmed_memory(first, store=self.store)
            immediate_confirmed_memory(newer, store=self.store)
        result = self.store.lookup(
            newer["sample_image"], newer["ng_image"], newer["aoi_info"],
        )
        self.assertEqual((result["status"], result["label"]), ("KNOWN", "OK"))
        self.assertEqual(result["confirmed_at_ns"], 200)
        self.assertEqual(result["source_event_id"], "new-aoi-event")
        self.assertEqual(result["superseded_labels"], ["NG"])
        self.assertEqual(len(self.store.records()), 2)  # History preserved
        verified = VerifiedKNNMemory(archive_store=self.store)
        looked_up = verified.lookup(
            None, newer["sample_image"], newer["ng_image"], newer["aoi_info"],
        )
        self.assertEqual(looked_up["label"], "OK")
        self.assertEqual(
            looked_up["memory_source"], "LATEST_OPERATOR_EXACT_PAIR"
        )
        # All new processes must reach the same winner without retraining.
        reboot = PersistentArchiveKNN(self.root)
        self.assertEqual(reboot.lookup(
            newer["sample_image"], newer["ng_image"], newer["aoi_info"],
        )["label"], "OK")

    def test_latest_human_can_switch_back_ng_and_keep_other_cases_unchanged(self):
        from unittest.mock import patch
        with patch("src.services.live_session_learning.time.time_ns",
                   side_effect=[100, 200, 300]):
            immediate_confirmed_memory(labeled_task("NG"), store=self.store)
            immediate_confirmed_memory(labeled_task("OK"), store=self.store)
            immediate_confirmed_memory(labeled_task("NG"), store=self.store)
        task = labeled_task()
        self.assertEqual(self.store.lookup(
            task["sample_image"], task["ng_image"], task["aoi_info"],
        )["label"], "NG")
        different = labeled_task(value=120)
        self.assertEqual(self.store.lookup(
            different["sample_image"], different["ng_image"],
            different["aoi_info"],
        )["status"], "NEW")
        different_category = labeled_task(category="INVERTIDO")
        self.assertEqual(self.store.lookup(
            different_category["sample_image"],
            different_category["ng_image"],
            different_category["aoi_info"],
        )["status"], "NEW")
        different_light = labeled_task(light="TOP")
        self.assertEqual(self.store.lookup(
            different_light["sample_image"], different_light["ng_image"],
            different_light["aoi_info"],
        )["status"], "NEW")

    def test_latest_operator_overrides_exact_archived_ng_without_relabelling_png(self):
        from src.core.archive_persistent_memory import make_record
        from src.core.anomaly_signature import build_anomaly_signature
        task = labeled_task("OK")
        ref, test = task["sample_image"], task["ng_image"]
        info = task["aoi_info"]
        self.store.add([make_record(
            label="NG",
            source_path="public/ng_archive/old_wrong.png",
            source_sha256="1" * 64,
            category="FALTANDO", lighting="SIDE",
            reference=ref, test=test, aoi_info=info,
            anomaly_signature=build_anomaly_signature(ref, test, {}, info),
        )])
        self.assertEqual(self.store.lookup(ref, test, info)["label"], "NG")
        immediate_confirmed_memory(task, store=self.store)
        self.assertEqual(self.store.lookup(ref, test, info)["label"], "OK")
        verified = VerifiedKNNMemory(archive_store=self.store)
        result = verified.lookup(None, ref, test, info)
        self.assertEqual((result["status"], result["label"]), ("KNOWN", "OK"))
        self.assertEqual(len(self.store.records()), 2)
        self.assertEqual(
            {row["label"] for row in self.store.records()}, {"NG", "OK"}
        )

    def test_opposite_operator_labels_without_timestamps_still_conflict(self):
        from src.core.archive_persistent_memory import make_operator_record
        from src.core.anomaly_signature import build_anomaly_signature
        task = labeled_task()
        ref, test = task["sample_image"], task["ng_image"]
        info = task["aoi_info"]
        signature = build_anomaly_signature(ref, test, {}, info)
        for label in ("OK", "NG"):
            self.store.add([make_operator_record(
                label=label, source="button", event_id=label,
                category="FALTANDO", lighting="SIDE", reference=ref,
                test=test, anomaly_signature=signature, aoi_info=info,
            )])
        self.assertEqual(
            self.store.lookup(ref, test, info)["status"], "CONFLICT"
        )

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

    def test_real_label_hook_commits_before_interface_releases_next_case(self):
        from types import SimpleNamespace

        class Button:
            def setEnabled(self, value):
                self.enabled = value
            def setText(self, value):
                self.text = value

        class Panel:
            def __init__(self):
                task = labeled_task()
                self.current_sample = task["sample_image"]
                self.current_ng = task["ng_image"]
                self.current_aoi_info = task["aoi_info"]
                self.current_analysis = task["analysis"]
                self.combo_mode = SimpleNamespace(currentText=lambda: "Modo Sombra")
                self.capture_cycle_source = "network"
                self.adhesive_multilight_learning_samples = {}
                self.adhesive_multilight_primary_event_id = ""
                self.btn_save_ok = Button()
                self.btn_save_ng = Button()
                self.btn_skip = Button()
                self.btn_start = Button()
                self._live_memory_writer = partial(
                    immediate_confirmed_memory, store=memory
                )
                self._anomaly_persistence_submitter = lambda task: collected.append(task)
                self.orchestrator = SimpleNamespace(experts={})
                self.advance_label = None
            def save_label(self, *args, **kwargs):
                raise AssertionError("No legacy save should be required")
            def send_command_to_xp(self, value):
                return True
            def prepare_for_next_network_image(self):
                match = memory.lookup(
                    self.current_sample, self.current_ng, self.current_aoi_info
                )
                self.advance_label = match.get("label")
            def update_history_status(self, *args):
                pass

        memory = self.store
        collected = []
        install_anomaly_learning(Panel)
        panel = Panel()
        panel.save_label("OK", source="xp_keyboard")
        self.assertEqual(panel.advance_label, "OK")
        self.assertEqual(len(collected), 1)
        self.assertEqual(self.store.records()[0]["label"], "OK")

    def test_online_cnn_eligibility_uses_all_4_categories_and_human_source(self):
        for category in ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO"):
            task = labeled_task(category=category)
            self.assertTrue(eligible_online_case(task))
            task["source"] = "production_auto"
            self.assertFalse(eligible_online_case(task))
            task["source"] = "button"
            task["analysis"]["detail"]["recognition_route"] = "KNOWN_KNN"
            self.assertFalse(eligible_online_case(task))

    def test_partial_shadow_cnn_journal_survives_archive_deletion(self):
        task = labeled_task("OK")
        ref, test = task["sample_image"], task["ng_image"]
        task["multilight_samples"] = [
            {"lighting_mode": light, "sample_image": ref, "test_image": test}
            for light in ("SIDE", "TOP")
        ]
        queue = OnlineLearningQueue(root=self.root, start_worker=False)
        event_id = queue.submit_saved(task)
        self.assertTrue(event_id)
        journal = self.root / "reports" / "neural_online" / "events"
        row = json.loads((journal / (event_id + ".json")).read_text("utf-8"))
        self.assertEqual(len(row["images"]), 2)
        self.assertEqual(
            {r["lighting_mode"] for r in row["images"]}, {"SIDE", "TOP"}
        )
        for item in row["images"]:
            self.assertTrue((journal / item["reference"]).is_file())
            self.assertTrue((journal / item["test"]).is_file())
        # Network archive PNGs can now be deleted; CNN samples are
        # separate durable data and replays remain possible.
        self.assertFalse((self.root / "public" / "ok_archive").exists())

    def test_human_dedup_across_repeated_confirmations(self):
        task = labeled_task()
        self.assertEqual(immediate_confirmed_memory(
            task, store=self.store)["new"], 1)
        self.assertEqual(immediate_confirmed_memory(
            task, store=self.store)["new"], 0)
        self.assertEqual(len(self.store.records()), 1)


if __name__ == "__main__":
    unittest.main()
