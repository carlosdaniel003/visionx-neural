"""Regressões de observações por evento sem duplicar capturas visuais."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.services.event_observation_journal import (
    EventObservationQueue, _active_snapshot, install_event_observation_journal,
)
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.operational_event_replay import (
    verified_event_manifests, replay,
)
from tests.test_operational_event_replay import FakeRuntime


def panel(event="abcdef0123456789", source="button"):
    img = np.zeros((64, 82, 3), dtype=np.uint8)
    img[9:36, 10:35] = 124
    return SimpleNamespace(
        capture_cycle_source="network",
        current_aoi_info={"category": "FALTANDO"},
        current_analysis={"verdict": "FALHA FALSA",
                          "is_defect": False, "detail": {}},
        adhesive_multilight_last_event_id=event,
        adhesive_multilight_last_source_frames={"SIDE": img},
        adhesive_multilight_last_analyses={},
        last_decision_command_success=None,
    ), img


class EventObservationTests(unittest.TestCase):
    def test_side_event_journal_only_saves_json_and_exact_frame_fingerprint(self):
        instance, image = panel()
        with patch(
            "src.services.event_observation_journal.network_xp_record_event_id",
            return_value="abcdef0123456789",
        ), tempfile.TemporaryDirectory() as temp:
            data = _active_snapshot(instance, "OK", "button")
            self.assertEqual(data["expected_label"], "OK")
            self.assertEqual(data["label_provenance"], "OPERATOR_INPUT")
            self.assertEqual(data["analyzed_lights"], ["SIDE"])
            self.assertEqual(data["skipped_lights"], ["TOP", "MID"])
            q = EventObservationQueue(Path(temp))
            self.assertTrue(q.submit(data))
            q.wait_until_idle()
            folder = Path(temp) / "reports/startup_regression/event_observations"
            docs = list(folder.glob("*.json"))
            self.assertEqual(len(docs), 1)
            self.assertEqual(list(folder.glob("*.png")), [])
            record = json.loads(docs[0].read_text(encoding="utf-8"))
            self.assertEqual(record["frames"]["SIDE"]["pixel_sha256"],
                             image_fingerprint(image))
            self.assertNotIn("_frames", record)
            self.assertTrue(q.submit(_active_snapshot(instance, "OK", "button")))
            q.wait_until_idle()
            self.assertEqual(len(list(folder.glob("*.json"))), 1)

    def test_auto_decisions_never_become_human_ground_truth(self):
        instance, image = panel()
        with patch(
            "src.services.event_observation_journal.network_xp_record_event_id",
            return_value="abcdef0123456789"
        ):
            auto = _active_snapshot(instance, "NG", "production_auto")
        self.assertEqual(auto["label_provenance"], "AUTOMATIC_DECISION")
        self.assertIsNone(auto["expected_label"])

    def test_wrong_event_and_stale_capture_never_registered(self):
        instance, _ = panel()
        with patch(
            "src.services.event_observation_journal.network_xp_record_event_id",
            return_value="another-event-id",
        ), patch(
            "src.services.event_observation_journal.network_xp_frame_snapshot",
            return_value=None,
        ):
            self.assertIsNone(_active_snapshot(instance, "OK", "button"))

    def test_full_cycle_keeps_all_three_illuminations(self):
        instance, image = panel()
        instance.adhesive_multilight_last_source_frames = {
            mode: image.copy() for mode in ("SIDE", "TOP", "MID")
        }
        with patch(
            "src.services.event_observation_journal.network_xp_record_event_id",
            return_value="abcdef0123456789",
        ):
            doc = _active_snapshot(instance, "NG", "xp_keyboard")
        self.assertEqual(doc["analyzed_lights"], ["SIDE", "TOP", "MID"])
        self.assertEqual(doc["skipped_lights"], [])

    def test_wrapper_does_not_turn_failed_production_command_into_label(self):
        class Panel:
            def save_label(self, user_decision, source="button"):
                self.last_decision_command_success = False
                return False

        install_event_observation_journal(Panel)
        instance = Panel()
        instance.current_analysis = {"verdict": "DEFEITO REAL"}
        with patch(
            "src.services.event_observation_journal._active_snapshot",
            return_value={"event_id": "abcdef0123456789"},
        ), patch(
            "src.services.event_observation_journal.EventObservationQueue",
        ) as queue:
            self.assertFalse(instance.save_label("NG", source="production_auto"))
            queue.assert_not_called()


class EventQualificationTests(unittest.TestCase):
    def _archive_and_observe(self, root: Path, label="OK", *,
                             decision_source="button", keep=True):
        image = np.zeros((64, 82, 3), dtype=np.uint8)
        image[12:37, 5:32] = 90
        archive = root / "public" / ("ok_archive" if label=="OK" else "ng_archive")
        archive.mkdir(parents=True, exist_ok=True)
        (root / "public" / ("ng_archive" if label=="OK" else "ok_archive")).mkdir(
            parents=True, exist_ok=True
        )
        if keep:
            self.assertTrue(cv2.imwrite(
                str(archive / "2026-10-22_0800_FALTANDO.png"), image
            ))
        q = EventObservationQueue(root)
        record = {
            "schema": "visionx.event_observation.v1",
            "event_id": "abcdef0123456789",
            "category": "FALTANDO",
            "decision_label": label,
            "expected_label": label if decision_source=="button" else None,
            "label_provenance": ("OPERATOR_INPUT" if decision_source=="button"
                                 else "AUTOMATIC_DECISION"),
            "_frames": {"SIDE": image},
        }
        self.assertTrue(q.submit(record))
        q.wait_until_idle()

    def test_journal_linked_by_exact_pixels_and_side_only_replays_one_light(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._archive_and_observe(root)
            inventory = inventory_archives(root)
            events, coverage = verified_event_manifests(inventory, root=root)
            self.assertEqual(coverage["qualified_event_count"], 1)
            self.assertEqual(coverage["qualified_journal_events"], 1)
            self.assertEqual(set(events[0]["frame_paths"]), {"SIDE"})
            replayed = replay(
                root, inventory=inventory,
                runtime_factory=lambda r: FakeRuntime(
                    r, early=True, verdict="FALHA FALSA"),
            )
            self.assertEqual(replayed["counts"]["passed"], 1)
            self.assertEqual(replayed["events"][0]["analyzed_lights"], ["SIDE"])

    def test_new_event_missing_retained_png_stays_unqualified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self._archive_and_observe(root, keep=False)
            events, coverage=verified_event_manifests(
                inventory_archives(root), root=root
            )
            self.assertEqual(events, [])
            self.assertEqual(
                coverage["unusable_manifests"]["EVIDENCE_NOT_RETAINED"], 1
            )

    def test_auto_label_excluded_even_when_file_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self._archive_and_observe(root, decision_source="production_auto")
            events, coverage=verified_event_manifests(
                inventory_archives(root), root=root
            )
            self.assertEqual(events, [])
            self.assertEqual(
                coverage["unusable_manifests"]["AUTOMATIC_UNVERIFIED_LABEL"], 1
            )


if __name__=="__main__":
    unittest.main()
