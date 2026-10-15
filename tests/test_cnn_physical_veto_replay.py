"""Replay físico offline: vínculo AOI, SHA-256 e falta de luz."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.startup_regression.cnn_physical_veto_replay import replay_event


class PhysicalVetoReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / "reports" / "neural_online" / "events"
        self.folder.mkdir(parents=True)
        self.event_id = "f39c34cd51bb48abb7538afd6eec550e"

    def write_event(self, lights=("SIDE", "TOP", "MID"), label="NG"):
        images = []
        for light in lights:
            row = {"lighting_mode": light}
            for side in ("reference", "test"):
                picture = np.full((80, 94, 3), 95, dtype=np.uint8)
                picture[24:62, 20:75] = (20, 20, 20)
                success, png = cv2.imencode(".png", picture)
                self.assertTrue(success)
                content = png.tobytes()
                name = f"one_{light}_{side}.png"
                (self.folder/name).write_bytes(content)
                row[side] = name
                row[side + "_sha256"] = sha256(content).hexdigest()
            images.append(row)
        event = {
            "schema": "visionx.specialist_online_case.v1",
            "source_aoi_event_id": self.event_id,
            "event_id": "one", "category": "FALTANDO",
            "label": label, "human_source": "button", "images": images,
        }
        (self.folder/"one.json").write_text(json.dumps(event), encoding="utf-8")

    def test_verified_three_lights_keep_human_label_without_changing_files(self):
        self.write_event()
        before = sorted((p.name, sha256(p.read_bytes()).hexdigest())
                        for p in self.folder.iterdir())
        result = replay_event(self.root, self.event_id)
        self.assertEqual(result["state"], "JOURNAL_FOUND")
        self.assertTrue(result["read_only"])
        self.assertEqual(result["events"][0]["human_label"], "NG")
        self.assertEqual(set(result["events"][0]["lights"]), {"SIDE", "TOP", "MID"})
        self.assertEqual(result["events"][0]["suspect_lights"], [])
        self.assertEqual(result["events"][0]["unavailable_lights"], [])
        self.assertFalse(result["events"][0]["would_veto_cnn_all_ok"])
        self.assertEqual(before, sorted(
            (p.name, sha256(p.read_bytes()).hexdigest())
            for p in self.folder.iterdir()
        ))

    def test_missing_mid_requires_review_not_fabricated_ng(self):
        self.write_event(lights=("SIDE", "TOP"))
        result = replay_event(self.root, self.event_id)
        row = result["events"][0]
        self.assertEqual(row["unavailable_lights"], ["MID"])
        self.assertTrue(row["would_veto_cnn_all_ok"])
        self.assertEqual(row["human_label"], "NG")

    def test_modified_png_never_counts_as_valid_physical_evidence(self):
        self.write_event()
        (self.folder/"one_SIDE_reference.png").write_bytes(b"corrupt")
        result = replay_event(self.root, self.event_id)
        self.assertEqual(result["events"][0]["lights"]["SIDE"]["status"], "UNAVAILABLE")

    def test_unknown_event_is_not_guessed_from_filenames(self):
        self.write_event()
        result = replay_event(self.root, "different-event")
        self.assertEqual(result["state"], "NO_JOURNAL")
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
