"""Regressão do diagnóstico local KNN NG/OK vs treino real da CNN."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.services.startup_regression.faltando_archive_neighbors_audit import (
    audit_event_neighbors,
)

LIGHTS = ("SIDE", "TOP", "MID")
EVENT = "239dc35fdce44b8190b6b69200ceee90"


class HumanEventNeighborAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folder = self.root / "reports/neural_online/events"
        self.folder.mkdir(parents=True)
        self.reference = np.full((80, 100, 3), 110, dtype=np.uint8)
        self.reference[20:60, 28:76] = (20, 20, 20)
        self.test = self.reference.copy()
        self.test[20:60, 28:76] = (80, 150, 70)

    def save_human_event(self):
        items = []
        for light in LIGHTS:
            row = {"lighting_mode": light}
            for key, image in (("reference", self.reference), ("test", self.test)):
                success, buf = cv2.imencode(".png", image)
                self.assertTrue(success)
                payload = buf.tobytes()
                name = f"one_{light}_{key}.png"
                (self.folder / name).write_bytes(payload)
                row[key] = name
                row[key + "_sha256"] = sha256(payload).hexdigest()
            items.append(row)
        (self.folder / "one.json").write_text(json.dumps({
            "schema": "visionx.specialist_online_case.v1",
            "event_id": "one", "source_aoi_event_id": EVENT,
            "label": "NG", "category": "FALTANDO",
            "human_source": "button", "images": items,
        }), encoding="utf-8")

    def record(self, label="NG", light="SIDE"):
        sig = build_anomaly_signature(
            self.reference, self.test, {},
            {"category": "FALTANDO", "lighting_mode": light},
        )
        return {
            "mode": "anomaly", "category": "FALTANDO",
            "lighting_mode": light, "label": label,
            "path": f"public/{label.lower()}_archive/example.png",
            "archive_persistent": True,
            "anomaly_signature": sig,
        }

    def test_three_lights_retrieve_ng_neighbors_but_do_not_vote(self):
        self.save_human_event()
        records_ng = [self.record(light=l) for l in LIGHTS]
        result = audit_event_neighbors(
            self.root, EVENT,
            records_ok=[self.record(label="OK")],
            records_ng=records_ng, top_k=2,
        )
        self.assertTrue(result["read_only"])
        self.assertFalse(result["decision_authorized"])
        self.assertFalse(result["active_cnn_weights_changed"])
        self.assertEqual(result["journal_state"], "JOURNAL_FOUND")
        case = result["events"][0]
        self.assertEqual(case["human_label"], "NG")
        self.assertEqual(case["learning_job_status"], "PENDING")
        self.assertIsNone(case["checkpoint_promoted_by_job"])
        for light in LIGHTS:
            record = case["lighting"][light]
            self.assertEqual(record["status"], "ANALYZED")
            self.assertEqual(record["neighbors"]["NG"]["candidates"], 1)
            self.assertEqual(
                record["neighbors"]["NG"]["nearest"][0]["source_kind"],
                "PERSISTENT_ARCHIVE",
            )
            if light == "SIDE":
                self.assertEqual(record["neighbors"]["OK"]["candidates"], 1)
            else:
                self.assertEqual(record["neighbors"]["OK"]["candidates"], 0)

    def test_unknown_event_does_not_create_fake_memory(self):
        self.save_human_event()
        result = audit_event_neighbors(
            self.root, "not-that-event",
            records_ok=[], records_ng=[],
        )
        self.assertEqual(result["journal_state"], "NO_JOURNAL")
        self.assertEqual(result["events"], [])

    def test_tampered_png_is_not_compared_against_archive(self):
        self.save_human_event()
        (self.folder / "one_MID_test.png").write_bytes(b"tampered")
        report = audit_event_neighbors(
            self.root, EVENT,
            records_ok=[], records_ng=[self.record("NG", "MID")],
        )
        mid = report["events"][0]["lighting"]["MID"]
        self.assertEqual(mid["status"], "UNAVAILABLE")
        self.assertEqual(mid["reason"], "ValueError")
        self.assertIsNone(mid["neighbors"])

    def test_auto_label_is_not_a_trusted_journal(self):
        self.save_human_event()
        p = self.folder / "one.json"
        event = json.loads(p.read_text(encoding="utf-8"))
        event["human_source"] = "production_auto"
        p.write_text(json.dumps(event), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fonte humana"):
            audit_event_neighbors(
                self.root, EVENT, records_ok=[], records_ng=[],
            )


if __name__ == "__main__":
    unittest.main()
