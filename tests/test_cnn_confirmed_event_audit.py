"""Auditoria de evento supervisionado CNN: journal e bytes, sem pesos."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.services.startup_regression.cnn_confirmed_event_audit import (
    audit_confirmed_event,
)


class ConfirmedCNNAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.events = self.root / "reports" / "neural_online" / "events"
        self.events.mkdir(parents=True)
        self.aoi = "f39c34cd51bb48abb7538afd6eec550e"

    def write_case(self, label="NG", lights=("SIDE", "TOP", "MID")):
        items = []
        for light in lights:
            row = {"lighting_mode": light}
            for part in ("reference", "test"):
                data = f"{light}-{part}".encode()
                name = f"one_{light}_{part}.png"
                (self.events / name).write_bytes(data)
                row[part] = name
                row[part + "_sha256"] = sha256(data).hexdigest()
            items.append(row)
        event = {
            "schema": "visionx.specialist_online_case.v1",
            "event_id": "one", "source_aoi_event_id": self.aoi,
            "human_source": "button", "label": label,
            "category": "FALTANDO", "recognition_route": "NEW_CNN",
            "images": items,
        }
        (self.events / "one.json").write_text(json.dumps(event), encoding="utf-8")

    def test_ng_three_lights_verified_without_claiming_trained(self):
        self.write_case()
        audit = audit_confirmed_event(self.root, self.aoi)
        self.assertEqual(audit["state"], "JOURNAL_FOUND")
        case = audit["events"][0]
        self.assertEqual(case["label"], "NG")
        self.assertTrue(case["assets_verified"])
        self.assertEqual(set(case["views"]), {"SIDE", "TOP", "MID"})
        self.assertEqual(case["job_status"], "PENDING")
        self.assertIsNone(case["promoted"])

    def test_explicit_rejection_is_not_successful_learning(self):
        self.write_case()
        status = self.root / "reports" / "neural_online" / "statuses"
        outcome = self.root / "reports" / "neural_online" / "outcomes"
        status.mkdir()
        outcome.mkdir()
        (status / "one.json").write_text(json.dumps({
            "status": "REJECTED", "exit_code": 0
        }), encoding="utf-8")
        (outcome / "one.json").write_text(json.dumps({
            "promoted": False, "validation_report": "reports/failed.json"
        }), encoding="utf-8")
        result = audit_confirmed_event(self.root, self.aoi)["events"][0]
        self.assertEqual(result["job_status"], "REJECTED")
        self.assertFalse(result["promoted"])
        self.assertFalse(result["checkpoint_active_for_event"])

    def test_missing_and_corrupt_assets_detected(self):
        self.write_case()
        target = self.events / "one_MID_test.png"
        target.write_bytes(b"not-the-original")
        result = audit_confirmed_event(self.root, self.aoi)["events"][0]
        self.assertFalse(result["assets_verified"])
        self.assertEqual(result["views"]["MID"]["test"]["reason"],
                         "SHA256_MISMATCH")

    def test_absent_event_not_misreported_as_learned(self):
        result = audit_confirmed_event(self.root, self.aoi)
        self.assertEqual(result["state"], "NO_JOURNAL")
        self.assertEqual(result["events"], [])

    def test_automatic_label_cannot_be_represented_as_confirmed(self):
        self.write_case()
        path = self.events / "one.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["human_source"] = "production_auto"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fonte humana"):
            audit_confirmed_event(self.root, self.aoi)


if __name__ == "__main__":
    unittest.main()
