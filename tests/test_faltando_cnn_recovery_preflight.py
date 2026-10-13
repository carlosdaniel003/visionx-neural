"""Recuperação CNN: somente 3 eventos corrigidos com pixels auditados."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.faltando_cnn_recovery_preflight import (
    latest_completed_correction, validate_recovery_inputs,
)

ID = "20261013T175115Z_ebf986ef"
LIGHTS = ("SIDE", "TOP", "MID")


class RecoveryPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root / "reports" / "label_corrections" / ID
        self.folder.mkdir(parents=True)
        self.online = self.root / "reports" / "neural_online" / "events"
        self.online.mkdir(parents=True)
        self.dataset_jsons = []
        self.journals = []
        self.folder.joinpath("status.json").write_text(json.dumps({
            "schema": "visionx.label_correction.v1",
            "state": "COMPLETED",
            "correction_id": ID,
            "images_corrected": 9,
            "events_corrected": 3,
            "dataset_records_corrected": 9,
            "cnn_online_journal_corrected": 3,
        }), encoding="utf-8")
        for evt, category in (("piece-1","FALTANDO"),
                              ("piece-2","INVERTIDO"),
                              ("piece-3","FALTANDO")):
            assets = []
            for ix, light in enumerate(LIGHTS):
                label = len(evt) + ix + (7 if category == "INVERTIDO" else 0)
                ref = np.full((50, 66, 3), label * 7, dtype=np.uint8)
                test = ref.copy()
                test[20:35, 20:38] = 0
                source = np.full((60, 90, 3), label * 3, dtype=np.uint8)
                dataset_dir = (
                    self.root / "public" / "dataset" / "anomalia" / category
                )
                dataset_dir.mkdir(parents=True, exist_ok=True)
                stem = f"memory_NG_{evt}_{light}"
                files = {}
                for role, value in (
                    ("source",source), ("reference", ref), ("test",test)
                ):
                    path = dataset_dir / f"{stem}_{role}.png"
                    self.assertTrue(cv2.imwrite(str(path), value))
                    files[role] = path
                metadata = {
                    "label": "NG", "label_correction": {
                        "from": "OK", "to": "NG", "correction_id": ID,
                    },
                    "decision": {"operator_label": "NG"},
                    "aoi_info": {
                        "category": category, "lighting_mode": light,
                    },
                    "multilight": {"event_id": evt},
                    "storage": {
                        role + "_image_file": files[role].name
                        for role in ("source", "reference", "test")
                    },
                }
                manifest = dataset_dir / f"{stem}.json"
                manifest.write_text(json.dumps(metadata), encoding="utf-8")
                self.dataset_jsons.append(manifest)

                light_row = {"lighting_mode": light}
                for role, image in (("reference", ref), ("test", test)):
                    path = self.online / f"{evt}_{light}_{role}.png"
                    self.assertTrue(cv2.imwrite(str(path), image))
                    light_row[role] = path.name
                    light_row[role + "_sha256"] = sha256(path.read_bytes()).hexdigest()
                assets.append(light_row)
            path = self.online / f"online_{evt}.json"
            path.write_text(json.dumps({
                "schema": "visionx.specialist_online_case.v1",
                "event_id": path.stem,
                "source_aoi_event_id": evt,
                "category": category,
                "label": "NG",
                "training_requested": True,
                "label_correction": {
                    "from": "OK", "to": "NG", "correction_id": ID,
                },
                "images": assets,
            }), encoding="utf-8")
            self.journals.append(path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accept_three_true_ng_events_from_complete_correction(self):
        self.assertEqual(latest_completed_correction(self.root), ID)
        result = validate_recovery_inputs(self.root, ID)
        self.assertEqual(len(result["events"]), 3)
        self.assertEqual(result["dataset_pairs"], 9)
        self.assertEqual(result["mode"], "CANDIDATE_ONLY_OFFLINE")
        self.assertFalse(result["training_authorized_for_production"])

    def test_rejects_wrong_dataset_label_without_mutating_data(self):
        path = self.dataset_jsons[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        data["label"] = "OK"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Dataset corrigido"):
            validate_recovery_inputs(self.root, ID)

    def test_rejects_tampered_online_image(self):
        image = self.online / "piece-2_TOP_test.png"
        image.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_recovery_inputs(self.root, ID)

    def test_rejects_missing_mid_or_mismatched_event(self):
        path = self.journals[0]
        row = json.loads(path.read_text(encoding="utf-8"))
        row["images"] = row["images"][:2]
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "3 iluminações"):
            validate_recovery_inputs(self.root, ID)

    def test_rejects_interrupted_correction(self):
        status = self.folder / "status.json"
        row = json.loads(status.read_text(encoding="utf-8"))
        row["state"] = "APPLYING"
        status.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Transação incompleta"):
            validate_recovery_inputs(self.root, ID)


if __name__ == "__main__":
    unittest.main()
