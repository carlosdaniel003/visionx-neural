"""Operator-verified validation labels and offline second repair safety gates."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.odin_operator_validation_labels import (
    EVIDENCE, verified_validation_labels,
)
from src.scripts.train_aoi_universal_24_human_repair import (
    _acceptable, _verified_metrics, _avoid_visual_replay_in_validation,
)


def _png(value):
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    good, blob = cv2.imencode(".png", image)
    assert good
    return blob.tobytes()


class DummyStore:
    def __init__(self, root):
        self.root = root
        self.manifest_path = root / "reports/faltando_neural/test/manifest.json"
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text('{"offline":true}', encoding="utf-8")
        self.by_path, self.files, self.expected_labels = {}, {}, {}

    def _validate_source(self, source):
        pass


class NewNineConfirmedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = DummyStore(self.root)
        self.items = []
        self.root.joinpath("docs").mkdir()
        for number in range(1, 10):
            label = "NG" if number in (5, 6, 7) else "OK"
            source = f"public/{label.lower()}_archive/2026-10-26_09{number:02d}_FALTANDO_SIDE.png"
            original = self.root / source
            reference = self.root / "reports/faltando_neural/test/pairs" / f"{number}_r.png"
            test = reference.with_name(f"{number}_t.png")
            for path, content in (
                (original, _png(number + 20)),
                (reference, _png(number + 40)),
                (test, _png(number + 60)),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            digest = lambda p: sha256(p.read_bytes()).hexdigest()
            entry = {
                "case_number": number,
                "source_path": source,
                "quality_label": label,
                "source_sha256": digest(original),
                "reference_sha256": digest(reference),
                "test_sha256": digest(test),
                "operator_reviewed_pair": True,
            }
            self.items.append(entry)
            self.store.by_path[source] = {
                "source_sha256": entry["source_sha256"],
                "status": "EXTRACTED_PENDING_REVIEW",
            }
            self.store.files[source] = original, reference, test
            self.store.expected_labels[source] = label
        self.evidence_file = self.root / EVIDENCE
        self.doc = {
            "schema": "visionx.operator_confirmed_validation_quality.v1",
            "reviewed_on": "2026-10-26",
            "approved_for_production": False,
            "physical_event_groups_verified": False,
            "source_manifest_sha256":
                sha256(self.store.manifest_path.read_bytes()).hexdigest(),
            "source_checkpoint_sha256": "a" * 64,
            "labels": self.items,
        }
        self.write_evidence()

    def write_evidence(self):
        self.evidence_file.write_text(
            json.dumps(self.doc), encoding="utf-8",
        )

    def test_9_entries_are_human_quality_not_archive_bulk_relabel(self):
        verified = verified_validation_labels(self.store)
        self.assertEqual(len(verified), 9)
        self.assertEqual(
            Counter(x["quality_label"] for x in verified.values()),
            Counter({"OK": 6, "NG": 3}),
        )

    def test_tampered_original_fails(self):
        case = self.items[0]["source_path"]
        self.store.files[case][0].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Integridade SHA"):
            verified_validation_labels(self.store)

    def test_tampered_gabarito_fails(self):
        case = self.items[6]["source_path"]
        self.store.files[case][1].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Integridade SHA"):
            verified_validation_labels(self.store)

    def test_label_conflicting_with_archive_fails(self):
        self.doc["labels"][3]["quality_label"] = "NG"
        self.write_evidence()
        with self.assertRaisesRegex(ValueError, "Caminho do PNG"):
            verified_validation_labels(self.store)

    def test_wrong_manifest_fails(self):
        self.store.manifest_path.write_text('{"new":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifesto original"):
            verified_validation_labels(self.store)


class SecondRepairGatesTests(unittest.TestCase):
    def test_24_human_metrics_correct(self):
        truth = [
            {"id": f"h{i}", "label": "NG" if i < 15 else "OK",
             "sources": [f"image{i}"]}
            for i in range(24)
        ]
        prediction = {"cases": [
            {"event_id_or_group_hint": x["id"], "label": x["label"],
             "predicted": x["label"], "ng_score": .99}
            for x in truth
        ]}
        outcome = _verified_metrics(prediction, truth)
        self.assertEqual(outcome["NG_false_OK"], 0)
        self.assertEqual(outcome["OK_false_NG"], 0)
        self.assertEqual(outcome["total_correct"], 24)

    def test_validation_blocks_any_worse_ng_or_ok(self):
        baseline = {"NG_false_OK": 3, "OK_false_NG": 4,
                    "balanced_accuracy": .9}
        supervised = {"NG_false_OK": 0, "OK_false_NG": 0}
        self.assertIsNone(_acceptable(
            supervised, {**baseline, "NG_false_OK": 4}, baseline,
        ))
        self.assertIsNone(_acceptable(
            supervised, {**baseline, "OK_false_NG": 5}, baseline,
        ))
        self.assertIsNotNone(_acceptable(supervised, baseline, baseline))

    def test_visual_neighbors_are_removed_before_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i, value in enumerate((50, 50, 180)):
                r, t = root / f"ref{i}.png", root / f"test{i}.png"
                r.write_bytes(_png(value))
                t.write_bytes(_png(value))
                rows.append({
                    "id": f"case{i}", "label": "NG" if i < 2 else "OK",
                    "split_group": f"AOI_MINUTE:2026-10-26_09{i:02d}",
                    "views": {"SIDE": {"reference": r, "test": t}},
                })
            kept, removed = _avoid_visual_replay_in_validation(
                [rows[1], rows[2]], [rows[0]],
            )
            self.assertEqual(removed, 1)
            self.assertEqual([x["id"] for x in kept], ["case2"])
            # Actual training veto requires both labels in remaining holdout;
            # this helper only removes visually related candidates.


if __name__ == "__main__":
    unittest.main()
