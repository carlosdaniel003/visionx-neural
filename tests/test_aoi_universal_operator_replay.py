"""Offline replay keeps source PNGs immutable and checks all 15 confirmations."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.scripts.audit_aoi_universal_operator_confirmed import (
    _build_records, _candidate, _lighting,
)


class OperatorReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "reports/faltando_neural/run_test"
        self.run.mkdir(parents=True)
        self.samples, self.human = [], {}
        self.pngs = {}
        for i in range(15):
            light = ("SIDE", "TOP", "MID")[i % 3]
            name = f"2026-10-26_1{i:03d}_FALTANDO_{light}.png"
            label = "OK" if i in (6, 8, 9) else "NG"
            folder = "ng_archive" if i not in (8, 9) else "ok_archive"
            source = f"public/{folder}/{name}"
            png = self.root / source
            png.parent.mkdir(parents=True, exist_ok=True)
            frame = np.full((32, 48, 3), i + 25, dtype=np.uint8)
            encoded = cv2.imencode(".png", frame)[1].tobytes()
            png.write_bytes(encoded)
            digest = sha256(encoded).hexdigest()
            self.pngs[source] = encoded
            self.human[source] = {
                "source_sha256": digest, "quality_label": label,
                "physical_type": "OPERATOR_NG" if label == "NG" else "OK",
                "pair_inspected_by_operator": i < 6,
            }
            if i < 6:
                derived = self.run / "pairs" / digest
                derived.mkdir(parents=True)
                (derived/"reference.png").write_bytes(encoded)
                (derived/"test.png").write_bytes(encoded)
                self.samples.append({
                    "source_path": source,
                    "source_sha256": digest,
                    "reference_path": f"pairs/{digest}/reference.png",
                    "test_path": f"pairs/{digest}/test.png",
                    "lighting_mode": light,
                    "status": "EXTRACTED_PENDING_REVIEW",
                })
        self.manifest = {"samples": self.samples}
        self.destination = self.root / "reports/aoi_universal_quality/operator_replay/run_test"

    def test_collects_exactly_six_prepared_and_nine_extracted(self):
        calls = []
        def fake_extractor(frame):
            calls.append(frame)
            return frame.copy(), frame.copy(), {"safe": True}
        rows, details = _build_records(
            self.root, self.run/"manifest.json", self.manifest,
            self.human, self.destination, extractor=fake_extractor,
        )
        self.assertEqual(details, {"pairs_prepared": 6, "pairs_reextracted": 9})
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(calls), 9)
        self.assertEqual(sum(r["label"] == "OK" for r in rows), 3)
        self.assertEqual(sum(r["label"] == "NG" for r in rows), 12)
        self.assertEqual(
            sum(r["pair_provenance"].startswith("REEXTRACTED") for r in rows), 9,
        )
        self.assertEqual(
            self.pngs,
            {name: (self.root / name).read_bytes() for name in self.pngs},
        )

    def test_original_source_changed_fails_closed(self):
        source = next(iter(self.pngs))
        (self.root / source).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Origem mudou"):
            _build_records(
                self.root, self.run/"manifest.json", self.manifest,
                self.human, self.destination, extractor=lambda x: (x, x, {}),
            )

    def test_missing_prepared_pair_rejected_not_silently_reextracted(self):
        first = self.samples[0]
        (self.run / first["reference_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Par preparado ausente"):
            _build_records(
                self.root, self.run/"manifest.json", self.manifest,
                self.human, self.destination, extractor=lambda x: (x, x, {}),
            )

    def test_candidate_sha_must_match_report(self):
        experiment = self.root/"reports/aoi_universal_quality/experiments/demo"
        experiment.mkdir(parents=True)
        checkpoint = experiment/"aoi_general_quality_offline_candidate.pt"
        checkpoint.write_bytes(b"fake candidate")
        report = experiment/"training_report.json"
        report.write_text(json.dumps({
            "schema": "visionx.aoi_general_quality_experiment_report.v1",
            "candidate_only": True, "promoted": False,
            "automatic_ok_authorized": False,
            "candidate_sha256": "0"*64,
            "source_candidate_relative_path": checkpoint.relative_to(self.root).as_posix(),
        }))
        with self.assertRaisesRegex(ValueError, "Hash do candidato divergente"):
            _candidate(self.root, report)

    def test_light_from_captured_filename(self):
        self.assertEqual(_lighting("public/ng_archive/foo_TOP_2.png"), "TOP")
        self.assertEqual(_lighting("public/ng_archive/foo_MID.png"), "MID")
        self.assertEqual(_lighting("public/ng_archive/legacy_FALTANDO.png"), "SIDE")


if __name__ == "__main__":
    unittest.main()
