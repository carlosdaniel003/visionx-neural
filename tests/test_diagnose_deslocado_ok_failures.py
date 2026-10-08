"""Gerar painéis comparativos sem retreinar nem tocar na produção."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

from src.scripts.diagnose_deslocado_ok_failures import diagnose_failures
from tests.test_deslocado_cnn import DeslocadoFixtures


class DiagnoseDeslocadoTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        _, self.run = self.prepared()
        self.manifest = self.run/"manifest.json"
        self.failures = [
            {
                "source_path": "public/ok_archive/2026-10-02_1349_DESLOCADO.png",
                "expected": "OK", "predicted": "NG",
                "lighting_mode": "SIDE", "ng_proxy_score_uncalibrated": .50294089
            },
            {
                "source_path": "public/ok_archive/2026-10-02_1350_DESLOCADO.png",
                "expected": "OK", "predicted": "NG",
                "lighting_mode": "SIDE", "ng_proxy_score_uncalibrated": .50141501
            },
        ]
        self.report = self.root/"reports"/"deslocado_neural"/"test_replay.json"
        self.record()

    def record(self):
        self.report.write_text(json.dumps({
            "schema": "visionx.deslocado_ok_archive_replay.v1",
            "model": "visionx.deslocado_comparative_cnn.v2",
            "evaluation_manifest_sha256": sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "failed_images": self.failures,
        }), encoding="utf-8")

    def call(self):
        return diagnose_failures(self.root, replay=self.report,
                                 manifest=self.manifest)

    def test_two_failures_create_two_side_by_side_pngs(self):
        originals = {
            p: sha256(p.read_bytes()).hexdigest()
            for p in (self.root/"public"/"ok_archive").glob("*.png")
        }
        report, directory = self.call()
        self.assertEqual(report["failed_cases"], 2)
        self.assertEqual(report["diagnostic_panels_created"], 2)
        self.assertEqual(len(list(directory.glob("*.png"))), 2)
        self.assertTrue((directory/"diagnostic_summary.json").exists())
        for case in report["cases"]:
            from cv2 import imread
            picture = imread(str(directory/case["diagnostic_png"]))
            self.assertEqual(picture.shape[:2], (320, 1260))
        self.assertEqual(originals, {
            p: sha256(p.read_bytes()).hexdigest() for p in originals
        })
        self.assertFalse(report["model_retrained"])
        self.assertFalse(report["production_modified"])

    def test_corrupt_manifest_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data["summary"]["total"] = 999
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "não é o do replay"):
            self.call()

    def test_unknown_path_is_rejected(self):
        self.failures[0]["source_path"] = "../private.jpg"
        self.record()
        with self.assertRaisesRegex(ValueError, "não verificável"):
            self.call()

    def test_zero_failures_writes_empty_diagnostic(self):
        self.failures.clear()
        self.record()
        report, folder = self.call()
        self.assertEqual(report["diagnostic_panels_created"], 0)
        self.assertEqual(list(folder.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
