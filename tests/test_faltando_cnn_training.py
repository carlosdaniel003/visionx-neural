"""Contrato de treino experimental CNN FALTANDO, sem fotos reais do operador."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.core.neural.faltando_cnn import FaltandoCNN
from src.scripts.train_faltando_cnn import (
    load_events, split_events, train,
)


class FaltandoCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "reports" / "faltando_neural" / "run_20261008T151051"
        self.run.mkdir(parents=True)
        (self.root / "public" / "ok_archive").mkdir(parents=True)
        (self.root / "public" / "ng_archive").mkdir(parents=True)
        self.samples = []
        self.triplets = []
        self.snapshots = {}
        for i in range(6):
            self._case(i, "OK", "SIDE", i * 4 + 2)
            self._case(20+i, "NG", "SIDE", i*5+3)
        for light, i in zip(("SIDE", "TOP", "MID"), (0, 1, 2)):
            self._case(80, "OK", light, 44+i, group=True)
        self.manifest = self.run / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema": "visionx.faltando_neural_preparation.v1",
            "root": str(self.root), "samples": self.samples,
            "name_only_triplet_candidates": [{
                "id": "2026-10-08_0735_FALTANDO",
                "label": "OK",
                "paths": {mode: self.triplets[i] for i, mode in
                          enumerate(("SIDE", "TOP", "MID"))},
            }],
        }), encoding="utf-8")

    def _save_img(self, path, rng, missing=False):
        image = rng.integers(0, 110, (85, 96, 3), dtype=np.uint8)
        image[25:63, 21:70] = (35, 35, 35)
        if not missing:
            image[29:58, 27:64] = (110, 150, 188)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())

    def _case(self, num, label, light, salt, group=False):
        name = f"case_{num}_{light}_FALTANDO.png"
        source = self.root / "public" / (
            "ok_archive" if label == "OK" else "ng_archive"
        ) / name
        rng = np.random.default_rng(salt)
        self._save_img(source, rng, missing=label == "NG")
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        self.snapshots[source] = original_hash
        base = Path("pairs") / (label.lower() + "_" + original_hash)
        ref = self.run / base / "reference.png"
        tst = self.run / base / "test.png"
        rng = np.random.default_rng(salt)
        self._save_img(ref, rng, missing=False)
        rng = np.random.default_rng(salt)
        self._save_img(tst, rng, missing=label == "NG")
        relative = source.relative_to(self.root).as_posix()
        self.samples.append({
            "source_path": relative, "source_sha256": original_hash,
            "expected_label_from_archive": label,
            "status": "EXTRACTED_PENDING_REVIEW",
            "lighting_mode": light,
            "reference_path": (base / "reference.png").as_posix(),
            "test_path": (base / "test.png").as_posix(),
            "ocr_observed": {
                "board": "TEST-1", "parts": "R" + str(num),
                "value": "0 <= 10 <= 30 FALTANDO",
            }
        })
        if group:
            self.triplets.append(relative)

    def test_forward_one_or_three_lights_and_missing_mask(self):
        model = FaltandoCNN()
        reference = torch.rand(2, 3, 3, 64, 64)
        test = torch.rand(2, 3, 3, 64, 64)
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.float32)
        logits, each = model(reference, test, mask)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(each.shape), (2, 3))
        self.assertTrue(torch.allclose(logits[0], each[0, 0]))
        self.assertTrue(torch.allclose(logits[1], each[1].max()))
        with self.assertRaises(ValueError):
            model(reference, test, torch.zeros_like(mask))

    def test_training_groups_and_holdout_never_share_same_component(self):
        events, data = load_events(self.manifest)
        self.assertEqual(len(events), 13)
        self.assertEqual(sum(len(x["observations"]) for x in events), 15)
        self.assertEqual(sum(len(x["observations"]) == 3 for x in events), 1)
        train_ids, val_ids, summary = split_events(events, data, seed=42)
        self.assertGreaterEqual(summary["validation"].get("NG", 0), 1)
        self.assertGreaterEqual(summary["validation"].get("OK", 0), 1)
        self.assertFalse(
            set(events[i]["split_key"] for i in train_ids)
            & set(events[i]["split_key"] for i in val_ids)
        )

    def test_one_epoch_produces_candidate_and_no_production_changes(self):
        report, target = train(
            self.manifest, epochs=1, batch_size=3, size=64, seed=42
        )
        self.assertEqual(report["counts"]["frames"], 15)
        self.assertEqual(report["counts"]["events"], 13)
        self.assertTrue(report["experimental"])
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["knn_used"])
        self.assertTrue((target / "faltando_cnn_candidate.pt").exists())
        self.assertTrue((target / "training_report.json").exists())
        checkpoint = torch.load(
            target / "faltando_cnn_candidate.pt", map_location="cpu",
            weights_only=True
        )
        self.assertFalse(checkpoint["production_approved"])
        self.assertEqual(checkpoint["image_size"], 64)
        self.assertEqual(
            {path: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in self.snapshots}, self.snapshots,
        )
        self.assertFalse(list((self.root / "public").rglob("*.pt")))

    def test_changed_source_aborts_before_model_training(self):
        original = next(iter(self.snapshots))
        original.write_bytes(original.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Origem mudou"):
            train(self.manifest, epochs=1, size=64)
        models = self.root / "reports" / "faltando_neural" / "models"
        self.assertFalse(models.exists())

    def test_missing_holdout_class_fails_without_model(self):
        for sample in self.samples:
            sample["expected_label_from_archive"] = "OK"
        self.manifest.write_text(json.dumps({
            "schema": "visionx.faltando_neural_preparation.v1",
            "root": str(self.root),
            "samples": self.samples,
            "name_only_triplet_candidates": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "grupos NG/OK"):
            train(self.manifest, epochs=1, size=64)


if __name__ == "__main__":
    unittest.main()
