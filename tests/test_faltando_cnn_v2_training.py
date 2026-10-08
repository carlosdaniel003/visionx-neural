"""CNN FALTANDO v2: gradientes, pares multiescala e relatórios sem produção."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn_v2 import (
    DualScaleEventDataset, _focus_crop, _letterbox_rgb,
    load_events, split_events, train_v2,
)


class CNNV2Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/"reports"/"faltando_neural"/"run_20261008T151051"
        self.run.mkdir(parents=True)
        for label in ("ok_archive", "ng_archive"):
            (self.root/"public"/label).mkdir(parents=True)
        samples, triplet = [], {}
        self.sources = []
        for k in range(6):
            samples.append(self._make(k, "OK", "SIDE", 10+k))
            samples.append(self._make(20+k, "NG", "SIDE", 80+k))
        for k, light in enumerate(("SIDE", "TOP", "MID")):
            item = self._make(77, "OK", light, 41+k)
            samples.append(item)
            triplet[light] = item["source_path"]
        self.manifest_path = self.run / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "schema": "visionx.faltando_neural_preparation.v1",
            "root": str(self.root), "samples": samples,
            "name_only_triplet_candidates": [{
                "id": "2026-10-08_0735_FALTANDO",
                "label": "OK", "paths": triplet,
            }],
        }), encoding="utf-8")

    def _write(self, path: Path, value: int, missing: bool) -> None:
        # Variar cor, fundo e geometria entre os casos sem alterar classe.
        rng = np.random.default_rng(value)
        image = rng.integers(0, 65, (110, 135, 3), dtype=np.uint8)
        image[20:90, 22:115] = 20 + value % 50
        if not missing:
            image[35:75, 42:95] = (90+value%15, 180, 190)
        path.parent.mkdir(parents=True, exist_ok=True)
        valid, encoded = cv2.imencode(".png", image)
        self.assertTrue(valid)
        path.write_bytes(encoded.tobytes())

    def _make(self, number, label, mode, value):
        source = (self.root/"public"/(
            "ok_archive" if label == "OK" else "ng_archive"
        )/f"case_{number}_{mode}_FALTANDO.png")
        self._write(source, value, label == "NG")
        self.sources.append((source, sha256(source.read_bytes()).hexdigest()))
        digest = sha256(source.read_bytes()).hexdigest()
        folder = Path("pairs")/(label.lower()+"_"+digest)
        reference = self.run/folder/"reference.png"
        test = self.run/folder/"test.png"
        self._write(reference, value, False)
        self._write(test, value, label == "NG")
        return {
            "source_path": source.relative_to(self.root).as_posix(),
            "source_sha256": digest,
            "status": "EXTRACTED_PENDING_REVIEW",
            "expected_label_from_archive": label,
            "lighting_mode": mode,
            "reference_path": (folder/"reference.png").as_posix(),
            "test_path": (folder/"test.png").as_posix(),
            "ocr_observed": {
                "board": "PCB TESTE", "parts": "R"+str(number),
                "value": "0 <= 2 <= 10 FALTANDO",
            },
        }


class FaltandoCNNV2Tests(CNNV2Fixture):
    def test_forward_explicit_light_mask_and_pair_features(self):
        model = FaltandoCNNV2()
        ref = torch.rand(2, 3, 3, 64, 64)
        tst = torch.rand_like(ref)
        mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
        output, per_light = model(ref, tst, ref, tst, mask)
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(tuple(per_light.shape), (2, 3))
        self.assertTrue(torch.allclose(output[0], per_light[0, 0]))
        self.assertTrue(torch.allclose(output[1], per_light[1].max()))
        with self.assertRaises(ValueError):
            model(ref, tst, ref, tst, torch.zeros_like(mask))

    def test_focus_keeps_central_region_without_changing_full_view(self):
        image = np.full((80, 120, 3), 40, dtype=np.uint8)
        image[30:50, 46:73] = 240
        focus = _focus_crop(image, .70)
        self.assertLess(focus.shape[0], image.shape[0])
        self.assertLess(focus.shape[1], image.shape[1])
        self.assertEqual(float(focus.max()), 240.)
        full = _letterbox_rgb(image, 64)
        patch = _letterbox_rgb(focus, 64)
        self.assertEqual(tuple(full.shape), (3, 64, 64))
        self.assertEqual(tuple(patch.shape), (3, 64, 64))
        self.assertFalse(torch.equal(full, patch))

    def test_split_and_multilight_dataset_keeps_event_together(self):
        events, data = load_events(self.manifest_path)
        self.assertEqual(len(events), 13)
        self.assertEqual(sum(len(x["observations"]) for x in events), 15)
        train_ids, valid_ids, report = split_events(events, data, seed=42)
        self.assertTrue(report["train"].get("NG", 0))
        self.assertTrue(report["validation"].get("NG", 0))
        keys_train = set(events[i]["split_key"] for i in train_ids)
        keys_val = set(events[i]["split_key"] for i in valid_ids)
        self.assertFalse(keys_train & keys_val)
        dataset = DualScaleEventDataset(events, data, size=64)
        tri = next(i for i, e in enumerate(events) if len(e["observations"]) == 3)
        case = dataset[tri]
        self.assertEqual(tuple(case[0].shape), (3, 3, 64, 64))
        self.assertEqual(case[4].tolist(), [1., 1., 1.])
        mono = next(i for i, e in enumerate(events) if len(e["observations"]) == 1)
        self.assertEqual(dataset[mono][4].tolist(), [1., 0., 0.])

    def test_one_epoch_saves_v2_checkpoint_and_case_predictions(self):
        result, output = train_v2(
            self.manifest_path, epochs=1, batch_size=4,
            size=64, patience=2,
        )
        self.assertEqual(result["model_schema"], MODEL_SCHEMA_V2)
        self.assertFalse(result["production_approved"])
        self.assertEqual(result["events"], 13)
        self.assertEqual(result["training_parameters"]["completed_epochs"], 1)
        self.assertIn("development_event_ids", result)
        self.assertEqual(
            len(result["per_case_dev_predictions"]),
            sum(result["split"]["validation"].values()),
        )
        for item in result["per_case_dev_predictions"]:
            self.assertIn("ng_score", item)
            self.assertIn("per_light_ng_scores", item)
            self.assertIn("error", item)
            self.assertTrue(0. <= item["ng_score"] <= 1.)
        checkpoint = torch.load(
            output/"faltando_cnn_v2_candidate.pt",
            weights_only=True, map_location="cpu"
        )
        self.assertEqual(checkpoint["schema"], MODEL_SCHEMA_V2)
        self.assertFalse(checkpoint["production_approved"])
        for name in ("training_report_v2.json", "training_summary_v2.txt",
                     "holdout_predictions_v2.json"):
            self.assertTrue((output/name).is_file())
        self.assertTrue(all(sha256(p.read_bytes()).hexdigest() == digest
                            for p, digest in self.sources))
        self.assertEqual(list((self.root/"public").rglob("*.pt")), [])

    def test_changed_source_refuses_training(self):
        source, _ = self.sources[0]
        source.write_bytes(source.read_bytes()+b"modified")
        with self.assertRaisesRegex(ValueError, "Origem mudou"):
            train_v2(self.manifest_path, epochs=1, size=64)
        self.assertFalse((self.run.parent/"models").exists())


if __name__ == "__main__":
    unittest.main()
