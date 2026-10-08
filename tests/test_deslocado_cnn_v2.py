"""DESLOCADO v2: proxy de componente, OK difíceis, isolamento e segurança."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from src.services.deslocado_proxy_v2 import (
    component_mask, simulate_component_shift, paired_photometric,
)
from src.scripts.train_deslocado_cnn_v2 import (
    DeslocadoV2Dataset, train_deslocado_v2,
)
from tests.test_deslocado_cnn import DeslocadoFixtures


class ProxyGeometryTests(unittest.TestCase):
    def test_rejects_uniform_or_invalid_unsegmented_images(self):
        frame = np.full((96,112,3), 42, dtype=np.uint8)
        self.assertIsNone(component_mask(frame))
        self.assertIsNone(simulate_component_shift(frame, 42))
        self.assertIsNone(simulate_component_shift(frame[:12], 42))

    def test_moves_component_without_shifting_entire_rectangular_patch(self):
        image = np.full((110, 140, 3), (30, 40, 45), dtype=np.uint8)
        image[35:77, 47:94] = (10, 180, 220)
        original = image.copy()
        out = simulate_component_shift(image, 31337)
        self.assertIsNotNone(out)
        self.assertGreater(abs(out.shift_xy[0])+abs(out.shift_xy[1]), 1)
        self.assertEqual(out.component_box, (47, 35, 47, 42))
        self.assertEqual(out.normal.shape, image.shape)
        self.assertEqual(out.shifted.shape, image.shape)
        self.assertTrue(np.array_equal(image, original))
        self.assertFalse(np.array_equal(out.normal, out.shifted))
        # Fora da área do componente e do destino os dois exemplos
        # devem permanecer idênticos: nenhum patch amplo é movido.
        self.assertTrue(np.array_equal(
            out.normal[:14], out.shifted[:14]
        ))
        self.assertTrue(np.array_equal(
            out.normal[:, :15], out.shifted[:, :15]
        ))
        self.assertTrue(np.array_equal(
            out.normal[-14:], out.shifted[-14:]
        ))

    def test_recomposition_has_same_original_background_both_classes(self):
        test = np.full((110,130,3), 70, dtype=np.uint8)
        test[31:76,42:86] = (170,210,225)
        out = simulate_component_shift(test, 99)
        self.assertIsNotNone(out)
        mask = np.any(out.normal != out.shifted, axis=2)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), 110*130*.3)
        # Origem e classe sintética não viram "NG real" no dataset.
        self.assertEqual(out.confidence_hint, "HEURISTIC_UNVERIFIED")

    def test_paired_photometric_keeps_geometry_and_is_reproducible(self):
        frame = np.full((72,90,3), 34, dtype=np.uint8)
        frame[25:45,30:70] = 130
        a,b = paired_photometric(frame,frame,42,independent=False)
        self.assertTrue(np.array_equal(a,b))
        c,d = paired_photometric(frame,frame,42,independent=False)
        self.assertTrue(np.array_equal(a,c))
        self.assertTrue(np.array_equal(b,d))


class DeslocadoV2TrainingTests(DeslocadoFixtures):
    def test_mask_proxy_and_multilight_event_are_grouped(self):
        _, folder = self.prepared()
        from src.scripts.train_deslocado_cnn import load_ok_events
        events, data = load_ok_events(folder/"manifest.json")
        ds = DeslocadoV2Dataset(events, data, size=64, proxy_variants=2)
        counts = ds.counts()
        self.assertEqual(counts["REAL_OK"], 4)
        self.assertEqual(counts["RECOMPOSED_OK"], 8)
        self.assertEqual(counts["SYNTHETIC_SHIFT_PROXY"], 8)
        self.assertEqual(len(ds.unresolved), 0)
        triplet = next(i for i,r in enumerate(ds.instances)
                       if r[1] == "SYNTHETIC_SHIFT_PROXY"
                       and len(ds.items[r[0]]["views"]) == 3)
        sample = ds[triplet]
        self.assertEqual(tuple(sample[0].shape), (3, 3, 64, 64))
        self.assertEqual(sample[4].tolist(), [1.,1.,1.])
        self.assertEqual(sample[7], "SYNTHETIC_SHIFT_PROXY")

    def test_one_epoch_candidate_is_never_used_for_production(self):
        report_prepared, folder = self.prepared()
        archived = {
            file: sha256(file.read_bytes()).hexdigest()
            for file in (self.root/"public"/"ok_archive").glob("*.png")
        }
        result, modeldir = train_deslocado_v2(
            folder/"manifest.json", epochs=1, size=64, batch_size=2,
            proxy_variants=1
        )
        self.assertEqual(result["real_ok_events"], 4)
        self.assertEqual(result["real_ok_images"], 6)
        self.assertEqual(result["real_ng_count"], 0)
        self.assertIsNone(result["real_ng_recall"])
        self.assertFalse(result["production_approved"])
        self.assertTrue(result["activation_disabled"])
        self.assertGreaterEqual(result["development"]["real_ok"], 1)
        self.assertEqual(
            result["development"]["real_ok_per_light"]["SIDE"]["tested"],
            result["development"]["real_ok"],
        )
        self.assertTrue(
            (modeldir/"holdout_predictions_deslocado_v2.json").is_file()
        )
        self.assertTrue(
            (modeldir/"training_report_deslocado_v2.json").is_file()
        )
        self.assertTrue(
            (modeldir/"training_summary_deslocado_v2.txt").is_file()
        )
        previews = result["development_proxy_preview_images"]
        self.assertGreaterEqual(len(previews), 1)
        self.assertTrue(all((modeldir/name).is_file() for name in previews))
        self.assertTrue(all(name.startswith("proxy_previews/") for name in previews))
        ckpt = torch.load(
            modeldir/"deslocado_cnn_v2_candidate.pt",
            map_location="cpu", weights_only=True
        )
        self.assertEqual(
            ckpt["schema"], "visionx.deslocado_comparative_cnn.v2"
        )
        self.assertEqual(ckpt["real_ng_used"], 0)
        self.assertFalse(ckpt["production_approved"])
        self.assertFalse(ckpt["allow_automatic_classification"])
        self.assertEqual(
            {file:sha256(file.read_bytes()).hexdigest()
             for file in archived}, archived,
        )
        self.assertFalse(
            (self.root/"reports"/"neural_online"/"live_active.json").exists()
        )

    def test_unsegmentable_ok_is_retained_but_no_fake_ng_is_made(self):
        _, folder = self.prepared()
        from src.scripts.train_deslocado_cnn import load_ok_events
        events, data = load_ok_events(folder/"manifest.json")
        chosen = next(iter(data["samples"].values()))
        ref = folder/chosen["reference_path"]
        test = folder/chosen["test_path"]
        import cv2
        solid = np.full((95,105,3), 80, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(ref),solid))
        self.assertTrue(cv2.imwrite(str(test),solid))
        ds = DeslocadoV2Dataset(events,data,size=64,proxy_variants=1)
        self.assertEqual(ds.counts()["REAL_OK"], 4)
        self.assertLess(ds.counts()["SYNTHETIC_SHIFT_PROXY"], 4)
        self.assertTrue(ds.unresolved)

    def test_v1_checkpoint_and_training_are_not_overwritten(self):
        _, folder = self.prepared()
        r, new_dir = train_deslocado_v2(
            folder/"manifest.json",epochs=1,size=64,batch_size=2,
            proxy_variants=1
        )
        self.assertTrue(new_dir.name.startswith("experiment_v2_"))
        self.assertFalse(
            list(self.root.glob("reports/deslocado_neural/models/experiment_*/deslocado_cnn_candidate.pt"))
        )


if __name__ == "__main__":
    unittest.main()
