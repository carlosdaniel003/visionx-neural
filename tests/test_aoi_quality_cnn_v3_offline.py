"""CNN V3 — comparação real com gradientes, offline e sem ativação XP."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3, MODEL_SCHEMA_V3
from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_aoi_quality_cnn_v3_offline import (
    PhotometricPairDataset, SCHEMA, _loss_v3,
    _selection_score, train_v3,
)
from src.services.aoi_universal_quality_archive import split_universal_events


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_siamese_pair_features_do_not_depend_on_reference_test_order(self):
        model = AOIQualityCNNV3(dropout=0).eval()
        ref = torch.rand((2, 3, 3, 64, 64))
        test = torch.rand_like(ref)
        focus_ref = torch.rand_like(ref)
        focus_test = torch.rand_like(ref)
        mask = torch.tensor([[1., 1., 0.], [0., 1., 1.]])
        with torch.inference_mode():
            a, lights_a = model(ref, test, focus_ref, focus_test, mask)
            b, lights_b = model(test, ref, focus_test, focus_ref, mask)
        self.assertTrue(torch.allclose(a, b, atol=1.e-6))
        self.assertTrue(torch.allclose(lights_a, lights_b, atol=1.e-6))
        self.assertEqual(tuple(a.shape), (2,))
        self.assertEqual(tuple(lights_a.shape), (2, 3))
        self.assertTrue(torch.allclose(a[0], lights_a[0, :2].max()))
        self.assertTrue(torch.allclose(a[1], lights_a[1, 1:].max()))
        self.assertEqual(MODEL_SCHEMA_V3, "visionx.aoi_pair_quality_cnn.v3.offline")
        self.assertNotEqual(MODEL_SCHEMA_V3, MODEL_SCHEMA_V2)

    def test_validates_partial_missing_and_invalid_masks(self):
        model = AOIQualityCNNV3(dropout=0).eval()
        imgs = [torch.zeros((1, 3, 3, 64, 64)) for _ in range(4)]
        for mask in (
            torch.zeros((1, 3)),
            torch.tensor([[.5, 1., 0.]]),
            torch.zeros((1, 2)),
        ):
            with self.subTest(mask=mask.tolist()):
                with self.assertRaises(ValueError):
                    model(*imgs, mask)
        with self.assertRaises(ValueError):
            model(*imgs[:3], imgs[3][..., :32, :32],
                  torch.tensor([[1., 0., 0.]]))

    def test_identity_regularizer_and_supervised_branch_backpropagate(self):
        model = AOIQualityCNNV3(dropout=0)
        ref = torch.rand((2, 3, 3, 64, 64))
        test = ref.clone()
        test[1, 0, :, 15:42, 15:42] = 0
        mask = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        batch = (ref, test, ref, test, mask, torch.tensor([0., 1.]))
        loss, parts = _loss_v3(model, batch)
        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreater(parts["supervised"], 0.)
        self.assertGreater(parts["identity_reference"], 0.)
        loss.backward()
        self.assertTrue(any(
            param.grad is not None and param.grad.abs().sum().item() > 0
            for param in model.parameters()
        ))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = (
            self.root/"reports/faltando_neural/run_fixture/manifest.json"
        )
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"test": true}', encoding="utf-8")
        self.root.joinpath("reports/neural_online").mkdir(parents=True)
        self.records = self._records()

    def _records(self):
        rng = np.random.default_rng(42)
        cases = []
        categories = ("EMBORCADO", "INVERTIDO", "FALTANDO")
        for i in range(36):
            category = categories[i % 3]
            # Quatorze casos humanos: 9 NG + 5 OK (acervo real).
            # Restante fica misto para validação/teste independente.
            label = (
                ("OK" if i < 5 else "NG") if i < 14
                else ("NG" if (i // 3) % 2 else "OK")
            )
            light = ("SIDE", "TOP", "MID")[i % 3]
            raw = rng.integers(30, 120, (72, 72, 3), dtype=np.uint8)
            reference = raw.copy()
            test = raw.copy()
            # Dois recortes diferentes, mas a verdade física depende
            # de uma ausência central, não do texto/nome da categoria.
            reference[24:46, 24:46] = 190
            test[24:46, 24:46] = 190 if label == "OK" else 45
            src = self.root/f"images/e{i:03d}_reference.png"
            tst = self.root/f"images/e{i:03d}_test.png"
            src.parent.mkdir(exist_ok=True)
            for path, arr in ((src, reference), (tst, test)):
                ok, buf = cv2.imencode(".png", arr)
                self.assertTrue(ok)
                path.write_bytes(buf.tobytes())
            cases.append({
                "id": f"event_{i:03d}", "category_hint": category,
                "label": label,
                "label_provenance": (
                    "HUMAN_V3_IMAGE_SHA256" if i < 14
                    else "ARCHIVE_FOLDER_PROVISIONAL"
                ),
                "split_group": f"AOI_MINUTE:{i:04d}",
                "views": {light: {"reference": src, "test": tst}},
                "sources": [f"public/{label.lower()}_archive/{i:03d}.png"],
            })
        return cases

    def _provenance(self):
        return {
            "manifest_sha256": sha256(self.manifest.read_bytes()).hexdigest(),
            "by_category_and_label": {},
        }

    def test_training_refuses_archive_labels_without_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "provisórios"):
            train_v3(
                self.root, self.manifest, epochs=1, image_size=64,
                compare_verified_pinned_cnn=False,
            )
        self.assertFalse((self.root/"reports/aoi_quality_cnn_v3").exists())

    def test_offline_candidate_trains_and_never_changes_production(self):
        pointer = self.root/"reports/neural_online/live_active.json"
        before_manifest = self.manifest.read_bytes()
        pngs = {
            str(pair[file]): sha256(pair[file].read_bytes()).hexdigest()
            for row in self.records
            for pair in row["views"].values()
            for file in ("reference", "test")
        }
        with patch(
            "src.scripts.train_aoi_quality_cnn_v3_offline.universal_archive_events",
            return_value=(self.records, self._provenance()),
        ):
            report = train_v3(
                self.root, self.manifest,
                epochs=1, image_size=64,
                allow_provisional_archive_labels=True,
                compare_verified_pinned_cnn=False,
            )
        self.assertEqual(report["schema"],
                         "visionx.aoi_quality_cnn_v3_training_report.v1")
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["automatic_xp_authorized"])
        self.assertFalse(report["memory_or_knn_used_for_inference"])
        self.assertFalse(report["safety"]["production_eligible"])
        self.assertFalse(report["aoi_category_used_for_training_selection_evaluation"])
        self.assertEqual(report["operator_image_review_verified"], 14)
        self.assertEqual(report["operator_image_review_ignored"], 1)
        self.assertTrue(report["operator_images_reserved_exclusively_for_training"])
        self.assertFalse(report["partition"]["aoi_category_used_for_split"])
        self.assertEqual(report["partition"]["pinned_training_source_count"], 14)
        self.assertNotIn("by_aoi_category", report["test"])
        self.assertNotIn("by_category_and_label", report["provenance"])
        self.assertEqual(report["coverage"]["train"]["human_verified_v3"], 14)
        self.assertEqual(report["coverage"]["validation"]["human_verified_v3"], 0)
        self.assertEqual(report["coverage"]["test"]["human_verified_v3"], 0)
        self.assertEqual(report["operator_verified_training_replay"]["metrics"]["total"], 14)
        self.assertTrue(
            report["operator_verified_training_replay"]["in_sample_not_independent_test"]
        )
        self.assertIsNone(report["previous_v3_operator_verified_training_replay"])
        self.assertTrue(report["training"]["test_not_used_for_selection"])
        self.assertEqual(len(report["epoch_history"]), 2)
        self.assertTrue(report["human_review_evaluated_every_epoch"])
        self.assertEqual(len(report["epoch_diagnostic_checkpoints"]), 1)
        self.assertEqual(report["epoch_diagnostic_schema"],
                         "visionx.aoi_quality_cnn_v3_epoch_diagnostic_only.v1")
        for item in report["epoch_history"]:
            replay = item["operator_training_replay"]
            self.assertEqual(replay["metrics"]["total"], 14)
            self.assertEqual(replay["metrics"]["NG_total"], 9)
            self.assertEqual(replay["metrics"]["OK_total"], 5)
            self.assertTrue(replay["in_sample_not_independent_test"])
        checkpoint = self.root / report["epoch_diagnostic_checkpoints"][0]["path"]
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            sha256(checkpoint.read_bytes()).hexdigest(),
            report["epoch_diagnostic_checkpoints"][0]["sha256"],
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertFalse(saved["automatic_xp_authorized"])
        self.assertFalse(saved["production_approved"])
        self.assertFalse(saved["test_evaluated_for_this_epoch"])
        self.assertNotIn("state_dict", saved)
        self.assertNotIn("test", saved)
        self.assertEqual(
            report["test"]["total"],
            report["test"]["NG_total"] + report["test"]["OK_total"],
        )
        self.assertFalse(report["partition"]["physical_event_association_verified"])
        self.assertFalse(report["safety"]["blind_holdout"])
        self.assertFalse(report["coverage"]["all_data"]["fully_operator_verified"])
        self.assertEqual(before_manifest, self.manifest.read_bytes())
        self.assertFalse(pointer.exists())
        self.assertEqual(pngs, {
            str(pair[file]): sha256(pair[file].read_bytes()).hexdigest()
            for row in self.records
            for pair in row["views"].values()
            for file in ("reference", "test")
        })
        path = self.root/report["candidate_path"]
        self.assertTrue(path.is_file())
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(weights["schema"], SCHEMA)
        self.assertEqual(weights["architecture_schema"], MODEL_SCHEMA_V3)
        self.assertNotEqual(weights["schema"], MODEL_SCHEMA_V2)
        self.assertFalse(weights["production_approved"])
        self.assertFalse(weights["automatic_xp_authorized"])
        self.assertTrue(weights["candidate_only"])
        AOIQualityCNNV3().load_state_dict(weights["state_dict"], strict=True)
        with self.assertRaises(RuntimeError):
            FaltandoCNNV2().load_state_dict(weights["state_dict"], strict=True)

    def test_augmented_views_preserve_pair_mask_and_change_brightness(self):
        ds = PhotometricPairDataset(
            self.records[:3], 64, .70, augment=True,
        )
        r, t, rf, tf, mask, y = ds[0]
        self.assertEqual(tuple(r.shape), (3, 3, 64, 64))
        self.assertEqual(int(mask.sum().item()), 1)
        self.assertEqual(float(y.item()), 0.)
        self.assertTrue(bool(torch.isfinite(r).all().item()))
        self.assertTrue(bool(torch.isfinite(t).all().item()))
        self.assertTrue(bool(torch.isfinite(rf).all().item()))
        self.assertTrue(bool(torch.isfinite(tf).all().item()))
        for light in range(3):
            if mask[light] == 0:
                self.assertEqual(float(r[light].abs().sum().item()), 0.)

    def test_selection_rejects_single_class_validation(self):
        with self.assertRaisesRegex(ValueError, "rótulos OK e NG"):
            _selection_score({"NG_total": 0, "OK_total": 12})

    def test_quality_only_split_ignores_any_aoi_category_and_pins_human_groups(self):
        rows = [
            {**r, "category_hint": "FALSAMENTE_" + str(i)}
            for i, r in enumerate(self.records)
        ]
        # Um outro arquivo, mesmo minuto de captura, deve acompanhar
        # a amostra humana para evitar vazamento para validação/teste.
        rows[14] = {**rows[14], "split_group": rows[0]["split_group"]}
        pinned = {r["sources"][0] for r in rows[:14]}
        args = {"balance_by_aoi_category": False,
                "pinned_training_sources": pinned}
        a, b, c, partition = split_universal_events(rows, **args)
        self.assertFalse(partition["aoi_category_used_for_split"])
        self.assertEqual(partition["pinned_training_source_count"], 14)
        self.assertTrue(pinned <= {r["sources"][0] for r in a})
        self.assertIn(rows[14]["id"], {r["id"] for r in a})
        self.assertTrue(all(
            r["sources"][0] not in pinned for r in b+c
        ))
        # Renomear arbitrariamente a categoria AOI não muda os folds.
        different = [{**r, "category_hint": None} for r in rows]
        aa, bb, cc, _ = split_universal_events(different, **args)
        self.assertEqual(
            [[r["id"] for r in fold] for fold in (a, b, c)],
            [[r["id"] for r in fold] for fold in (aa, bb, cc)],
        )
        with self.assertRaisesRegex(ValueError, "ausente"):
            split_universal_events(rows, pinned_training_sources={"missing"},
                                   balance_by_aoi_category=False)

    def test_training_can_warm_start_from_hash_verified_v3_candidate(self):
        import json
        folder = self.root/"reports/aoi_quality_cnn_v3/candidates/old_v3"
        folder.mkdir(parents=True)
        candidate = folder/"aoi_quality_cnn_v3_candidate.pt"
        model = AOIQualityCNNV3()
        manifest_sha = sha256(self.manifest.read_bytes()).hexdigest()
        torch.save({
            "schema": SCHEMA,
            "architecture_schema": MODEL_SCHEMA_V3,
            "source_manifest_sha256": manifest_sha,
            "image_size": 64, "focus_fraction": .70,
            "lights": ("SIDE", "TOP", "MID"),
            "candidate_only": True, "production_approved": False,
            "automatic_xp_authorized": False,
            "state_dict": model.state_dict(),
        }, candidate)
        report_file = folder/"training_report.json"
        report_file.write_text(json.dumps({
            "schema": "visionx.aoi_quality_cnn_v3_training_report.v1",
            "candidate_only": True,
            "production_approved": False,
            "automatic_xp_authorized": False,
            "provenance": {"manifest_sha256": manifest_sha},
            "candidate_path": candidate.relative_to(self.root).as_posix(),
            "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        original_sha = sha256(candidate.read_bytes()).hexdigest()
        with patch(
            "src.scripts.train_aoi_quality_cnn_v3_offline.universal_archive_events",
            return_value=(self.records, self._provenance()),
        ):
            result = train_v3(
                self.root, self.manifest,
                epochs=1, image_size=64,
                initialize_from_v3_report=report_file,
                allow_provisional_archive_labels=True,
                compare_verified_pinned_cnn=False,
            )
        self.assertEqual(
            result["initial_v3_candidate"]["candidate_sha256"], original_sha,
        )
        self.assertIsNotNone(result["verified_previous_v3_baseline"])
        self.assertEqual(result["operator_verified_training_replay"]["metrics"]["total"], 14)
        self.assertEqual(
            result["previous_v3_operator_verified_training_replay"]["metrics"]["total"],
            14,
        )
        self.assertTrue(
            result["training"][
                "selected_epoch_cannot_worsen_either_validation_error_against_start"
            ]
        )
        self.assertEqual(len(result["epoch_history"]), 2)
        self.assertEqual(len(result["epoch_diagnostic_checkpoints"]), 1)
        self.assertTrue(result["human_learning_diagnosis"][
            "test_evaluated_only_for_safe_candidate"])
        self.assertEqual(
            result["epoch_history"][0]["operator_training_replay"]["metrics"]["total"],
            14,
        )
        self.assertFalse(result["production_approved"])
        self.assertEqual(sha256(candidate.read_bytes()).hexdigest(), original_sha)

    def test_group_split_keeps_duplicates_and_minutes_together(self):
        t, v, h, partition = split_universal_events(self.records)
        self.assertEqual(len(t)+len(v)+len(h), len(self.records))
        self.assertFalse(partition["physical_event_association_verified"])
        self.assertEqual(
            len({row["id"] for fold in (t, v, h) for row in fold}),
            len(self.records),
        )


if __name__ == "__main__":
    unittest.main()
