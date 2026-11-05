"""Fim a fim: 25 NG físicos em 24 pares únicos, treino V3 isolado.

O checkpoint de origem é simulado, mas o modelo, o backpropagation,
a separação temporal/visual e os relatórios são reais.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3
from src.scripts.train_aoi_quality_cnn_v3_offline import train_v3


class V3NGReviewTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "reports/faltando_neural/run_test/manifest.json"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"stage":"synthetic"}', encoding="utf-8")
        (self.root / "docs").mkdir()
        review = self.root / "docs/aoi_v3_operator_ng_escape_review_20261105.json"
        review.write_text('{"schema":"fixture"}', encoding="utf-8")
        self.review_sha = sha256(review.read_bytes()).hexdigest()

    def _records(self):
        rng = np.random.default_rng(851)
        records = []
        escapes = set()
        for i in range(60):
            if i < 5:
                label, provenance = "OK", "HUMAN_V3_IMAGE_SHA256"
            elif i < 14:
                label, provenance = "NG", "HUMAN_V3_IMAGE_SHA256"
            elif i <= 36:
                label, provenance = "NG", "HUMAN_V3_NG_ESCAPE_SHA256"
            else:
                label, provenance = (
                    ("OK" if i % 2 else "NG"), "ARCHIVE_FOLDER_PROVISIONAL",
                )
            # Random backgrounds break artificial visual-near-duplicate splits.
            src = rng.integers(30, 220, (72, 72, 3), dtype=np.uint8)
            ref, test = src.copy(), src.copy()
            ref[20:38, 20:38] = 180
            test[20:38, 20:38] = 180 if label == "OK" else 35
            ref_path = self.root / f"staging/{i:03d}_ref.png"
            test_path = self.root / f"staging/{i:03d}_test.png"
            ref_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(ref_path), ref)
            cv2.imwrite(str(test_path), test)
            source = f"public/{label.lower()}_archive/{i:03d}.png"
            sources = [source]
            if i == 20:
                sources.append("public/ng_archive/020_same_pair_top.png")
            if i == 5 or 14 <= i <= 36:
                escapes.update(sources)
            records.append({
                "id": f"event_{i:03d}",
                "label": label,
                "label_provenance": provenance,
                "split_group": f"AOI_MINUTE_{i:03d}",
                "views": {
                    ("SIDE", "TOP", "MID")[i % 3]: {
                        "reference": ref_path, "test": test_path,
                    }
                },
                "sources": sources,
            })
        assert len(escapes) == 25
        return records, escapes

    def test_unapproved_review_mode_cannot_run_without_verified_epoch3(self):
        with self.assertRaisesRegex(ValueError, "exigem? inicialização|exigem?"):
            train_v3(
                self.root, self.manifest, epochs=1, image_size=64,
                allow_provisional_archive_labels=True,
                compare_verified_pinned_cnn=False,
                use_operator_ng_escape_reviews=True,
            )
        self.assertFalse(
            (self.root / "reports/aoi_quality_cnn_v3/candidates").exists()
        )

    def test_supervised_ng_replays_never_enter_validation_or_test(self):
        torch.manual_seed(71)
        records, escape_sources = self._records()
        provenance = {
            "manifest_sha256": sha256(self.manifest.read_bytes()).hexdigest(),
            "by_category_and_label": {},
            "v3_ng_escape_review_enabled": True,
            "v3_ng_escape_review_sha256": self.review_sha,
            "v3_ng_escape_confirmed_sources": 25,
            "v3_ng_escape_source_paths": sorted(escape_sources),
            "v3_ng_escape_sources_in_dataset": 25,
            "v3_ng_escape_unique_pairs_in_dataset": 24,
        }
        init = AOIQualityCNNV3()
        checkpoint = {
            "image_size": 64,
            "model_weights_not_production_state_dict": init.state_dict(),
        }
        snapshot = self.root / "reports/aoi_quality_cnn_v3/epoch3_test_only.pt"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b"unloaded; patched validated data")
        with (
            patch(
                "src.scripts.train_aoi_quality_cnn_v3_offline.universal_archive_events",
                return_value=(records, provenance),
            ),
            patch(
                "src.core.neural.aoi_quality_cnn_v3_production_trial.validate_trial_snapshot",
                return_value=(checkpoint, snapshot),
            ),
        ):
            report = train_v3(
                self.root, self.manifest, epochs=1, image_size=64,
                allow_provisional_archive_labels=True,
                compare_verified_pinned_cnn=False,
                initialize_from_verified_epoch3=True,
                use_operator_ng_escape_reviews=True,
            )
        self.assertTrue(report["training"]["v3_initialized_from_verified_epoch3"])
        self.assertTrue(report["operator_ng_escape_review_enabled"])
        self.assertEqual(report["operator_ng_escape_review_verified"], 25)
        self.assertEqual(report["partition"]["pinned_training_source_count"], 38)
        self.assertEqual(
            report["operator_verified_ng_escape_training_replay"]["NG_total"], 24
        )
        self.assertEqual(
            report["previous_v3_ng_escape_training_replay"]["NG_total"], 24
        )
        self.assertEqual(report["operator_verified_training_replay"]["metrics"]["total"], 14)
        self.assertEqual(report["coverage"]["validation"]["human_verified_v3_ng_escape"], 0)
        self.assertEqual(report["coverage"]["test"]["human_verified_v3_ng_escape"], 0)
        self.assertTrue(
            report["safety"]["operator_verified_25_NG_escapes_excluded_from_validation_test"]
        )
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["automatic_xp_authorized"])
        self.assertFalse(report["safety"]["production_eligible"])
        initial_val = report["epoch_history"][0]["validation"]
        chosen_val = report["validation"]
        self.assertLessEqual(chosen_val["OK_false_NG"], initial_val["OK_false_NG"])
        self.assertLessEqual(chosen_val["NG_false_OK"], initial_val["NG_false_OK"])
        self.assertFalse(
            (self.root / "reports/aoi_quality_cnn_v3/production_v3_trial.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
