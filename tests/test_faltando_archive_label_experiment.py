"""Provisional, opt-in FALTANDO archive training: no fabricated human review."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import unittest

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    qualified_archive_events, split_verified_events, train_verified_archive,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)
from src.services.faltando_neural_qualification import QualificationStore
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


class BulkArchiveExperimentTests(CNNV2Fixture):
    def test_provisional_does_not_claim_review_or_multilight(self):
        with self.assertRaisesRegex(ValueError, "qualification.json ausente"):
            qualified_archive_events(self.manifest_path)
        events, provenance = archive_labeled_experimental_events(self.manifest_path)
        self.assertEqual(provenance["source_pairs"], 15)
        self.assertEqual(len(events), 15)
        self.assertFalse(provenance["individual_pair_review_complete"])
        self.assertEqual(provenance["human_multilight_groups"], 0)
        self.assertTrue(all(len(e["views"]) == 1 for e in events))
        self.assertFalse((self.manifest_path.parent / "qualification.json").exists())
        self.assertEqual(len([
            e for e in events if e.get("split_group") and "::name::" in e["split_group"]
        ]), 3)
        train, hold, _ = split_verified_events(events)
        self.assertEqual(len(train) + len(hold), 15)
        grouped = [e for e in events if e.get("split_group")
                   and "::name::" in e["split_group"]]
        self.assertEqual(len({e["split_group"] for e in grouped}), 1)
        self.assertTrue(all(e in train for e in grouped)
                        or all(e in hold for e in grouped))

    def test_modified_source_rejected_before_learning(self):
        source = self.root / "public/ng_archive/case_20_SIDE_FALTANDO.png"
        source.write_bytes(source.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "modificado"):
            archive_labeled_experimental_events(self.manifest_path)

    def test_explicitly_rejected_case_is_not_overridden_by_bulk(self):
        store = QualificationStore(self.manifest_path)
        first = next(iter(store.by_path))
        store.review_case(first, "REJECTED", "Recorte ruim")
        with self.assertRaisesRegex(ValueError, "previamente rejeitado"):
            archive_labeled_experimental_events(self.manifest_path)

    def test_no_training_without_opt_in(self):
        with self.assertRaisesRegex(ValueError, "qualification.json ausente"):
            train_verified_archive(self.root, self.manifest_path, epochs=1)

    def test_true_gradient_candidate_only_and_no_review_forgery(self):
        events, _ = archive_labeled_experimental_events(self.manifest_path)
        champion = FaltandoCNNV2()
        base = self.root / "base.pt"
        checkpoint = {
            "schema": MODEL_SCHEMA_V2,
            "state_dict": champion.state_dict(),
            "lights": ("SIDE", "TOP", "MID"),
            "image_size": 64, "focus_fraction": .70,
            "experimental": True, "production_approved": False,
        }
        torch.save(checkpoint, base)

        def safe_assess(model, records, size, focus):
            return {
                "all_correct": True,
                "correct_images": sum(len(e["views"]) for e in records),
                "ng_false_ok": 0,
                "cases": [
                    {"label": e["label"], "correct": True,
                     "per_light": {light: {"correct": True}
                                   for light in e["views"]}}
                    for e in records
                ],
            }

        with (
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._base_model",
                  return_value=(base, checkpoint)),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._historical",
                  return_value=events[:2]),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._all_online",
                  return_value=[]),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._assess",
                  side_effect=safe_assess),
        ):
            report = train_verified_archive(
                self.root, self.manifest_path, epochs=1,
                archive_labels_experimental=True,
            )

        self.assertTrue(report["candidate_only"])
        self.assertFalse(report["promoted"])
        self.assertFalse(report["automatic_ok_authorized"])
        self.assertFalse(report["production_approval_eligible"])
        self.assertEqual(report["qualification_mode"],
                         "ARCHIVE_FOLDER_BULK_EXPERIMENT")
        self.assertIn("archive_experiments", report["candidate_relative_path"])
        candidate = self.root / report["candidate_relative_path"]
        self.assertTrue(candidate.exists())
        weights = torch.load(candidate, map_location="cpu", weights_only=True)
        self.assertTrue(weights["archive_labels_experimental"])
        self.assertFalse(weights["production_approved"])
        self.assertFalse(weights["individual_pair_review_complete"])
        self.assertFalse((self.root / "reports/neural_online/live_active.json").exists())
        self.assertFalse((self.manifest_path.parent / "qualification.json").exists())


if __name__ == "__main__":
    unittest.main()
