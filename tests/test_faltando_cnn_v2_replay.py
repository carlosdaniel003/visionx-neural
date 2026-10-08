"""Replay FALTANDO v2: todos os PNGs/3 luzes, sem KNN e sem mudanças ao ODIN."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import torch

from src.scripts.train_faltando_cnn_v2 import train_v2
from src.scripts.replay_faltando_cnn_v2 import replay_archive, _counts
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


class ReplayArchiveCNNV2Tests(CNNV2Fixture):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def create_checkpoint(self):
        _, run = train_v2(
            self.manifest_path, epochs=1, size=64,
            batch_size=4, patience=2,
        )
        return run/"faltando_cnn_v2_candidate.pt"

    def test_replay_uses_each_png_and_each_event_without_knn(self):
        checkpoint = self.create_checkpoint()
        source_hashes = {p: digest for p, digest in self.sources}
        report, out = replay_archive(
            self.root, checkpoint=checkpoint, manifest=self.manifest_path
        )
        self.assertEqual(report["images_scanned"], 15)
        self.assertEqual(report["events_scanned"], 13)
        self.assertEqual(report["counts_per_image"]["cases"], 15)
        self.assertEqual(report["counts_per_event"]["cases"], 13)
        self.assertEqual(report["counts_per_light"]["SIDE"]["cases"], 13)
        self.assertEqual(report["counts_per_light"]["TOP"]["cases"], 1)
        self.assertEqual(report["counts_per_light"]["MID"]["cases"], 1)
        self.assertEqual(report["counts_legacy_side"]["cases"], 0)
        self.assertEqual(len(report["image_records"]), 15)
        self.assertEqual(len(report["event_records"]), 13)
        self.assertTrue(all(len(row["sources"]) in (1, 3)
                            for row in report["event_records"]))
        self.assertEqual(len({row["source_path"] for row in report["image_records"]}), 15)
        self.assertFalse(report["ready_for_automatic_production"])
        self.assertFalse(report["can_prove_generalization"])
        self.assertFalse(report["knn_used"])
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["production_modified"])
        self.assertTrue((out/"archive_replay_v2.json").is_file())
        self.assertTrue((out/"archive_replay_v2.txt").is_file())
        self.assertEqual(
            {p: sha256(p.read_bytes()).hexdigest() for p in source_hashes},
            source_hashes,
        )
        self.assertEqual(list((self.root/"public").rglob("*.pt")), [])

    def test_changed_archive_after_training_aborts_as_incomplete(self):
        checkpoint = self.create_checkpoint()
        extra = self.root/"public"/"ng_archive"/"new_FALTANDO.png"
        extra.write_bytes(self.sources[0][0].read_bytes())
        with self.assertRaisesRegex(ValueError, "Adicionados=1"):
            replay_archive(
                self.root, manifest=self.manifest_path, checkpoint=checkpoint
            )

    def test_missing_archive_after_training_aborts(self):
        checkpoint = self.create_checkpoint()
        self.sources[0][0].unlink()
        with self.assertRaises(ValueError):
            replay_archive(
                self.root, manifest=self.manifest_path, checkpoint=checkpoint
            )

    def test_checkpoint_rejects_mismatched_manifest(self):
        checkpoint = self.create_checkpoint()
        before = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        before["new_note"] = "alteração no manifesto sem retreinar"
        self.manifest_path.write_text(json.dumps(before), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "não corresponde"):
            replay_archive(
                self.root, manifest=self.manifest_path, checkpoint=checkpoint
            )

    def test_count_defects_are_errors_not_accuracy_only(self):
        rows = [
            {"label": "OK", "predicted": "OK"},
            {"label": "OK", "predicted": "OK"},
            {"label": "NG", "predicted": "OK"},
        ]
        counts = _counts(rows)
        self.assertEqual(counts["accuracy"], 0.666667)
        self.assertEqual(counts["FN_NG_as_OK"], 1)
        self.assertFalse(counts["passed_all"])
        rows[-1]["predicted"] = "NG"
        self.assertTrue(_counts(rows)["passed_all"])


if __name__ == "__main__":
    unittest.main()
