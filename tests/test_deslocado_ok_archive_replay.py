"""Replay estrito DESLOCADO v2 de TODOS os OK (arquivo conhecido).

Sem NG reais não há autorização de liberar Produção, mesmo com 100% OK.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.scripts.train_deslocado_cnn_v2 import train_deslocado_v2
from src.scripts.replay_deslocado_ok_v2 import (
    all_archived_deslocado, replay_deslocado, summarize,
)
from tests.test_deslocado_cnn import DeslocadoFixtures


class DeslocadoArchiveReplayTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        _, self.run = self.prepared()
        _, self.model_dir = train_deslocado_v2(
            self.run/"manifest.json", epochs=1, size=64,
            batch_size=2, proxy_variants=1,
        )
        self.checkpoint = self.model_dir/"deslocado_cnn_v2_candidate.pt"

    def replay(self):
        return replay_deslocado(
            self.root, manifest=self.run/"manifest.json",
            checkpoint=self.checkpoint, batch_size=2
        )

    def test_each_archived_ok_image_and_multilight_event_is_evaluated(self):
        existing = {
            p: sha256(p.read_bytes()).hexdigest()
            for p in (self.root/"public"/"ok_archive").glob("*.png")
        }
        report, output = self.replay()
        self.assertEqual(report["total_archive_pngs_evaluated"], 6)
        self.assertEqual(report["total_events_evaluated"], 4)
        self.assertEqual(report["per_lighting"]["SIDE"]["OK_total"], 4)
        self.assertEqual(report["per_lighting"]["TOP"]["OK_total"], 1)
        self.assertEqual(report["per_lighting"]["MID"]["OK_total"], 1)
        self.assertEqual(report["legacy_SIDE"]["OK_total"], 3)
        self.assertEqual(report["per_image"]["OK_total"], 6)
        self.assertEqual(report["per_event"]["OK_total"], 4)
        self.assertEqual(
            len({r["source_path"] for r in report["all_image_predictions"]}), 6
        )
        self.assertFalse(report["safe_to_replace_operational_engine"])
        self.assertFalse(report["safe_for_auto_OK"])
        self.assertFalse(report["can_validate_NG_detection"])
        self.assertEqual(report["real_NG_images_evaluated"], 0)
        self.assertFalse(report["knn_used"])
        self.assertFalse(report["model_promoted"])
        self.assertEqual(report["model_best_epoch"], 1)
        self.assertTrue((output/"deslocado_ok_replay_v2.json").is_file())
        self.assertTrue((output/"deslocado_ok_replay_v2.txt").is_file())
        self.assertEqual(
            existing,
            {p: sha256(p.read_bytes()).hexdigest() for p in existing},
        )

    def test_new_archive_png_blocks_incomplete_regression(self):
        file = self.root/"public"/"ok_archive"/"another_DESLOCADO.png"
        cv2.imwrite(str(file), np.full((60,70,3), 30, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "manifesto não cobre TODO"):
            self.replay()

    def test_ng_real_appearing_blocks_ok_only_protocol(self):
        file = self.root/"public"/"ng_archive"/"2026-10-08_1630_DESLOCADO.png"
        cv2.imwrite(str(file), np.full((80, 90, 3), 50, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "NG DESLOCADO real"):
            self.replay()

    def test_modified_historical_png_blocks_evaluation(self):
        orig = self.root/"public"/"ok_archive"/"2026-10-02_1349_DESLOCADO.png"
        orig.write_bytes(orig.read_bytes() + b"ALTERADO")
        with self.assertRaisesRegex(ValueError, "original alterado"):
            self.replay()

    def test_all_ok_accuracy_not_sufficient_for_prod_activation(self):
        ok = summarize([
            {"correct": True}, {"correct": True},
            {"correct": True}
        ])
        self.assertTrue(ok["all_OK_correct"])
        self.assertEqual(ok["false_NG_on_OK"], 0)
        report, _ = self.replay()
        self.assertFalse(report["safe_to_replace_operational_engine"])


if __name__ == "__main__":
    unittest.main()
