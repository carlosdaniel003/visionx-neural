"""CNN V2: arquivo OK/NG só entra em treinamento após revisão humana."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    qualified_archive_events, split_verified_events, candidate_acceptance,
    train_verified_archive,
)
from src.services.faltando_neural_qualification import QualificationStore
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


class VerifiedArchiveFineTuneTests(CNNV2Fixture):
    def qualify(self, *, approve_all=True):
        store = QualificationStore(self.manifest_path)
        for source, row in sorted(store.by_path.items()):
            if approve_all or Path(source).name != "case_1_SIDE_FALTANDO.png":
                label = row["expected_label_from_archive"]
                store.review_case(source, "CONFIRMED_" + label)
        group_id = "2026-10-08_0735_FALTANDO"
        store.review_group(group_id, "CONFIRMED_VISUAL_ASSOCIATION")
        return store

    def test_archive_folder_alone_is_not_verified_training(self):
        with self.assertRaisesRegex(ValueError, "qualification.json ausente"):
            qualified_archive_events(self.manifest_path)

    def test_reviewed_pairs_and_group_are_real_training_events(self):
        store = self.qualify()
        events, provenance = qualified_archive_events(self.manifest_path)
        self.assertEqual(len(events), 13)  # 6 OK + 6 NG + 1 trinca OK.
        self.assertEqual(sum(len(e["views"]) for e in events), 15)
        self.assertEqual(provenance["human_multilight_groups"], 1)
        self.assertEqual(provenance["confirmed_labels"], {"OK": 7, "NG": 6})
        triple = [e for e in events if len(e["views"]) == 3]
        self.assertEqual(len(triple), 1)
        self.assertEqual(triple[0]["label"], "OK")
        self.assertEqual(set(triple[0]["views"]), {"SIDE", "TOP", "MID"})
        # Na avaliação, cada grupo humano é indivisível.
        train, hold, report = split_verified_events(events, seed=42)
        self.assertEqual(len(train)+len(hold), len(events))
        self.assertEqual(set(e["id"] for e in train) &
                         set(e["id"] for e in hold), set())
        self.assertTrue(report["no_near_duplicates_cross_boundary"])
        self.assertGreaterEqual(report["holdout_by_label"]["NG"], 2)
        self.assertGreaterEqual(report["holdout_by_label"]["OK"], 2)
        self.assertFalse(store.output.is_symlink())

    def test_unreviewed_pngs_are_excluded(self):
        self.qualify(approve_all=False)
        events, provenance = qualified_archive_events(self.manifest_path)
        self.assertEqual(len(events), 12)
        self.assertEqual(provenance["excluded_unconfirmed"], 1)

    def test_mutating_archive_png_blocks_training(self):
        self.qualify()
        path = self.root/"public/ng_archive/case_20_SIDE_FALTANDO.png"
        path.write_bytes(path.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "modificado"):
            qualified_archive_events(self.manifest_path)

    def test_qualification_file_without_human_review_is_not_accepted(self):
        store = self.qualify()
        raw = json.loads(store.output.read_text(encoding="utf-8"))
        raw["automated_training_allowed"] = True
        store.output.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Qualificação inválida"):
            qualified_archive_events(self.manifest_path)

    def test_same_pair_with_opposing_human_labels_is_refused(self):
        self.qualify()
        events, _ = qualified_archive_events(self.manifest_path)
        a = deepcopy(events[0])
        a["id"] = "opposite"
        a["label"] = "NG" if events[0]["label"] == "OK" else "OK"
        with self.assertRaisesRegex(ValueError, "OK e NG visualmente duplicados"):
            split_verified_events(events+[a])

    def test_near_duplicate_cannot_cross_train_and_holdout(self):
        self.qualify()
        events, _ = qualified_archive_events(self.manifest_path)
        x = deepcopy(events[0])
        x["id"] = "same-other-capture"
        x["component"] = ("SOME-OTHER-BOARD", "PART-42")
        train, hold, _ = split_verified_events(events+[x])
        self.assertEqual(
            any(e["id"] == events[0]["id"] for e in train),
            any(e["id"] == x["id"] for e in train),
        )

    def test_component_identity_prevents_splitting_repeated_captures(self):
        self.qualify()
        events, _ = qualified_archive_events(self.manifest_path)
        x = deepcopy(events[0])
        x["id"] = "other-scan-same-component"
        # Reutilizar outro par da mesma classe, mas preservar placa/peça.
        another = next(e for e in events if e["label"] == x["label"]
                       and e["id"] != x["id"] and set(e["views"]) == set(x["views"]))
        x["views"] = another["views"]
        train, hold, _ = split_verified_events(events+[x])
        self.assertEqual(
            any(e["id"] == events[0]["id"] for e in train),
            any(e["id"] == x["id"] for e in train),
        )

    def test_baseline_candidate_holdout_must_not_create_new_errors(self):
        def result(labels):
            return {
                "all_correct": all(label == prediction for label, prediction in labels),
                "correct_images": sum(label == prediction for label, prediction in labels),
                "ng_false_ok": sum(label == "NG" and prediction == "OK"
                                   for label, prediction in labels),
                "cases": [
                    {"label": l, "correct": l == p,
                     "per_light": {"SIDE": {"correct": l == p}}}
                    for l, p in labels
                ],
            }
        original = result([("OK", "OK"), ("NG", "OK"), ("NG", "NG")])
        new_good = result([("OK", "OK"), ("NG", "NG"), ("NG", "NG")])
        accepted = candidate_acceptance(
            result([("OK", "OK")]), result([("OK", "OK")]),
            result([]), result([]), original, new_good,
        )
        self.assertTrue(accepted["passed"])
        new_bad = result([("OK", "NG"), ("NG", "NG"), ("NG", "NG")])
        rejected = candidate_acceptance(
            result([("OK", "OK")]), result([("OK", "OK")]),
            result([]), result([]), original, new_bad,
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["gates"]["holdout_ok_not_worse"])

    def test_real_optimizer_writes_distinct_candidate_without_pointer(self):
        self.qualify()
        # O histórico completo é simulado somente neste teste: o gradiente
        # e o arquivo .pt são REAIS, os gates sintéticos não certificam acurácia.
        champion = FaltandoCNNV2()
        checkpoint_path = self.root / "base.pt"
        checkpoint = {
            "schema": MODEL_SCHEMA_V2,
            "state_dict": champion.state_dict(),
            "lights": ("SIDE", "TOP", "MID"),
            "image_size": 64, "focus_fraction": .70,
            "experimental": True, "production_approved": False,
        }
        torch.save(checkpoint, checkpoint_path)
        all_events, _ = qualified_archive_events(self.manifest_path)
        history = all_events[:2]
        def safe_assess(model, records, size, focus):
            return {
                "all_correct": True,
                "correct_images": sum(len(e["views"]) for e in records),
                "ng_false_ok": 0,
                "cases": [
                    {"label": e["label"], "correct": True,
                     "per_light": {
                         light: {"correct": True}
                         for light in e["views"]
                     }}
                    for e in records
                ],
            }
        with (
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._base_model",
                  return_value=(checkpoint_path, checkpoint)),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._historical",
                  return_value=history),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._all_online",
                  return_value=[]),
            patch("src.scripts.train_faltando_cnn_v2_verified_archive._assess",
                  side_effect=safe_assess),
        ):
            report = train_verified_archive(
                self.root, self.manifest_path, epochs=1,
            )
        self.assertTrue(report["candidate_only"])
        self.assertFalse(report["promoted"])
        self.assertTrue(report["acceptance"]["passed"])
        candidate = self.root / report["candidate_relative_path"]
        self.assertTrue(candidate.exists())
        self.assertEqual(sha256(candidate.read_bytes()).hexdigest(),
                         report["candidate_sha256"])
        self.assertNotEqual(sha256(checkpoint_path.read_bytes()).hexdigest(),
                            report["candidate_sha256"])
        self.assertFalse((self.root/"reports/neural_online/live_active.json").exists())
        self.assertTrue((candidate.parent/"training_report.json").is_file())
        self.assertEqual(report["epochs"], 1)
        self.assertEqual(report["provenance"]["human_multilight_groups"], 1)


if __name__ == "__main__":
    unittest.main()
