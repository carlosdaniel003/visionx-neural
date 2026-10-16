"""The corrective FALTANDO V2 trainer cannot see holdout during selection."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts import faltando_cnn_v2_guarded_light_repair as repair
from src.scripts.train_faltando_cnn_v2_verified_archive import split_verified_events
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


def perfect_assess(_model, rows, _size, _focus):
    cases = []
    for event in rows:
        score = .999 if event["label"] == "NG" else .001
        cases.append({
            "label": event["label"], "correct": True,
            "per_light": {
                light: {"score": score, "correct": True}
                for light in event["views"]
            },
        })
    images = sum(len(e["views"]) for e in rows)
    return {
        "events": len(rows), "images": images,
        "correct_events": len(rows), "correct_images": images,
        "all_correct": bool(rows), "ng_false_ok": 0, "cases": cases,
    }


class GuardedLightRepairTests(CNNV2Fixture):
    def test_hard_replay_has_ng_each_light_and_original_ok_protection(self):
        class Dummy:
            pass

        def event(i, label):
            return {
                "id": f"{label}:{i}", "label": label,
                "views": {light: {} for light in LIGHTS},
            }

        original_ok = [event(1, "OK"), event(2, "OK")]
        archive_train = [event(i, "NG") for i in range(1, 5)] + [event(3, "OK")]
        with patch.object(repair, "_assess", side_effect=perfect_assess):
            paired, counts = repair._select_hard_pairs(
                Dummy(), original_ok, [], archive_train, 64, .70,
                per_light=2,
            )
        self.assertEqual(counts["NG"], 6)
        self.assertEqual(counts["OK"], 6)
        self.assertEqual(counts["hard_ng_by_illumination"],
                         {"SIDE": 2, "TOP": 2, "MID": 2})
        self.assertEqual(counts["protected_original_ok_steps"], 2)
        self.assertTrue(all(row["label"] == "NG" for row in paired[::2]))
        self.assertTrue(all(row["label"] == "OK" for row in paired[1::2]))
        self.assertIn(original_ok[0], paired[1::2])

    def test_repair_candidate_only_and_holdout_outside_selection(self):
        events, provenance = archive_labeled_experimental_events(self.manifest_path)
        train, hold, split = split_verified_events(events)
        model = FaltandoCNNV2()
        base = {
            "schema": MODEL_SCHEMA_V2,
            "state_dict": deepcopy(model.state_dict()),
            "lights": ("SIDE", "TOP", "MID"),
            "image_size": 64, "focus_fraction": .70,
            "experimental": True, "production_approved": False,
        }
        base_path = self.root / "base.pt"
        torch.save(base, base_path)
        from hashlib import sha256
        digest = sha256(base_path.read_bytes()).hexdigest()
        start = {
            **base,
            "source_champion_sha256": digest,
            "provisional_archive_manifest_sha256": provenance["manifest_sha256"],
            "archive_labels_experimental": True, "individual_pair_review_complete": False,
            "weight_blend_offline": True,
        }
        start_report = {
            "base_checkpoint_sha256": digest,
            "candidate_sha256": "start_sha_test",
            "split": split,
        }
        # Only fake model assessments; gradient updates run using real fixture PNGs.
        visited = []
        def assess(model, rows, size, focus):
            visited.append(tuple(e["id"] for e in rows))
            return perfect_assess(model, rows, size, focus)
        ok = [e for e in train if e["label"] == "OK"]
        ng = [e for e in train if e["label"] == "NG"]
        self.assertTrue(ok and ng)
        with (
            patch.object(repair, "_base_model", return_value=(base_path, base)),
            patch.object(repair, "_load_start", return_value=(start_report, start)),
            patch.object(repair, "_historical", return_value=ok[:2]),
            patch.object(repair, "_all_online", return_value=ng[:2]),
            patch.object(repair, "_assess", side_effect=assess),
            patch.object(
                repair, "_select_hard_pairs",
                return_value=(
                    [ng[0], ok[0]] * 2,
                    {
                        "NG": 2, "OK": 2,
                        "hard_ng_by_illumination": {"SIDE": 2, "TOP": 0, "MID": 0},
                        "protected_original_ok_steps": 1,
                    },
                ),
            ),
        ):
            report = repair.repair_offline(
                self.root, self.manifest_path,
                self.root / "dummy_blend_report.json", epochs=1,
            )
        self.assertEqual(report["selected_epoch"], 0)
        self.assertTrue(report["candidate_only"])
        self.assertFalse(report["promoted"])
        self.assertFalse(report["production_approval_eligible"])
        self.assertFalse(report["holdout_used_for_selection"])
        self.assertTrue(report["holdout_previously_examined_in_prior_experiments"])
        self.assertFalse(report["independent_validation_claim"])
        self.assertTrue(report["training_only_head"])
        self.assertTrue(report["acceptance"]["passed"])
        self.assertTrue((self.root / report["candidate_relative_path"]).is_file())
        out_ckpt = torch.load(
            self.root / report["candidate_relative_path"],
            weights_only=True, map_location="cpu",
        )
        self.assertFalse(out_ckpt["production_approved"])
        self.assertTrue(out_ckpt["guarded_light_repair_offline"])
        self.assertFalse((self.root / "reports/neural_online/live_active.json").exists())
        self.assertTrue(all(set(batch).isdisjoint({e["id"] for e in hold})
                            for batch in visited[:-2] if batch != tuple()))
        self.assertEqual(
            visited[-2:],
            [tuple(e["id"] for e in hold), tuple(e["id"] for e in hold)],
        )

    def test_refuses_online_pointer_before_train(self):
        pointer = self.root / "reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Checkpoint online ativo"):
            repair.repair_offline(self.root, self.manifest_path, "missing.json")

    def test_refuses_untracked_start(self):
        events, provenance = archive_labeled_experimental_events(self.manifest_path)
        train, hold, _ = split_verified_events(events)
        with self.assertRaises(ValueError):
            repair._load_start(
                self.root, self.manifest_path,
                provenance["manifest_sha256"], train, hold,
            )
