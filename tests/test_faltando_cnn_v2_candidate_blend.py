"""Offline CNN V2 weight interpolation: integrity, no holdout tuning, no promotion."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.faltando_cnn_v2_blend_candidates import (
    _mixed_weights, _verified_candidate, blend_offline,
)
from src.scripts.train_faltando_cnn_v2_verified_archive import split_verified_events
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


class CandidateBlendTests(CNNV2Fixture):
    def _make_candidates(self):
        events, provenance = archive_labeled_experimental_events(self.manifest_path)
        train, hold, split = split_verified_events(events)
        model = FaltandoCNNV2()
        champion = {
            "schema": MODEL_SCHEMA_V2,
            "state_dict": deepcopy(model.state_dict()),
            "lights": ("SIDE", "TOP", "MID"),
            "image_size": 64, "focus_fraction": .70,
            "experimental": True, "production_approved": False,
        }
        base = self.root / "base.pt"
        torch.save(champion, base)
        result = []
        for i in range(2):
            folder = self.root / f"reports/faltando_neural/archive_experiments/candidate_{i}"
            folder.mkdir(parents=True)
            ckpt = {
                **champion,
                "state_dict": deepcopy(model.state_dict()),
                "archive_labels_experimental": True,
                "individual_pair_review_complete": False,
                "source_champion_sha256": _sha(base),
                "provisional_archive_manifest_sha256": provenance["manifest_sha256"],
            }
            first_key = next(iter(ckpt["state_dict"]))
            ckpt["state_dict"][first_key].add_(i * .0001)
            path = folder / "faltando_v2_archive_experimental_candidate.pt"
            torch.save(ckpt, path)
            report = {
                "qualification_mode": "ARCHIVE_FOLDER_BULK_EXPERIMENT",
                "candidate_only": True, "promoted": False,
                "production_approval_eligible": False,
                "base_checkpoint_sha256": _sha(base),
                "candidate_relative_path": path.relative_to(self.root).as_posix(),
                "candidate_sha256": _sha(path),
                "provenance": provenance,
                "train_event_ids": [e["id"] for e in train],
                "holdout_event_ids": [e["id"] for e in hold],
                "split": split,
            }
            report_path = folder / "training_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result.append((report_path, ckpt))
        return base, champion, events, result

    def test_mixed_weights_are_only_tensors_and_preserve_endpoints(self):
        base, champion, _, reports = self._make_candidates()
        left, right = reports[0][1], reports[1][1]
        middle = _mixed_weights(left, right, .5)
        key = next(iter(middle))
        self.assertTrue(torch.allclose(
            middle[key],
            (left["state_dict"][key] + right["state_dict"][key]) / 2,
        ))
        with self.assertRaisesRegex(ValueError, "Estados"):
            _mixed_weights(left, right, -0.1)

    def test_modified_candidate_or_report_rejected(self):
        _, _, _, reports = self._make_candidates()
        first = reports[0][0]
        verified, checkpoint = _verified_candidate(self.root, first)
        self.assertEqual(verified["candidate_sha256"], _sha(
            self.root / verified["candidate_relative_path"]
        ))
        ckpt_path = self.root / verified["candidate_relative_path"]
        ckpt_path.write_bytes(ckpt_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _verified_candidate(self.root, first)
        with self.assertRaisesRegex(ValueError, "fora da pasta"):
            _verified_candidate(self.root, self.manifest_path)

    def test_blend_only_audited_candidate_without_activation(self):
        base, champion, events, candidates = self._make_candidates()
        from src.scripts import faltando_cnn_v2_blend_candidates as blend

        def assess(_model, rows, _size, _focus):
            cases = [
                {
                    "label": event["label"], "correct": True,
                    "per_light": {
                        light: {"score": .999 if event["label"] == "NG" else .001,
                                "correct": True}
                        for light in event["views"]
                    },
                } for event in rows
            ]
            return {
                "events": len(rows),
                "images": sum(len(e["views"]) for e in rows),
                "correct_images": sum(len(e["views"]) for e in rows),
                "all_correct": True, "ng_false_ok": 0, "cases": cases,
            }
        with (
            patch.object(blend, "_base_model", return_value=(base, champion)),
            patch.object(blend, "_historical", return_value=events[:2]),
            patch.object(blend, "_all_online", return_value=events[:2]),
            patch.object(blend, "_assess", side_effect=assess),
        ):
            result = blend_offline(
                self.root, self.manifest_path,
                candidates[0][0], candidates[1][0], steps=2,
            )
        self.assertEqual(len(result["blend_grid"]), 3)
        self.assertEqual(result["selected_aggressive_weight"], 0.)
        self.assertTrue(result["acceptance"]["passed"])
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["production_approval_eligible"])
        self.assertFalse(result["independent_validation_claim"])
        self.assertFalse(result["promoted"])
        self.assertFalse(result["holdout_used_for_selection"])
        self.assertTrue(result["holdout_previously_examined_in_prior_experiments"])
        new_ckpt = torch.load(
            self.root / result["candidate_relative_path"],
            map_location="cpu", weights_only=True,
        )
        self.assertFalse(new_ckpt["production_approved"])
        self.assertTrue(new_ckpt["weight_blend_offline"])
        self.assertFalse((self.root / "reports/neural_online/live_active.json").exists())

    def test_refuse_mismatched_holdout_or_live_pointer(self):
        base, champion, events, candidates = self._make_candidates()
        p1, p2 = (candidates[0][0], candidates[1][0])
        data = json.loads(p2.read_text(encoding="utf-8"))
        data["holdout_event_ids"] = ["untrusted_event"]
        p2.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "diferentes origens/splits"):
            blend_offline(self.root, self.manifest_path, p1, p2, steps=2)
        pointer = self.root / "reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True)
        pointer.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Checkpoint online ativo"):
            blend_offline(self.root, self.manifest_path, p1, p2, steps=2)
