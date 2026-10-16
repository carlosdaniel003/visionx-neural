"""MIL supervision must not invent NG ground-truth for each light."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from unittest.mock import patch

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts import faltando_cnn_v2_guarded_light_repair as repair
from src.scripts.train_faltando_cnn_v2_verified_archive import split_verified_events
from src.services.faltando_neural_archive_experiment import archive_labeled_experimental_events
from tests.test_faltando_cnn_v2_training import CNNV2Fixture


def _assess(_net, events, _size, _focus):
    cases = []
    for event in events:
        label = event["label"]
        value = .8 if label == "NG" else .2
        cases.append({
            "event_id_or_group_hint": event["id"], "label": label,
            "predicted": label, "ng_score": value,
            "correct": True,
            "per_light": {
                light: {"score": value, "correct": True}
                for light in event["views"]
            },
        })
    n = sum(len(event["views"]) for event in events)
    return {
        "events": len(events), "images": n,
        "correct_events": len(events), "correct_images": n,
        "all_correct": bool(events), "ng_false_ok": 0, "cases": cases,
    }


class EventMILSupervisionTests(CNNV2Fixture):
    def test_multilight_ng_does_not_supervise_all_views_positive(self):
        lights = torch.tensor([[.0, 2., -.5]], requires_grad=True)
        mask = torch.ones_like(lights)
        logits = lights.max(1).values
        ng = torch.tensor([1.])
        loss = repair._weak_event_mil_loss(logits, lights, mask, ng, light_weight=.5)
        loss.backward()
        self.assertEqual(float(lights.grad[0, 0]), 0.)
        self.assertLess(float(lights.grad[0, 1]), 0.)
        self.assertEqual(float(lights.grad[0, 2]), 0.)

    def test_single_view_ng_and_multilight_ok_are_supervised_per_light(self):
        for truth, mask in (
            (torch.tensor([1.]), torch.tensor([[0., 1., 0.]])),
            (torch.tensor([0.]), torch.tensor([[1., 1., 1.]])),
        ):
            views = torch.tensor([[.5, .1, -.2]], requires_grad=True)
            logit = views.masked_fill(mask <= 0, -1e6).max(dim=1).values
            repair._weak_event_mil_loss(
                logit, views, mask, truth, light_weight=.5,
            ).backward()
            for idx, m in enumerate(mask[0]):
                if m:
                    self.assertNotEqual(float(views.grad[0, idx]), 0.)
                else:
                    self.assertEqual(float(views.grad[0, idx]), 0.)

    def test_replay_per_light_ng_only_from_single_view_and_online_by_event(self):
        def event(name, label, lights):
            return {
                "id": name, "label": label,
                "views": {light: {} for light in lights},
            }
        archived = [
            event(f"ng-{light}", "NG", (light,)) for light in LIGHTS
        ] + [event("ok-archive", "OK", ("SIDE",))]
        online = [event("ng-online-multilight", "NG", LIGHTS)]
        history = [event("ok-history", "OK", ("SIDE",))]
        with patch.object(repair, "_assess", side_effect=_assess):
            replay, counts = repair._select_mil_hard_pairs(
                FaltandoCNNV2(), history, online, archived, 64, .7,
                per_light=2,
            )
        self.assertEqual(counts["NG"], 8)
        self.assertEqual(counts["OK"], 8)
        self.assertEqual(counts["hard_ng_single_view_by_light"],
                         {"SIDE": 2, "TOP": 2, "MID": 2})
        self.assertEqual(counts["hard_ng_multi_view_event_only"], 2)
        self.assertEqual(counts["protected_original_ok_steps"], 4)
        self.assertFalse(counts["weak_multilight_ng_per_view_supervision"])
        self.assertTrue(all(e["label"] == "NG" for e in replay[::2]))
        self.assertTrue(all(e["label"] == "OK" for e in replay[1::2]))
        self.assertTrue(any(e["id"] == "ok-history" for e in replay[1::2]))
        self.assertTrue(any(e["id"] == "ng-online-multilight" for e in replay[::2]))

    def test_mil_selection_does_not_use_online_per_light_correctness(self):
        hist = _assess(None, [{"id": "hist", "label": "OK",
                               "views": {"SIDE": {}}}], 64, .7)
        online = _assess(None, [{"id": "evt", "label": "NG",
                                 "views": {light: {} for light in LIGHTS}}], 64, .7)
        dev = _assess(None, [{"id": "single", "label": "NG",
                              "views": {"MID": {}}}], 64, .7)
        first = repair._mil_selection_score(hist, online, dev)
        online["correct_images"] = -99999  # Should not impact scoring
        for case in online["cases"]:
            for light in case["per_light"]:
                case["per_light"][light]["correct"] = False
        self.assertEqual(first, repair._mil_selection_score(hist, online, dev))

    def test_opt_in_mil_training_produces_candidate_without_promotion(self):
        archive, provenance = archive_labeled_experimental_events(self.manifest_path)
        train, hold, split = split_verified_events(archive)
        model = FaltandoCNNV2()
        champion = {
            "schema": MODEL_SCHEMA_V2,
            "state_dict": deepcopy(model.state_dict()),
            "lights": ("SIDE", "TOP", "MID"),
            "image_size": 64, "focus_fraction": .7,
            "experimental": True, "production_approved": False,
        }
        origin = self.root / "base.pt"
        torch.save(champion, origin)
        start = {
            **champion,
            "source_champion_sha256": sha256(origin.read_bytes()).hexdigest(),
            "provisional_archive_manifest_sha256": provenance["manifest_sha256"],
            "archive_labels_experimental": True,
            "individual_pair_review_complete": False,
            "weight_blend_offline": True,
        }
        start_report = {
            "base_checkpoint_sha256": sha256(origin.read_bytes()).hexdigest(),
            "candidate_sha256": "sha",
            "split": split,
        }
        ok = [e for e in train if e["label"] == "OK"]
        ng = [e for e in train if e["label"] == "NG"]
        seen = []
        def assess_traced(net, events, size, focus):
            seen.append(tuple(e["id"] for e in events))
            return _assess(net, events, size, focus)

        with (
            patch.object(repair, "_base_model", return_value=(origin, champion)),
            patch.object(repair, "_load_start", return_value=(start_report, start)),
            patch.object(repair, "_historical", return_value=ok[:2]),
            patch.object(repair, "_all_online", return_value=ng[:2]),
            patch.object(repair, "_assess", side_effect=assess_traced),
            patch.object(repair, "_select_mil_hard_pairs", return_value=(
                [ng[0], ok[0]] * 2,
                {"NG": 2, "OK": 2},
            )),
        ):
            result = repair.repair_offline(
                self.root, self.manifest_path, self.root / "start.json",
                epochs=1, supervision_mode="event_mil",
            )
        self.assertEqual(result["supervision_mode"], "event_mil")
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["production_approval_eligible"])
        self.assertFalse(result["holdout_used_for_selection"])
        self.assertEqual(seen[-2:],
                         [tuple(e["id"] for e in hold)] * 2)
        ckpt = torch.load(
            self.root / result["candidate_relative_path"], map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(ckpt["supervision_mode"], "event_mil")
        self.assertFalse(ckpt["production_approved"])
        self.assertFalse((self.root/"reports/neural_online/live_active.json").exists())
