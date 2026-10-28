"""Regression: operator removed wrong NG (1414), canonical OK (1416) remains.

No stale NG archive entry is permitted in the KNN C pool. The supervised
recognition validates the REAL canonical OK SHA, not the removed file.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from src.core.operator_confirmed_archive_labels import (
    CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256, RETIRED_SOURCE,
    verified_canonical_ok_source,
)
from src.core.knn_c_live import LiveKNNStrategyC
from src.core.knn_visual_corroboration import apply_knn_visual_corroboration
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from tests.test_knn_three_light_ok_counterevidence import fused, report


def event(side_local_verdict="FALHA FALSA"):
    a = fused()
    a.update(verdict="REVISÃO OBRIGATÓRIA", is_defect=False,
             production_review_required=True, confidence=.5)
    d = a["detail"]
    d.update(
        fusion_rule="multilight_local_review",
        multilight_positive_modes=[],
        multilight_strong_positive_modes=[],
        multilight_review_modes=["SIDE", "TOP", "MID"],
        cnn_v2_light_diagnostics={
            light: {
                "route": "NEW_CNN", "category": "FALTANDO",
                "cnn_active": True, "checkpoint_verified": True,
                "cnn_status": "INFERENCE_OK",
                "verdict": (
                    side_local_verdict if light == "SIDE"
                    else "REVISÃO OBRIGATÓRIA"
                ),
                "checkpoint_sha256": "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c",
                "ng_score_uncalibrated": score,
            } for light, score in (
                ("SIDE", .0343675725), ("TOP", .25600528717),
                ("MID", .85979676247),
            )
        },
    )
    # After 1414 is removed, 1416 is the closest confirmed OK SIDE.
    d["knn_c_light_observer_votes"]["SIDE"].update(
        status="REVISAO", reason="DISCORDANCIA_KNN_A_B",
        vote_A="NG", vote_B="OK", suggestion_A="REVISAO",
        suggestion_B="REVISAO", similarity_A=.86618201,
        similarity_B=.83100362,
    )
    for light in ("TOP", "MID"):
        d["knn_c_light_observer_votes"][light].update(
            status="REVISAO", vote_A="OK", vote_B="OK",
            suggestion_A="REVISAO", suggestion_B="REVISAO",
        )
    def n(label, sim, path="", pixel=""):
        return {"label": label, "similarity": sim,
                "path": path or f"public/{label.lower()}_archive/sample.png",
                "source_pixel_sha256": pixel or "d"*64}
    side_a = [
        n("OK", .86618201, CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256),
        n("NG", .82163046), n("OK", .82143573),
        n("NG", .81505456), n("NG", .81477241),
    ]
    side_b = [
        n("OK", .83100362, CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256),
        n("OK", .78586923), n("OK", .78237802),
        n("OK", .77834246), n("OK", .77572797),
    ]
    def group(ng_slot):
        return [n("NG" if i == ng_slot else "OK", .87-i*.02)
                for i in range(5)]
    d["knn_c_light_neighbor_evidence"] = {
        "SIDE": {"neighbors_A": side_a, "neighbors_B": side_b},
        "TOP": {"neighbors_A": group(-1), "neighbors_B": group(4)},
        "MID": {"neighbors_A": group(1), "neighbors_B": group(1)},
    }
    visual = report()
    visual["strong_ng_lights"] = []
    visual["trend_ng_lights"] = []
    for light, ok, ng in (
        ("SIDE", .977865, .988808),
        ("TOP", .972946, .935611),
        ("MID", .960225, .932796),
    ):
        visual["lights"][light].update(
            status="CONFLICT",
            ok={"best_similarity": ok},
            ng={"best_similarity": ng},
            ng_margin_over_ok=round(ng-ok, 6),
        )
    return a, visual


class RetiredWrongNGTests(unittest.TestCase):
    def test_canonical_source_verification_requires_wrong_ng_absent(self):
        import src.core.operator_confirmed_archive_labels as mod
        with TemporaryDirectory() as folder:
            root = Path(folder)
            img = root / CANONICAL_OK_SOURCE
            img.parent.mkdir(parents=True)
            img.write_bytes(b"operator-approved-canonical-OK")
            digest = sha256(img.read_bytes()).hexdigest()
            doc = root / mod.EVIDENCE_PATH
            doc.parent.mkdir(parents=True)
            payload = {
                "schema": "visionx.operator_confirmed_quality.v1",
                "reviewed_on": "2026-10-26",
                "approved_for_production": False,
                "labels": [
                    {"source_path": mod.RETIRED_SOURCE,
                     "source_sha256": mod.RETIRED_SOURCE_FILE_SHA256,
                     "quality_label": "OK"},
                    {"source_path": CANONICAL_OK_SOURCE, "source_sha256": digest,
                     "quality_label": "OK", "physical_type": "COMPONENT_PRESENT"},
                ],
            }
            doc.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(mod, "CANONICAL_OK_FILE_SHA256", digest):
                self.assertTrue(verified_canonical_ok_source(root))
                wrong = root / RETIRED_SOURCE
                wrong.parent.mkdir(parents=True)
                wrong.write_bytes(b"wrong-NG-is-present")
                self.assertFalse(verified_canonical_ok_source(root))
                wrong.unlink()
                img.write_bytes(b"tampered")
                self.assertFalse(verified_canonical_ok_source(root))
                img.write_bytes(b"operator-approved-canonical-OK")
                payload["labels"][1]["quality_label"] = "NG"
                doc.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(verified_canonical_ok_source(root))

    def test_knn_c_discards_wrong_ng_even_if_old_inventory_contains_it(self):
        wrong = {
            "path": RETIRED_SOURCE, "expected_label": "NG",
            "pixel_sha256": "a"*64, "file_sha256": "b"*64,
            "lighting_mode": "SIDE", "status": "VALID_PNG",
        }
        good = {
            "path": CANONICAL_OK_SOURCE, "expected_label": "OK",
            "pixel_sha256": CANONICAL_OK_PIXEL_SHA256,
            "file_sha256": "c"*64, "lighting_mode": "SIDE",
            "status": "VALID_PNG",
        }
        inventory = {
            "schema": "visionx.archive_inventory.v1",
            "images": [wrong, good],
            "summary": {"png_count": 2}, "cross_label_conflicts": [],
        }
        with TemporaryDirectory() as folder:
            bank = LiveKNNStrategyC(
                root=folder, inventory=inventory,
                extractor=lambda frame: (frame, frame, {}),
                frame_loader=lambda root, item: np.ones((40,40,3),dtype=np.uint8),
            )
            with patch("src.core.knn_c_live.build_focus_signature",
                       return_value={"focus": [1,2,3]}), patch(
                "src.core.knn_c_live.valid_focus_signature", return_value=True,
            ), patch("src.core.knn_c_live.build_geometry_signature",
                     return_value={"geometry": 1}):
                result = bank.warmup()
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["indexed_by_label"], {"OK": 1, "NG": 0})
        self.assertEqual(result["retired_wrong_ng_entries"], 1)
        self.assertEqual([e["source_path"] for e in bank._pools["SIDE"]],
                         [CANONICAL_OK_SOURCE])

    def test_supervised_recognition_and_auto_ok_only_with_canonical_source(self):
        a, visual = event()
        with patch(
            "src.core.operator_confirmed_archive_labels.verified_canonical_ok_source",
            return_value=True,
        ), patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=deepcopy(visual),
        ):
            actual = apply_knn_visual_corroboration(a, {}, "FALTANDO", object())
            self.assertEqual(actual["verdict"], "FALHA FALSA")
            info = actual["detail"]["operator_corrected_archive_ok_recall"]
            self.assertEqual(info["historical_source_path"], CANONICAL_OK_SOURCE)
            self.assertFalse(info["current_pair_exact_match"])
            self.assertFalse(production_decision_policy(actual)["auto_allowed"])
            actual["detail"]["production_continuous_auto"] = True
            policy = production_decision_policy(actual)
            self.assertEqual(policy["proposed_decision"], "OK")
            self.assertTrue(policy["auto_allowed"])
            self.assertEqual(policy["reason"],
                             "supervised_operator_corrected_archive_ok_auto_v1")
            self.assertEqual(verdict_feedback_state(actual), ("FALHA FALSA", "ok"))

    def test_side_review_variant_from_old_test_is_still_supported(self):
        # Avoid regressing events where SIDE is also in local review.
        a, visual = event(side_local_verdict="REVISÃO OBRIGATÓRIA")
        with patch(
            "src.core.operator_confirmed_archive_labels.verified_canonical_ok_source",
            return_value=True,
        ), patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=deepcopy(visual),
        ):
            actual = apply_knn_visual_corroboration(a, {}, "FALTANDO", object())
            self.assertEqual(actual["verdict"], "FALHA FALSA")
            actual["detail"]["production_continuous_auto"] = True
            self.assertEqual(production_decision_policy(actual)["proposed_decision"], "OK")

    def test_side_local_ng_or_untrusted_checkpoint_does_not_auto_approve(self):
        for local, score, allowed in (
            ("DEFEITO REAL", .0343675725, False),
            ("FALHA FALSA", .97, False),
            ("FALHA FALSA", .0343675725, True),
        ):
            with self.subTest(local=local, score=score):
                a, visual = event(side_local_verdict=local)
                a["detail"]["cnn_v2_light_diagnostics"]["SIDE"][
                    "ng_score_uncalibrated"
                ] = score
                with patch(
                    "src.core.operator_confirmed_archive_labels.verified_canonical_ok_source",
                    return_value=True,
                ), patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=deepcopy(visual),
                ):
                    actual = apply_knn_visual_corroboration(
                        a, {}, "FALTANDO", object(),
                    )
                    actual["detail"]["production_continuous_auto"] = True
                    policy = production_decision_policy(actual)
                    self.assertEqual(
                        policy["proposed_decision"] == "OK", allowed,
                    )

    def test_forged_source_ng_or_physical_defect_blocks_auto_ok(self):
        variants = (
            ("wrong_path", lambda a,r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][0].update(path=RETIRED_SOURCE)),
            ("wrong_pixel_sha",lambda a,r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_B"][0].update(source_pixel_sha256="0"*64)),
            ("wrong_label",lambda a,r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][0].update(label="NG")),
            ("strong_side_ng",lambda a,r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][1].update(similarity=.95)),
            ("physical_missing",lambda a,r:a["detail"].update(physical_score=.7)),
            ("strong_ng_lights",lambda a,r:r.update(strong_ng_lights=["SIDE"])),
            ("side_margin",lambda a,r:r["lights"]["SIDE"].update(
                ng={"best_similarity":1.0},
                ng_margin_over_ok=1.0-.977865)),
            ("top_vote_ng",lambda a,r:
             a["detail"]["knn_c_light_observer_votes"]["TOP"].update(vote_B="NG")),
        )
        for name, change in variants:
            with self.subTest(name=name):
                a, visual = event()
                change(a, visual)
                with patch(
                    "src.core.operator_confirmed_archive_labels.verified_canonical_ok_source",
                    return_value=True,
                ), patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=visual,
                ):
                    actual=apply_knn_visual_corroboration(a,{}, "FALTANDO",object())
                self.assertNotEqual(actual["verdict"], "FALHA FALSA")
        a,visual=event()
        with patch(
            "src.core.operator_confirmed_archive_labels.verified_canonical_ok_source",
            return_value=False,
        ), patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=visual,
        ):
            actual=apply_knn_visual_corroboration(a,{},"FALTANDO",object())
        self.assertNotEqual(actual["verdict"], "FALHA FALSA")


if __name__ == "__main__":
    unittest.main()
