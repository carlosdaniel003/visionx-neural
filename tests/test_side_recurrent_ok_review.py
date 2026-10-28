"""Regression for single-SIDE early MANUAL review of operator-confirmed 14:16 OK."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.operator_confirmed_archive_labels import (
    CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256, RETIRED_SOURCE,
)
from src.core.side_recurrent_ok_review import (
    build_side_recurrent_ok_review, side_recurrent_review_valid,
)
from src.ui.production_confidence_gate import production_decision_policy


def side():
    def neighbor(label, similarity, path=None, pixel=None):
        return {
            "label": label, "similarity": similarity,
            "path": path or f"public/{label.lower()}_archive/example.png",
            "source_pixel_sha256": pixel or "f"*64,
        }
    source = lambda similarity: neighbor(
        "OK", similarity, CANONICAL_OK_SOURCE, CANONICAL_OK_PIXEL_SHA256
    )
    return {
        "lighting_mode": "SIDE", "verdict": "FALHA FALSA",
        "is_defect": False, "confidence": .96563242748,
        "production_review_required": True,
        "detail": {
            "recognition_route": "NEW_CNN",
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_status": "INFERENCE_OK",
            "cnn_v2_checkpoint_verified": True,
            "cnn_v2_trial_manual_only": True,
            "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
            "cnn_v2_aoi_category": "FALTANDO",
            "cnn_v2_lighting_mode": "SIDE",
            "cnn_v2_ng_score_uncalibrated": .034367572516,
            "hard_missing_evidence": False, "physical_score": 0.0,
            "operator_review_required": True,
            "decision_trace": {"operator_review_required": True},
            "knn_c": {
                "schema": "visionx.knn_c_live_review_gated.v1",
                "bank_status": "READY", "diagnostic_only": True,
                "auto_decision_authorized": False,
                "status": "REVISAO", "vote_A": "NG", "vote_B": "OK",
                "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
                "neighbors_A": [
                    source(.86618201),
                    neighbor("NG", .82163046),
                    neighbor("OK", .82143573),
                    neighbor("NG", .81505456),
                    neighbor("NG", .81477241),
                ],
                "neighbors_B": [
                    source(.83100362),
                    neighbor("OK", .78586923),
                    neighbor("OK", .78237802),
                    neighbor("OK", .77834246),
                    neighbor("OK", .77572797),
                ],
            },
        },
    }


def audit(status="GEOMETRY_REVIEW"):
    return {
        "status": status,
        "reason": "TWO_SCALE_ROI_CHECK",
        "scales": [
            {"focus_fraction": .58, "suspected_absence": False},
            {"focus_fraction": .76, "suspected_absence": False},
        ],
        "geometry": {
            "method": "center_trim_no_rescale" if status == "GEOMETRY_REVIEW"
                      else "exact",
            "requires_review": status == "GEOMETRY_REVIEW",
        },
    }


class SideRecurrentOkReviewTests(unittest.TestCase):
    def setUp(self):
        self.source = side()
        self.ref = np.zeros((318, 570, 3), dtype=np.uint8)
        self.test = np.zeros((317, 567, 3), dtype=np.uint8)
        self.verified = patch(
            "src.core.side_recurrent_ok_review.verified_canonical_ok_source",
            return_value=True,
        )
        self.verified.start()
        self.addCleanup(self.verified.stop)

    def build(self, analysis=None, result=None):
        return build_side_recurrent_ok_review(
            analysis if analysis is not None else self.source,
            "FALTANDO", self.ref, self.test,
            probe=lambda ref, test: audit() if result is None else result,
        )

    def test_repeated_side_reference_skips_lights_but_never_autosends(self):
        final = self.build()
        self.assertIsNotNone(final)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["side_early_exit"]["skipped_lights"],
                         ["TOP", "MID"])
        self.assertFalse(final["eligible_for_final_decision"])
        self.assertTrue(side_recurrent_review_valid(final))
        final["detail"]["production_continuous_auto"] = True
        p = production_decision_policy(final)
        self.assertEqual(p["verdict"], "FALHA FALSA")
        self.assertFalse(p["auto_allowed"])
        self.assertTrue(p["operator_review_required"])
        self.assertEqual(p["proposed_decision"], "")
        self.assertEqual(p["reason"],
                         "side_verified_archive_visual_requires_operator")
        self.assertEqual(final["detail"]["side_recurrent_visual_review"][
            "source"], CANONICAL_OK_SOURCE)
        self.assertFalse(final["detail"]["side_recurrent_visual_review"][
            "live_pair_exact"])

    def test_different_ng_source_or_similar_ng_blocks_shortcut(self):
        changes = [
            lambda a: a["detail"]["knn_c"]["neighbors_A"][0].update(
                path=RETIRED_SOURCE
            ),
            lambda a: a["detail"]["knn_c"]["neighbors_B"][0].update(
                source_pixel_sha256="0"*64
            ),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(
                similarity=.9
            ),
            lambda a: a["detail"]["knn_c"]["neighbors_A"][1].update(
                similarity=.851
            ),
            lambda a: a["detail"]["knn_c"].update(bank_status="LOADING"),
            lambda a: a["detail"].update(cnn_v2_ng_score_uncalibrated=.15),
            lambda a: a["detail"].update(hard_missing_evidence=True),
            lambda a: a["detail"].update(recognition_match="EXACT_PAIR"),
            lambda a: a["detail"].update(cnn_v2_checkpoint_verified=False),
            lambda a: a["detail"].update(operator_review_required=False),
        ]
        for change in changes:
            candidate = deepcopy(self.source)
            change(candidate)
            with self.subTest(change=str(change)):
                self.assertIsNone(self.build(candidate))

    def test_no_source_identity_no_shortcut(self):
        with patch(
            "src.core.side_recurrent_ok_review.verified_canonical_ok_source",
            return_value=False,
        ):
            self.assertIsNone(self.build())

    def test_suspect_or_unavailable_probe_never_early_exits(self):
        for status in ("SUSPECT_ABSENCE", "UNAVAILABLE"):
            with self.subTest(status=status):
                bad = audit()
                bad["status"] = status
                self.assertIsNone(self.build(result=bad))

    def test_physical_probe_must_prove_both_scales(self):
        variants = [
            {"status": "GEOMETRY_REVIEW"},
            {**audit(), "scales": []},
            {**audit(), "scales": [
                {"suspected_absence": False}, {"suspected_absence": True}
            ]},
            {**audit(), "geometry": {"requires_review": False,
                                    "method": "exact"}},
        ]
        for item in variants:
            with self.subTest(item=item):
                self.assertIsNone(self.build(result=item))

    def test_forged_marker_cannot_change_production_to_auto_ok(self):
        final = self.build()
        for mutation in (
            lambda a: a["detail"]["side_early_exit"].update(
                production_auto_allowed=True
            ),
            lambda a: a["detail"]["side_early_exit"].update(
                skipped_lights=[]
            ),
            lambda a: a["detail"]["side_recurrent_visual_review"].update(
                source="public/ok_archive/wrong.png"
            ),
        ):
            with self.subTest(mutation=str(mutation)):
                forged = deepcopy(final)
                mutation(forged)
                self.assertFalse(side_recurrent_review_valid(forged))
                self.assertNotEqual(production_decision_policy(forged)[
                    "proposed_decision"], "OK")


if __name__ == "__main__":
    unittest.main()
