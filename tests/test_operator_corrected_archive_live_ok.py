"""Regression for known 2026-10-21 AOI-NG / operator-OK label conflict.

Replays event ec843a9ae7f04b26ab0681eebec8a1f8 and verifies the
source-SHA override without treating approximate matches as exact pairs.
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
    CORRECTED_SOURCE, CORRECTED_SOURCE_PIXEL_SHA256,
    verified_operator_archive_override,
)
from src.core.knn_c_live import LiveKNNStrategyC
from src.core.knn_visual_corroboration import apply_knn_visual_corroboration
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from tests.test_knn_three_light_ok_counterevidence import fused, report


def event(side_label="NG"):
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
                "verdict": "REVISÃO OBRIGATÓRIA",
                "checkpoint_sha256": "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c",
                "ng_score_uncalibrated": score,
            } for light, score in (
                ("SIDE", .0343675725), ("TOP", .25600528717),
                ("MID", .85979676247),
            )
        },
    )
    side = d["knn_c_light_observer_votes"]["SIDE"]
    side.update(status="SUGERE_NG", reason="A_SELETIVA_E_VOTOS_A_B_CONCORDAM",
                vote_A="NG" if side_label == "NG" else "OK",
                vote_B="NG" if side_label == "NG" else "OK",
                suggestion_A="SUGERE_NG" if side_label == "NG" else "REVISAO",
                suggestion_B="REVISAO",
                similarity_A=.97620107, similarity_B=.96378327)
    for light in ("TOP", "MID"):
        d["knn_c_light_observer_votes"][light].update(
            status="REVISAO", vote_A="OK", vote_B="OK",
            suggestion_A="REVISAO", suggestion_B="REVISAO",
        )
    def n(label, similarity, path="", pixel=""):
        return {"label": label, "similarity": similarity,
                "path": path or f"public/{label.lower()}_archive/sample.png",
                "source_pixel_sha256": pixel or "d" * 64}
    side_a = [
        n(side_label, .97620107, CORRECTED_SOURCE, CORRECTED_SOURCE_PIXEL_SHA256),
        n("OK", .86618201), n("NG", .82163046),
        n("OK", .82143573), n("NG", .81505456),
    ]
    side_b = [
        n(side_label, .96378327, CORRECTED_SOURCE, CORRECTED_SOURCE_PIXEL_SHA256),
        n("OK", .83100362), n("OK", .78586923),
        n("OK", .78237802), n("OK", .77834246),
    ]
    def ok_group(ng_slot):
        return [n("NG" if i == ng_slot else "OK", .86-i*.015)
                for i in range(5)]
    d["knn_c_light_neighbor_evidence"] = {
        "SIDE": {"neighbors_A": side_a, "neighbors_B": side_b},
        "TOP": {"neighbors_A": ok_group(-1), "neighbors_B": ok_group(4)},
        "MID": {"neighbors_A": ok_group(1), "neighbors_B": ok_group(1)},
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


class OperatorCorrectedArchiveTests(unittest.TestCase):
    def test_human_label_override_verifies_evidence_json_and_real_png_bytes(self):
        import src.core.operator_confirmed_archive_labels as mod
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / mod.CORRECTED_SOURCE
            source.parent.mkdir(parents=True)
            source.write_bytes(b"operator-approved-image")
            digest = sha256(source.read_bytes()).hexdigest()
            manifest = root / mod.EVIDENCE_PATH
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({
                "schema": "visionx.operator_confirmed_quality.v1",
                "reviewed_on": "2026-10-26",
                "approved_for_production": False,
                "labels": [{
                    "source_path": mod.CORRECTED_SOURCE,
                    "source_sha256": digest, "quality_label": "OK",
                    "physical_type": "COMPONENT_PRESENT",
                }],
            }), encoding="utf-8")
            with patch.object(mod, "CORRECTED_SOURCE_FILE_SHA256", digest):
                self.assertEqual(
                    verified_operator_archive_override(root), "OK",
                )
                source.write_bytes(b"altered")
                self.assertIsNone(verified_operator_archive_override(root))
                source.write_bytes(b"operator-approved-image")
                record = json.loads(manifest.read_text())
                record["labels"][0]["quality_label"] = "NG"
                manifest.write_text(json.dumps(record))
                self.assertIsNone(verified_operator_archive_override(root))

    def test_knn_c_warmup_relabels_only_sha_pinned_source_not_folder(self):
        image = {
            "path": CORRECTED_SOURCE,
            "expected_label": "NG", "pixel_sha256": CORRECTED_SOURCE_PIXEL_SHA256,
            "file_sha256": "a" * 64, "lighting_mode": "SIDE",
            "status": "VALID_PNG",
        }
        inventory = {
            "schema": "visionx.archive_inventory.v1", "images": [image],
            "summary": {"png_count": 1}, "cross_label_conflicts": [],
        }
        with TemporaryDirectory() as folder:
            bank = LiveKNNStrategyC(
                root=folder, inventory=inventory,
                extractor=lambda frame: (frame, frame, {}),
                frame_loader=lambda root, item: np.ones((40, 40, 3), dtype=np.uint8),
            )
            with patch("src.core.knn_c_live.verified_operator_archive_override",
                       return_value="OK"), patch(
                "src.core.knn_c_live.build_focus_signature",
                return_value={"focus": [1, 2, 3]},
            ), patch(
                "src.core.knn_c_live.valid_focus_signature", return_value=True,
            ), patch(
                "src.core.knn_c_live.build_geometry_signature",
                return_value={"geo": 1},
            ):
                result = bank.warmup()
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["indexed_by_label"], {"OK": 1, "NG": 0})
            self.assertEqual(result["operator_corrected_archive_sources"], 1)
            entry = bank._pools["SIDE"][0]
            self.assertEqual(entry["expected_label"], "OK")
            self.assertEqual(entry["archive_folder_label"], "NG")

    def test_fusion_and_production_auto_ok_for_known_corrected_source(self):
        for label in ("NG", "OK"):
            with self.subTest(source_memory_label=label):
                a, report_data = event(label)
                with patch(
                    "src.core.operator_confirmed_archive_labels.verified_operator_archive_override",
                    return_value="OK",
                ), patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=deepcopy(report_data),
                ):
                    actual = apply_knn_visual_corroboration(
                        a, {}, "FALTANDO", object(),
                    )
                    self.assertEqual(actual["verdict"], "FALHA FALSA")
                    self.assertEqual(
                        actual["detail"]["fusion_rule"],
                        "operator_corrected_archive_ok_three_light_v1",
                    )
                    info = actual["detail"]["operator_corrected_archive_ok_recall"]
                    self.assertFalse(info["current_pair_exact_match"])
                    self.assertTrue(info["source_human_quality_verified"])
                    policy_manual = production_decision_policy(actual)
                    self.assertFalse(policy_manual["auto_allowed"])
                    actual["detail"]["production_continuous_auto"] = True
                    policy = production_decision_policy(actual)
                    self.assertEqual(policy["proposed_decision"], "OK")
                    self.assertTrue(policy["auto_allowed"])
                    self.assertEqual(
                        policy["reason"],
                        "supervised_operator_corrected_archive_ok_auto_v1",
                    )
                    self.assertEqual(verdict_feedback_state(actual),
                                     ("FALHA FALSA", "ok"))

    def test_disagreement_or_unverified_source_never_auto_approves(self):
        variants = (
            ("bad_source", lambda a, r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_A"][0].update(path="public/ok_archive/other.png")),
            ("wrong_sha", lambda a, r:
             a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"]
             ["neighbors_B"][0].update(source_pixel_sha256="0"*64)),
            ("physical_missing", lambda a, r:
             a["detail"].update(physical_score=.7)),
            ("strong_ng", lambda a, r: r.update(strong_ng_lights=["SIDE"])),
            ("side_ng_margin", lambda a, r: r["lights"]["SIDE"].update(
                ng={"best_similarity": 1.0},
                ng_margin_over_ok=1.0-.977865)),
            ("top_ng", lambda a, r:
             a["detail"]["knn_c_light_observer_votes"]["TOP"].update(vote_A="NG")),
            ("cnn_strong_ng", lambda a, r:
             a["detail"]["cnn_v2_light_diagnostics"]["SIDE"].update(
                 ng_score_uncalibrated=.99)),
        )
        for name, change in variants:
            with self.subTest(name=name):
                a, report_data = event()
                change(a, report_data)
                with patch(
                    "src.core.operator_confirmed_archive_labels.verified_operator_archive_override",
                    return_value="OK",
                ), patch(
                    "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
                    return_value=report_data,
                ):
                    result = apply_knn_visual_corroboration(
                        a, {}, "FALTANDO", object(),
                    )
                self.assertNotEqual(result["verdict"], "FALHA FALSA")
        a, report_data = event()
        with patch(
            "src.core.operator_confirmed_archive_labels.verified_operator_archive_override",
            return_value=None,
        ), patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=report_data,
        ):
            result = apply_knn_visual_corroboration(
                a, {}, "FALTANDO", object(),
            )
            self.assertNotEqual(result["verdict"], "FALHA FALSA")


if __name__ == "__main__":
    unittest.main()
