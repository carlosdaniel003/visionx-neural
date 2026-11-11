"""Identidade de tela XP no acervo e encerramento genérico SIDE V3."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from src.core.archive_frame_identity import (
    corroborated_archive_exact_side, verified_archive_exact_frame,
)
from src.core.knn_c_live import LiveKNNStrategyC
from src.core.side_early_decision import build_side_early_decision
from src.services.image_archive_dedup import image_fingerprint
from tests.test_side_early_decision import _v3_side


class ArchiveFrameIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "public" / "ok_archive"
        folder.mkdir(parents=True)
        self.frame = np.full((240, 280, 3), 40, dtype=np.uint8)
        cv2.rectangle(self.frame, (55, 60), (190, 170), (230, 30, 60), -1)
        self.relative = "public/ok_archive/2026-11-11_0800_AOI_SIDE.png"
        self.path = self.root / self.relative
        self.assertTrue(cv2.imwrite(str(self.path), self.frame))
        self.pixel_sha = image_fingerprint(self.frame)
        self.file_sha = sha256(self.path.read_bytes()).hexdigest()
        item = {
            "path": self.relative, "status": "VALID_PNG",
            "expected_label": "OK", "lighting_mode": "SIDE",
            "pixel_sha256": self.pixel_sha, "file_sha256": self.file_sha,
        }
        def extractor(_source):
            ref = np.full((64, 64, 3), 40, dtype=np.uint8)
            test = ref.copy()
            cv2.rectangle(test, (5, 6), (22, 28), (100, 100, 100), -1)
            return ref, test, {"category": "UNKNOWN"}
        self.bank = LiveKNNStrategyC(
            root=self.root,
            inventory={
                "schema": "visionx.archive_inventory.v1",
                "images": [item], "summary": {"png_count": 1},
                "cross_label_conflicts": [],
            },
            extractor=extractor,
            frame_loader=lambda _root, _item: "OK",
        )
        self.assertEqual(self.bank.warmup()["status"], "READY")

    def proof(self):
        candidate = self.bank.lookup_exact_frame(self.frame.copy(), "SIDE")
        self.assertEqual(candidate["status"], "EXACT_FRAME")
        return candidate

    def test_exact_frame_uses_full_pixels_not_knn_score(self):
        proof = self.proof()
        self.assertTrue(verified_archive_exact_frame(proof))
        self.assertEqual(proof["source_path"], self.relative)
        modified = self.frame.copy()
        modified[100, 100, 1] ^= 1
        self.assertEqual(self.bank.lookup_exact_frame(modified, "SIDE")["status"],
                         "NO_EXACT_FRAME")
        self.assertEqual(self.bank.lookup_exact_frame(self.frame, "TOP")["status"],
                         "NO_EXACT_FRAME")

    def test_deleted_or_modified_source_is_not_proof(self):
        proof = self.proof()
        modified = self.frame.copy()
        modified[100, 100, 1] ^= 1
        self.assertTrue(cv2.imwrite(str(self.path), modified))
        self.assertFalse(verified_archive_exact_frame(proof))
        self.assertEqual(self.bank.lookup_exact_frame(self.frame, "SIDE")["status"],
                         "UNAVAILABLE")

    def test_mixed_knn_votes_do_not_erase_exact_same_source(self):
        proof = self.proof()
        side = _v3_side(score=8.112491741485428e-06)
        knn = side["detail"]["knn_c"]
        knn.update(status="REVISAO", reason="DISCORDANCIA_KNN_A_B",
                   vote_A="NG", vote_B="OK",
                   suggestion_A="REVISAO", suggestion_B="REVISAO",
                   production_approved=False)
        for key in ("neighbors_A", "neighbors_B"):
            knn[key][0].update(label="OK",
                               source_pixel_sha256=self.pixel_sha,
                               path=self.relative)
        proof["physical_probe_status"] = "NO_STRONG_EVIDENCE"
        side["detail"]["archive_exact_frame"] = proof
        self.assertTrue(corroborated_archive_exact_side(side))
        final = build_side_early_decision(
            side, "MUITO ADESIVO", allow_supervised_v3=True,
        )
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertEqual(final["detail"]["side_early_exit"]["analyzed_lights"],
                         ["SIDE"])
        self.assertEqual(final["detail"]["side_early_exit"]["corroboration"],
                         "ARCHIVE_EXACT_FRAME_V3_AGREEMENT")
        self.assertIsNone(build_side_early_decision(side, "MUITO ADESIVO"))

        for change in (
            lambda d: d["archive_exact_frame"].update(physical_probe_status="UNAVAILABLE"),
            lambda d: d["archive_exact_frame"].update(frame_sha256="0" * 64),
            lambda d: d["knn_c"]["neighbors_A"][0].update(label="NG"),
            lambda d: d["knn_c"]["neighbors_B"][0].update(source_pixel_sha256="1" * 64),
            lambda d: d.update(confirmed_visual_recall={"status": "HUMAN_NG_VISUAL_CANDIDATE"}),
        ):
            bad = deepcopy(side)
            change(bad["detail"])
            self.assertIsNone(build_side_early_decision(
                bad, "MUITO ADESIVO", allow_supervised_v3=True,
            ))

    def test_opposite_archive_label_never_changes_cnn_verdict(self):
        proof = self.proof()
        proof["physical_probe_status"] = "NO_STRONG_EVIDENCE"
        side = _v3_side(verdict="DEFEITO REAL", score=0.99999)
        side["detail"]["archive_exact_frame"] = proof
        self.assertFalse(corroborated_archive_exact_side(side))


    def test_production_gate_revalidates_exact_archive_before_xp(self):
        from src.ui.production_confidence_gate import production_decision_policy

        proof = self.proof()
        proof["physical_probe_status"] = "NO_STRONG_EVIDENCE"
        side = _v3_side(score=8.112491741485428e-06)
        knn = side["detail"]["knn_c"]
        knn.update(
            status="REVISAO", reason="DISCORDANCIA_KNN_A_B",
            vote_A="NG", vote_B="OK",
            suggestion_A="REVISAO", suggestion_B="REVISAO",
            production_approved=False,
        )
        for key in ("neighbors_A", "neighbors_B"):
            knn[key][0].update(
                label="OK", source_pixel_sha256=self.pixel_sha,
                path=self.relative,
            )
        side["detail"]["archive_exact_frame"] = proof
        final = build_side_early_decision(
            side, "UNKNOWN", allow_supervised_v3=True,
        )
        self.assertIsNotNone(final)
        final["detail"]["production_continuous_auto"] = True
        final["detail"]["production_binary_auto"] = True
        decision = production_decision_policy(final)
        self.assertTrue(decision["auto_allowed"])
        self.assertEqual(decision["proposed_decision"], "OK")
        self.assertEqual(decision["reason"], "production_supervised_v3_side_visual_ok")

        # Nenhuma etiqueta provisória autoriza o XP após mudar a fonte.
        changed = self.frame.copy()
        changed[100, 100, 0] ^= 1
        self.assertTrue(cv2.imwrite(str(self.path), changed))
        stale = production_decision_policy(final)
        self.assertFalse(stale["auto_allowed"])
        self.assertNotEqual(stale["proposed_decision"], "OK")

    def test_archive_conflict_cannot_be_reported_as_exact_ok(self):
        self.bank._exact_conflicts = frozenset({self.pixel_sha})
        result = self.bank.lookup_exact_frame(self.frame, "SIDE")
        self.assertEqual(result["status"], "EXACT_FRAME_CONFLICT")
        self.assertEqual(result["label"], "")

if __name__ == "__main__":
    unittest.main()
