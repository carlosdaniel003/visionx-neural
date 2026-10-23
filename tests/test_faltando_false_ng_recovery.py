"""Plano de exemplos CNN FALTANDO SIDE e proteção NG: zero mutação."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import tempfile
from pathlib import Path
import unittest

from src.scripts.faltando_false_ng_recovery import recovery_plan
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    hard_ok_recovery_gate,
)


def row(path, label, cnn, light="SIDE", *, ok_neighbor=False):
    return {
        "source_path": path, "source_file_sha256": "0"*64,
        "expected_label": label, "category": "FALTANDO", "lighting": light,
        "cnn": {"status": "PASSOU" if label == cnn else "ERRO_CLASSIFICACAO",
                "label": cnn, "ng_score_uncalibrated": .99985,
                "checkpoint_sha256":
                  "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"},
        "knn_visual": {
            "suggested_label": None, "neighbors": [{
                "source_path": "public/ok_archive/similar.png" if ok_neighbor else
                               "public/ng_archive/similar.png",
                "label": "OK" if ok_neighbor else "NG",
                "similarity": .98231,
            }],
        },
    }


def report(rows):
    return {
        "schema": "visionx.knn_cnn_arbitration_shadow.v1",
        "mode": "OFFLINE_PNG_KNN_CNN_AB_ONLY", "complete": True,
        "production_approved": False, "xp_command_authorized": False,
        "production_route_reproduced": False,
        "archive_pngs": len(rows), "rows": rows,
        "evidence_contract": {
            "original_png_hashes_verified_between_reports": True,
            "archives_same_files": True, "independent_event_holdout": False,
        },
    }


def assessment(ok_wrong=0, ng_wrong=0, side_ok_wrong=None):
    if side_ok_wrong is None:
        side_ok_wrong = ok_wrong
    cases = []
    for i in range(2):
        bad = i < ok_wrong
        side_bad = i < side_ok_wrong
        cases.append({
            "label": "OK", "correct": not bad,
            "per_light": {
                "SIDE": {"correct": not side_bad},
                "TOP": {"correct": not bad},
            },
        })
    for i in range(2):
        bad = i < ng_wrong
        cases.append({
            "label": "NG", "correct": not bad,
            "per_light": {"SIDE": {"correct": not bad}},
        })
    return {
        "cases": cases, "ng_false_ok": ng_wrong, "all_correct": False,
        "correct_images": 8 - 2 * ok_wrong - ng_wrong,
    }


class HardOKRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"public/ok_archive").mkdir(parents=True)
        (self.root/"public/ng_archive").mkdir(parents=True)

    def test_real_case_shape_side_false_ng_and_hard_ng_control(self):
        data = report([
            row("public/ok_archive/confirmed_ok.png", "OK", "NG",
                ok_neighbor=True),
            row("public/ng_archive/confirmed_ng.png", "NG", "NG",
                ok_neighbor=True),
            row("public/ok_archive/top.png", "OK", "NG", "TOP"),
        ])
        before = deepcopy(data)
        res = recovery_plan(data, self.root)
        self.assertEqual(res["false_ng"]["by_light"], {"SIDE": 1, "TOP": 1})
        self.assertEqual(res["false_ng"]["prioritized_cases"][0]["light"], "SIDE")
        self.assertEqual(res["protected_ng"]["count"], 1)
        self.assertTrue(res["protected_ng"]["prioritized_cases"][0]["hard_negative"])
        self.assertFalse(res["auto_ok_authorized"])
        self.assertFalse(res["candidate_checkpoint_generated"])
        self.assertTrue(res["provisional_labels_not_human_verified"])
        self.assertEqual(data, before)

    def test_hash_check_refuses_stale_report_and_requires_present_pngs(self):
        path = self.root/"public/ok_archive/ok.png"
        path.write_bytes(b"original bytes")
        n = row("public/ok_archive/ok.png", "OK", "NG")
        n["source_file_sha256"] = sha256(path.read_bytes()).hexdigest()
        ctrl = row("public/ng_archive/ng.png", "NG", "NG")
        file_ng = self.root/"public/ng_archive/ng.png"
        file_ng.write_bytes(b"other bytes")
        ctrl["source_file_sha256"] = sha256(file_ng.read_bytes()).hexdigest()
        data = report([n, ctrl])
        self.assertTrue(recovery_plan(data, self.root, verify_files=True)[
            "source_files_verified"])
        path.write_bytes(b"changed but same size")
        with self.assertRaisesRegex(ValueError, "mudou"):
            recovery_plan(data, self.root, verify_files=True)

    def test_rejects_bad_report_unsafe_path_missing_hash_or_unverified_protocol(self):
        src = report([
            row("public/ok_archive/a.png", "OK", "NG"),
            row("public/ng_archive/b.png", "NG", "NG"),
        ])
        for field, val in (("production_approved", True),
                           ("production_route_reproduced", True),
                           ("complete", False), ("archive_pngs", 900)):
            trial = deepcopy(src)
            trial[field] = val
            with self.subTest(field=field), self.assertRaises(ValueError):
                recovery_plan(trial, self.root)
        for corrupted in ("../../secret.png", "public/ok_archive/../ng_archive/x.png"):
            trial = deepcopy(src)
            trial["rows"][0]["source_path"] = corrupted
            with self.subTest(path=corrupted), self.assertRaises(ValueError):
                recovery_plan(trial, self.root)
        trial = deepcopy(src)
        trial["rows"][0]["source_file_sha256"] = ""
        with self.assertRaises(ValueError):
            recovery_plan(trial, self.root)

    def test_qualified_balanced_gate_passes_only_with_fewer_side_false_ng_zero_ng_ok(self):
        old_hist = assessment(ok_wrong=0)
        old_online = assessment(ok_wrong=0)
        old_hold = assessment(ok_wrong=2)
        new_hold = assessment(ok_wrong=1)
        outcome = hard_ok_recovery_gate(
            old_hist, old_hist, old_online, old_online,
            old_hold, new_hold, individually_qualified=True,
        )
        self.assertTrue(outcome["passed"], outcome)
        self.assertEqual(outcome["old_holdout_side_false_ng"], 2)
        self.assertEqual(outcome["new_holdout_side_false_ng"], 1)
        self.assertFalse(outcome["auto_ok_authorized"])
        self.assertFalse(outcome["production_approval_eligible"])

    def test_gate_blocks_one_ng_to_ok_even_when_side_false_ng_improves(self):
        good = assessment(ok_wrong=0)
        old = assessment(ok_wrong=2)
        better_but_unsafe = assessment(ok_wrong=1, ng_wrong=1)
        result = hard_ok_recovery_gate(
            good, good, good, good, old, better_but_unsafe,
            individually_qualified=True,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["candidate_no_ng_as_ok_holdout"])
        self.assertFalse(result["gates"]["candidate_no_ng_light_regressions_holdout"])

    def test_gate_rejects_unconfirmed_folder_labels(self):
        a = assessment(ok_wrong=0)
        b = assessment(ok_wrong=2)
        c = assessment(ok_wrong=1)
        self.assertFalse(hard_ok_recovery_gate(
            a,a,a,a,b,c,individually_qualified=False,
        )["passed"])


if __name__ == "__main__":
    unittest.main()
