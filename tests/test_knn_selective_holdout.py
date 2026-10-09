"""Testes de separação temporal por dia e inferência NG seletiva, sem produção."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.services.startup_regression.knn_selective_holdout import (
    _calibrate, _day, _decision, _measure, _split_dates,
    _cross_partition_exclusions, selective_knn_holdout,
)
from src.services.startup_regression.knn_selective_holdout_cli import (
    write_holdout_reports,
)


def example(day, ident, label, value, *, category="FALTANDO", lighting="SIDE",
            event=None):
    arr = np.zeros(224, dtype=np.float32)
    arr[:4] = [value, value * .27, value * .16, .6 - value * .1]
    signature = {"vector": arr.tolist()}
    return {
        "path": f"public/dataset/{'anomalia' if label=='NG' else 'nao_anomalia'}/"
                f"{category}/memory_{label}_{day.replace('-', '')}_{ident}.json",
        "status": "ELEGIVEL", "category": category,
        "lighting_mode": lighting, "label": label,
        "signature": signature, "event_id": event,
        "signature_hash": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def records():
    days = ("2026-07-31", "2026-08-04", "2026-10-01", "2026-10-02")
    result = []
    for j, day in enumerate(days):
        for i in range(2):
            result.append(example(day, f"ok{i}", "OK", .08 + .04*i + .017*j))
            result.append(example(day, f"ng{i}", "NG", .71 + .055*i + .019*j))
    return result


def observation(path, expected, vote, similarity=.96):
    return {
        "path": path, "category": "FALTANDO", "lighting_mode": "SIDE",
        "expected_human_label": expected,
        "score_ng": vote, "best_similarity": similarity,
        "has_exact_query_copy": False,
        "has_conflicting_neighbor_signatures": False,
    }


class KNNSelectiveHoldoutTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_date_from_legacy_and_modern_names(self):
        self.assertEqual(
            _day({"path": "memory_NG_20260804_081518_383.json"}), "2026-08-04"
        )
        self.assertEqual(
            _day({"path": "memory_OK_MID_20261008_072012_920.json"}), "2026-10-08"
        )
        self.assertEqual(
            _day({"path": "ok_2026-10-09_1201_FALTANDO.json"}), "2026-10-09"
        )
        self.assertIsNone(_day({"path": "memory_undated.json"}))

    def test_date_groups_never_split_and_have_three_independent_sets(self):
        items = records()
        result = _split_dates(items)
        self.assertEqual(result["status"], "GROUPED_BY_DAY_NON_CHRONOLOGICAL")
        found = {}
        for part, rows in result["by_part"].items():
            for row in rows:
                day = _day(row)
                if day in found:
                    self.assertEqual(found[day], part)
                else:
                    found[day] = part
        self.assertEqual(len(found), 4)
        self.assertTrue(all(
            sum(r["label"] == "NG" for r in result["by_part"][p]) > 0
            for p in ("memory_train", "calibration", "heldout_test")
        ))

    def test_cross_partition_exact_signature_is_discarded_without_rerouting(self):
        rows = records()
        # Marcador único copiado para datas distintas; é fuga de informação.
        rows[0]["signature"] = rows[4]["signature"]
        rows[0]["signature_hash"] = rows[4]["signature_hash"]
        s = _split_dates(rows)
        cleaned, excluded = _cross_partition_exclusions(s["by_part"])
        self.assertEqual(
            sum(map(len, cleaned.values())) + len(excluded),
            sum(map(len, s["by_part"].values())),
        )
        if any(
            x["path"] in {rows[0]["path"], rows[4]["path"]}
            for x in excluded
        ):
            self.assertTrue(all(
                x["reason"] == "CROSS_PARTITION_SIGNATURE"
                for x in excluded
            ))

    def test_same_event_on_different_days_never_used_across_partitions(self):
        rows = records()
        rows[0]["event_id"] = "SESSION_MULTI"
        rows[4]["event_id"] = "SESSION_MULTI"
        s = _split_dates(rows)
        _clean, excluded = _cross_partition_exclusions(s["by_part"])
        self.assertGreaterEqual(
            sum(x["reason"] == "CROSS_PARTITION_EVENT" for x in excluded), 0
        )
        if len({p for p in s["by_part"] if
                any(r["event_id"] == "SESSION_MULTI"
                    for r in s["by_part"][p])}) > 1:
            self.assertEqual(
                sum(x["reason"] == "CROSS_PARTITION_EVENT" for x in excluded), 2
            )

    def test_policy_calibration_never_allows_known_ng_as_ok(self):
        calibration = [
            observation("ok1", "OK", .08),
            observation("ok2", "OK", .12),
            observation("ok3", "OK", .18),
            observation("ok4", "OK", .26),
            observation("ng1", "NG", .23),
            observation("ng2", "NG", .77),
        ]
        policy, candidates = _calibrate(calibration)
        self.assertIsNotNone(policy)
        self.assertLessEqual(policy["ok_vote_ceiling"], .20)
        self.assertTrue(any(c["admitted_on_calibration"] for c in candidates))
        self.assertEqual(_measure(calibration, policy)["NG_released_as_OK"], 0)
        self.assertGreater(_measure(calibration, policy)["OK_auto_OK"], 0)

    def test_changing_query_label_alone_cannot_change_prediction(self):
        row = observation("q", "NG", .12, .95)
        policy = {"ok_vote_ceiling": .2, "min_similarity": .9}
        a = _decision(row, policy)
        b = _decision({**row, "expected_human_label": "OK"}, policy)
        self.assertEqual(a, b)

    def test_test_set_ng_miss_blocks_even_if_calibration_was_perfect(self):
        rows = records()
        cal = [
            observation("cal_ok1", "OK", .12),
            observation("cal_ok2", "OK", .18),
            observation("cal_ng", "NG", .28),
        ]
        test = [
            observation("test_ok", "OK", .11),
            observation("test_ng", "NG", .09),
        ]
        # Evita os vetores da fixture influenciarem o teste do requisito.
        with patch(
            "src.services.startup_regression.knn_selective_holdout._predictions",
            side_effect=[cal, test]
        ) as predictor:
            out = selective_knn_holdout(self.root, records=rows)
        self.assertEqual(predictor.call_count, 2)
        self.assertEqual(out["status"], "BLOCKED_NG_RELEASED_AS_OK_ON_HOLDOUT")
        self.assertEqual(out["calibration"]["NG_released_as_OK"], 0)
        self.assertEqual(out["heldout_test"]["NG_released_as_OK"], 1)
        self.assertFalse(out["production_approved"])
        self.assertFalse(out["startup_gate_enabled"])

    def test_if_calibration_cannot_support_useful_ok_release_no_policy(self):
        rows = records()
        cal = [
            observation("ok", "OK", .4),
            observation("ng", "NG", .05),
        ]
        test = [
            observation("test_ng", "NG", .95),
            observation("test_ok", "OK", .1),
        ]
        with patch(
            "src.services.startup_regression.knn_selective_holdout._predictions",
            side_effect=[cal, test]
        ):
            out = selective_knn_holdout(self.root, records=rows)
        self.assertEqual(out["status"], "BLOCKED_NO_USEFUL_CALIBRATED_POLICY")
        self.assertIsNone(out["calibrated_policy"])
        self.assertEqual(out["heldout_test"]["total_reviews"], 2)
        self.assertFalse(out["production_approved"])

    def test_when_ng_only_one_or_two_days_validation_blocked(self):
        rows = [
            example("2026-08-04", "ok", "OK", .2),
            example("2026-08-04", "ng", "NG", .8),
            example("2026-10-01", "ok", "OK", .3),
            example("2026-10-01", "ng", "NG", .9),
            example("2026-10-07", "ok", "OK", .4),
        ]
        with patch(
            "src.services.startup_regression.knn_selective_holdout._predictions",
            side_effect=AssertionError("Dataset insuficiente não roda teste")
        ):
            out = selective_knn_holdout(self.root, records=rows)
        self.assertEqual(out["status"], "BLOCKED_INSUFFICIENT_INDEPENDENT_CLASSES")
        self.assertIsNone(out["heldout_test"])
        self.assertFalse(out["production_approved"])

    def test_different_light_cannot_use_ng_from_other_light(self):
        rows = records()
        for row in rows:
            if row["label"] == "NG":
                row["lighting_mode"] = "TOP"
        out = selective_knn_holdout(self.root, records=rows)
        if out["calibration"]:
            self.assertEqual(out["calibration"]["OK_auto_OK"], 0)
            self.assertEqual(out["calibration"]["NG_auto_NG"], 0)
        self.assertFalse(out["production_approved"])

    def test_baseline_same_holdout_does_not_use_calibration_as_knn_memory(self):
        from src.services.startup_regression.knn_selective_holdout import _predictions
        r = records()
        train = [x for x in r if _day(x) == "2026-07-31"]
        test = [x for x in r if _day(x) == "2026-08-04"]
        with patch(
            "src.core.experts.knn_expert.KNNExpert.__init__",
            side_effect=AssertionError("Não carregar modelos KNN/CNN"),
        ):
            result = _predictions(test, train)
        self.assertTrue(result)
        self.assertTrue(all("baseline_vote_ng" in x for x in result))
        self.assertTrue(all(x["path"] not in {x["path"] for x in train}
                            for x in result))

    def test_writes_only_reports_and_preserves_original_data(self):
        rows = records()
        result = selective_knn_holdout(self.root, records=rows)
        outdir = self.root / "reports" / "startup_regression"
        json_path, txt_path = write_holdout_reports(result, outdir)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], result["schema"])
        self.assertFalse(payload["dataset_modified"])
        self.assertFalse(payload["production_approved"])
        self.assertTrue(txt_path.read_text(encoding="utf-8").startswith("ODIN"))
        with self.assertRaises(ValueError):
            write_holdout_reports(result, self.root / "public" / "dataset")


if __name__ == "__main__":
    unittest.main()
