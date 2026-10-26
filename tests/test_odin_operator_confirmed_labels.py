"""Human-confirmed CNN quality labels must never silently become folder truth."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.services.odin_operator_confirmed_labels import (
    EVIDENCE, qualify_inspected_pairs, verified_operator_labels,
)


class FakeStore:
    def __init__(self, root, labels, *, missing_one=False, incompatible=False):
        self.root = root.resolve()
        chosen = [(src, row) for src, row in labels.items()
                  if row["pair_inspected_by_operator"]]
        if missing_one:
            chosen = chosen[:-1]
        self.by_path = {
            src: {"source_sha256": row["source_sha256"]}
            for src, row in chosen
        }
        self.case_reviews = {}
        self.output = root / "qualification.json"
        self.written = []
        self.incompatible = incompatible

    def _validate_source(self, source):
        if self.incompatible:
            raise ValueError("recorte AOI mudou")

    def case_status(self, source):
        return self.case_reviews.get(source, {}).get("status", "PENDING")

    def review_case(self, source, status, notes):
        self.written.append((source, status))
        self.case_reviews[source] = {
            "status": status, "pair_inspected": True,
        }


class OperatorQualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = []
        for i in range(15):
            label = "OK" if i in (6, 8, 9) else "NG"
            source = ("public/ng_archive/2026-10-21_1414_FALTANDO.png"
                      if i == 6 else
                      f"public/{'ok_archive' if i in (8, 9) else 'ng_archive'}/case_{i}.png")
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-pixels" + bytes([i]))
            self.records.append({
                "source_path": source,
                "source_sha256": sha256(path.read_bytes()).hexdigest(),
                "quality_label": label, "physical_type": "CHECKED",
                "pair_inspected_by_operator": i < 6,
            })
        doc = {
            "schema": "visionx.operator_confirmed_quality.v1",
            "approved_for_production": False,
            "physical_event_groups_verified": False,
            "reviewed_on": "2026-10-26",
            "labels": self.records,
        }
        dest = self.root / EVIDENCE
        dest.parent.mkdir(parents=True)
        dest.write_text(json.dumps(doc), encoding="utf-8")

    def test_fifteen_pinned_labels_include_corrected_case_seven(self):
        result = verified_operator_labels(self.root)
        self.assertEqual(Counter(r["quality_label"] for r in result.values()),
                         Counter({"NG": 12, "OK": 3}))
        self.assertEqual(result[
            "public/ng_archive/2026-10-21_1414_FALTANDO.png"
        ]["quality_label"], "OK")

    def test_changed_source_refuses_to_import_labels(self):
        (self.root / self.records[6]["source_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "divergente"):
            verified_operator_labels(self.root)

    def test_qualify_only_six_reviewed_pairs_without_group_claim(self):
        labels = verified_operator_labels(self.root)
        store = FakeStore(self.root, labels)
        with patch(
            "src.services.odin_operator_confirmed_labels.QualificationStore",
            return_value=store,
        ):
            result = qualify_inspected_pairs(
                self.root, Path("unused/manifest.json"), apply=True,
            )
        self.assertEqual(result["pair_reviews_written"], 6)
        self.assertEqual(result["images_without_qualified_pairs"], 9)
        self.assertEqual(result["case_7_corrected_quality"], "OK")
        self.assertEqual(result["group_reviews_created"], 0)
        self.assertEqual(len(store.written), 6)
        self.assertTrue(all(status == "CONFIRMED_NG" for _, status in store.written))

    def test_preflight_fails_without_six_pairs_and_writes_nothing(self):
        labels = verified_operator_labels(self.root)
        store = FakeStore(self.root, labels, missing_one=True)
        with patch(
            "src.services.odin_operator_confirmed_labels.QualificationStore",
            return_value=store,
        ):
            with self.assertRaisesRegex(ValueError, "seis pares"):
                qualify_inspected_pairs(
                    self.root, Path("unused/manifest.json"), apply=True,
                )
        self.assertEqual(store.written, [])

    def test_conflicting_existing_human_verdict_blocks_overwrite(self):
        labels = verified_operator_labels(self.root)
        store = FakeStore(self.root, labels)
        source = next(iter(store.by_path))
        store.case_reviews[source] = {
            "status": "CONFIRMED_OK", "pair_inspected": True,
        }
        with patch(
            "src.services.odin_operator_confirmed_labels.QualificationStore",
            return_value=store,
        ):
            with self.assertRaisesRegex(ValueError, "contraditória"):
                qualify_inspected_pairs(
                    self.root, Path("unused/manifest.json"), apply=True,
                )
        self.assertEqual(store.written, [])


if __name__ == "__main__":
    unittest.main()
