"""Regressões para os 25 NG físicos da auditoria V3, estritas por SHA.

Não abre XP, não cria checkpoints e não transforma categoria AOI
em diagnóstico físico.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.services.aoi_v3_ng_escape_reviews import (
    REVIEW_PATH, load_v3_ng_escape_reviews,
)


class StubStore:
    def __init__(self, root, entries):
        self.root = root
        self.by_path = {}
        self.validated = []
        for item in entries:
            relative = item["source_path"]
            self.by_path[relative] = {
                "status": "EXTRACTED_PENDING_REVIEW",
                "source_sha256": item["source_sha256"],
            }

    def _validate_source(self, source):
        self.validated.append(source)


class ReviewTests(unittest.TestCase):
    def test_committed_review_is_all_25_and_excludes_deleted_case(self):
        doc = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        self.assertEqual(doc["source_audit_totals"]["NG_confirmed"], 25)
        self.assertEqual(len(doc["entries"]), 25)
        self.assertEqual(
            {row["case_number"] for row in doc["entries"]},
            set(range(2, 27)),
        )
        self.assertTrue(all(
            row["quality_label"] == "NG"
            and row["source_path"].startswith("public/ng_archive/")
            for row in doc["entries"]
        ))
        self.assertNotIn(
            doc["deleted_excluded_source"],
            {row["source_path"] for row in doc["entries"]},
        )
        self.assertEqual(
            next(row["source_sha256"] for row in doc["entries"]
                 if row["source_path"].endswith("2026-10-30_1359_FALTANDO.png")),
            "6641bce61ce53594eef548e07fbad22b6c9af68534ac49e00cdf0bf7d1d57a2c",
        )
        # A imagem UNKNOWN NÃO foi transformada em rótulo CNN por
        # categoria AOI; qualidade NG e tipo observado são separados.
        self.assertEqual(
            next(row["physical_observation"] for row in doc["entries"]
                 if row["source_path"].endswith("2026-10-21_1446_UNKNOWN.png")),
            "DARK_ADHESIVE_ON_PAD",
        )

    def _fixture(self, root):
        document = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        for item in document["entries"]:
            target = root / item["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = ("original:" + item["source_path"]).encode("utf-8")
            target.write_bytes(payload)
            item["source_sha256"] = sha256(payload).hexdigest()
        destination = root / REVIEW_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return document, StubStore(root, document["entries"])

    def test_loader_checks_all_original_hashes_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            reviews, manifest_sha = load_v3_ng_escape_reviews(store)
            self.assertEqual(len(reviews), 25)
            self.assertEqual(len(store.validated), 25)
            self.assertEqual(
                manifest_sha, sha256((root/REVIEW_PATH).read_bytes()).hexdigest()
            )
            self.assertEqual(
                {row["source_sha256"] for row in reviews.values()},
                {row["source_sha256"] for row in doc["entries"]},
            )

    def test_modified_source_and_stale_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            case = doc["entries"][0]
            (root / case["source_path"]).write_bytes(b"not the same image")
            with self.assertRaisesRegex(ValueError, "modificado"):
                load_v3_ng_escape_reviews(store)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            store.by_path[doc["entries"][3]["source_path"]]["source_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "manif esto".replace(" ", "")):
                load_v3_ng_escape_reviews(store)

    def test_missing_or_conflicting_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            doc["entries"][0]["quality_label"] = "OK"
            (root / REVIEW_PATH).write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rótulo"):
                load_v3_ng_escape_reviews(store)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            doc["entries"][5]["source_sha256"] = "f"*64
            (root / REVIEW_PATH).write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manif esto".replace(" ", "")):
                load_v3_ng_escape_reviews(store)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            del store.by_path[doc["entries"][9]["source_path"]]
            with self.assertRaisesRegex(ValueError, "não está no manifesto"):
                load_v3_ng_escape_reviews(store)

    def test_review_may_not_import_symlink_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            doc, store = self._fixture(root)
            source = root / doc["entries"][0]["source_path"]
            backup = source.with_suffix(".bak")
            source.rename(backup)
            try:
                source.symlink_to(backup)
            except (OSError, NotImplementedError):
                self.skipTest("Sem suporte a symlink nesta plataforma")
            with self.assertRaisesRegex(ValueError, "symlink|modificado"):
                load_v3_ng_escape_reviews(store)


if __name__ == "__main__":
    unittest.main()
