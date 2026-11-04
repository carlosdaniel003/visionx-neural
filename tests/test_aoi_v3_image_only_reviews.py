"""Regressões: 15 julgamentos humanos e a V3 sem categoria AOI."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3
from src.services.aoi_v3_operator_image_reviews import (
    REVIEW_PATH, load_v3_image_reviews,
)
from src.services.aoi_v3_binary_quality_metrics import (
    binary_quality_metrics, quality_coverage,
)
from src.services.aoi_v3_verified_candidate import (
    load_verified_v3_candidate,
)


class FakeManifestStore:
    def __init__(self, root: Path, document: dict):
        self.root = root
        self.by_path = {}
        self.files = {}
        self.validated = []
        for item in document["entries"]:
            source = item["source_path"]
            self.by_path[source] = {
                "status": "EXTRACTED_PENDING_REVIEW",
                "source_sha256": item["source_sha256"],
                "category_hint": "COMPLETAMENTE_INCORRETA",
            }
            self.files[source] = (root/source, None, None)

    def _validate_source(self, source):
        # Em Produção esta etapa usa QualificationStore._validate_source,
        # que reabre a captura e confere SHA da imagem no disco.
        self.validated.append(source)


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        (self.root/"docs").mkdir()
        self.path = self.root/REVIEW_PATH
        self.write_doc()
        self.store = FakeManifestStore(self.root, self.doc)

    def write_doc(self):
        self.path.write_text(
            json.dumps(self.doc, ensure_ascii=False), encoding="utf-8",
        )

    def test_reuses_fourteen_by_sha_without_reading_aoi_category(self):
        found = load_v3_image_reviews(self.store)
        self.assertEqual(len(found), 15)
        self.assertEqual(len(self.store.validated), 14)
        self.assertEqual(
            sorted(row["quality_label"] for row in found.values()).count("NG"),
            9,
        )
        self.assertEqual(
            sorted(row["quality_label"] for row in found.values()).count("OK"),
            5,
        )
        self.assertEqual(
            sorted(row["quality_label"] for row in found.values()).count("IGNORE"),
            1,
        )
        self.assertNotIn("category_hint", next(iter(found.values())))

    def test_explicit_ignore_still_works_after_user_deletes_old_png(self):
        ignored = next(
            x for x in self.doc["entries"] if x["quality_label"] == "IGNORE"
        )
        # O manifesto antigo ainda cita a imagem, mas não há PNG.
        self.assertFalse((self.root/ignored["source_path"]).exists())
        found = load_v3_image_reviews(self.store)
        self.assertEqual(found[ignored["source_path"]]["quality_label"], "IGNORE")
        self.store.by_path.pop(ignored["source_path"])
        self.store.files.pop(ignored["source_path"])
        found_again = load_v3_image_reviews(self.store)
        self.assertEqual(len(found_again), 15)

    def test_mutated_shas_and_operator_truth_are_rejected(self):
        for mutate in (
            lambda d: d["entries"][0].update(source_sha256="f"*64),
            lambda d: d["entries"][0].update(quality_label="NG"),
            lambda d: d["entries"][0].update(physical_observation="UNCERTAIN"),
            lambda d: d["entries"][1].update(source_path=d["entries"][0]["source_path"]),
            lambda d: d["entries"].pop(),
            lambda d: d.update(approved_for_production=True),
        ):
            with self.subTest(mutation=str(mutate)):
                self.doc = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
                mutate(self.doc)
                self.write_doc()
                with self.assertRaises(ValueError):
                    load_v3_image_reviews(self.store)

    def test_missing_confirmed_png_blocks_instead_of_silently_ignoring(self):
        source = self.doc["entries"][0]["source_path"]
        self.store.by_path.pop(source)
        with self.assertRaisesRegex(ValueError, "confirmado ausente"):
            load_v3_image_reviews(self.store)


class BinaryOnlyMetricTests(unittest.TestCase):
    def test_metrics_without_category_and_for_three_lights(self):
        records = [
            {"id": "a", "label": "NG", "views": {"SIDE": {}, "TOP": {}}},
            {"id": "b", "label": "OK", "views": {"MID": {}}},
            {"id": "c", "label": "NG", "views": {"SIDE": {}}},
        ]
        data = {"cases": [
            {"event_id_or_group_hint": "a", "label": "NG", "predicted": "OK",
             "per_light": {
                 "SIDE": {"score": .2, "correct": False},
                 "TOP": {"score": .3, "correct": False},
             }},
            {"event_id_or_group_hint": "b", "label": "OK", "predicted": "OK",
             "per_light": {"MID": {"score": .05, "correct": True}}},
            {"event_id_or_group_hint": "c", "label": "NG", "predicted": "NG",
             "per_light": {"SIDE": {"score": .9, "correct": True}}},
        ]}
        metrics = binary_quality_metrics(records, data)
        self.assertEqual(metrics["NG_false_OK"], 1)
        self.assertEqual(metrics["OK_false_NG"], 0)
        self.assertEqual(metrics["by_lighting"]["SIDE"]["NG_correct"], 1)
        self.assertNotIn("by_aoi_category", metrics)
        self.assertFalse(metrics["aoi_category_used_for_metrics"])
        with self.assertRaisesRegex(ValueError, "correspondência|divergente"):
            binary_quality_metrics(records, {"cases": data["cases"][:-1]})

    def test_coverage_counts_only_quality_and_illumination(self):
        rows = [
            {"label": "OK", "label_provenance": "HUMAN_V3_IMAGE_SHA256",
             "views": {"MID": {}}, "category_hint": "INCORRETA"},
            {"label": "NG", "label_provenance": "ARCHIVE_FOLDER_PROVISIONAL",
             "views": {"MID": {}}, "category_hint": None},
        ]
        c = quality_coverage(rows)
        self.assertTrue(c["each_available_light_has_both_labels"])
        self.assertFalse(c["fully_operator_verified"])
        self.assertEqual(c["human_verified_v3"], 1)
        self.assertNotIn("by_category_light", c)


class WarmStartIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/"reports/aoi_quality_cnn_v3/candidates/sample"
        self.folder.mkdir(parents=True)
        self.candidate = self.folder/"aoi_quality_cnn_v3_candidate.pt"
        self.report = self.folder/"training_report.json"
        self.manifest_sha = "a"*64
        model = AOIQualityCNNV3()
        self.weights = {
            "schema": "visionx.aoi_quality_cnn_v3_candidate_only.v1",
            "architecture_schema": "visionx.aoi_pair_quality_cnn.v3.offline",
            "source_manifest_sha256": self.manifest_sha,
            "image_size": 96, "focus_fraction": .70,
            "lights": ("SIDE", "TOP", "MID"),
            "candidate_only": True, "production_approved": False,
            "automatic_xp_authorized": False,
            "state_dict": model.state_dict(),
        }
        self.write_checkpoint()

    def write_checkpoint(self):
        torch.save(self.weights, self.candidate)
        doc = {
            "schema": "visionx.aoi_quality_cnn_v3_training_report.v1",
            "candidate_only": True, "production_approved": False,
            "automatic_xp_authorized": False,
            "provenance": {"manifest_sha256": self.manifest_sha},
            "candidate_path": self.candidate.relative_to(self.root).as_posix(),
            "candidate_sha256": sha256(self.candidate.read_bytes()).hexdigest(),
        }
        self.report.write_text(json.dumps(doc), encoding="utf-8")

    def test_previous_v3_checkpoint_restores_with_exact_sha(self):
        model, detail = load_verified_v3_candidate(
            self.root, self.report, self.manifest_sha,
        )
        self.assertIsInstance(model, AOIQualityCNNV3)
        self.assertEqual(
            detail["candidate_sha256"],
            sha256(self.candidate.read_bytes()).hexdigest(),
        )

    def test_checkpoint_or_schema_tampering_is_rejected(self):
        self.candidate.write_bytes(self.candidate.read_bytes()+b"modified")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_verified_v3_candidate(self.root, self.report, self.manifest_sha)
        self.weights["schema"] = "visionx.faltando_comparative_cnn.v2"
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, "Checkpoint V3"):
            load_verified_v3_candidate(self.root, self.report, self.manifest_sha)

    def test_cross_manifest_use_is_not_silently_allowed(self):
        with self.assertRaisesRegex(ValueError, "relatório|auditado"):
            load_verified_v3_candidate(self.root, self.report, "b"*64)


if __name__ == "__main__":
    unittest.main()
