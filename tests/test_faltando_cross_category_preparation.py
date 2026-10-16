"""All-category AOI coverage is diagnostic, not missing-defect ground truth."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.faltando_neural_dataset import (
    AOIPairExtractor, prepare_faltando,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)
from tests.test_faltando_neural_dataset import FakeMonitor


class AllAOICategoriesPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for label in ("ok_archive", "ng_archive"):
            (self.root / "public" / label).mkdir(parents=True)
        self.extractor = AOIPairExtractor(FakeMonitor())
        self.paths = []
        categories = (
            ("FALTANDO", "NG"),
            ("INVERTIDO", "NG"),
            ("EMBORCADO", "OK"),
            ("DESLOCADO", "NG"),
            ("MUITOADESIVO", "NG"),
            ("INCLASSIFICAVEL", "NG"),
        )
        for index, (category, label) in enumerate(categories):
            frame = np.full((30, 60, 3), 50 + 12*index, dtype=np.uint8)
            valid, data = cv2.imencode(".png", frame)
            self.assertTrue(valid)
            target = self.root / "public" / (label.lower() + "_archive")
            filename = target / f"2026-10-16_120{index}_{category}_SIDE.png"
            filename.write_bytes(data.tobytes())
            self.paths.append(filename)

    def test_all_categories_extracted_without_claiming_visual_defect(self):
        original = {str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                    for x in self.paths}
        result, directory = prepare_faltando(
            self.root, extractor=self.extractor,
            include_all_aoi_categories=True,
        )
        self.assertEqual(result["summary"]["total"], 6)
        self.assertEqual(result["summary"]["extracted"], 6)
        self.assertEqual(set(result["summary"]["by_aoi_category_hint"]), {
            "FALTANDO", "INVERTIDO", "EMBORCADO", "DESLOCADO",
            "MUITOADESIVO", "UNKNOWN",
        })
        self.assertTrue(result["all_aoi_categories_preparation_only"])
        self.assertTrue(result["aoi_category_not_visual_defect_type"])
        self.assertFalse(result["dataset_ready"])
        self.assertEqual(result["summary"]["training_ready"], 0)
        self.assertTrue(all(x["visual_defect_type"] is None
                            for x in result["samples"]))
        self.assertTrue(all(
            "CONFIRM_VISUAL_DEFECT_TYPE" in x["review_required"]
            for x in result["samples"]
        ))
        self.assertTrue(all(x["quality_label_is_provisional"]
                            for x in result["samples"]))
        self.assertTrue(all(x["aoi_category_is_not_visual_defect_type"]
                            for x in result["samples"]))
        self.assertEqual(original, {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in original
        })
        self.assertFalse((directory/"qualification.json").exists())
        with self.assertRaisesRegex(ValueError, "multcategoria é diagnóstico"):
            archive_labeled_experimental_events(directory/"manifest.json")

    def test_default_stays_legacy_for_existing_verified_archive(self):
        result, directory = prepare_faltando(self.root, extractor=self.extractor)
        self.assertEqual(result["summary"]["total"], 1)
        self.assertFalse(result["all_aoi_categories_preparation_only"])
        self.assertFalse(result["aoi_category_not_visual_defect_type"])
        self.assertEqual(result["samples"][0]["category_hint"], "FALTANDO")

    def test_all_aoi_option_type_is_strict(self):
        with self.assertRaisesRegex(ValueError, "deve ser booleana"):
            prepare_faltando(
                self.root, extractor=self.extractor,
                include_all_aoi_categories="true",
            )
