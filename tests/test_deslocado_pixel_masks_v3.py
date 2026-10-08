"""Regressão de máscaras binárias DESLOCADO v3; nenhum treino operacional."""
from __future__ import annotations

from hashlib import sha256
import json

import cv2
import numpy as np

from src.services.deslocado_body_masks_v3 import prepare_body_masks, validate_body_masks
from src.services.deslocado_pixel_masks_v3 import (
    REVIEW_SCHEMA, VALIDATED_SCHEMA, _save_mask, check_pixel_mask, image_hash,
    prepare_pixel_review, validate_pixel_review,
)
from tests.test_deslocado_cnn import DeslocadoFixtures


class PixelMaskReviewTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        _, self.run = self.prepared()
        _, review_folder = prepare_body_masks(self.run / "manifest.json")
        review_file = review_folder / "body_masks_review.json"
        legacy = json.loads(review_file.read_text(encoding="utf-8"))
        for row in legacy["rows"]:
            row["approved"] = True
            row["review_notes"] = "Operador conferiu o retângulo inicial."
        review_file.write_text(json.dumps(legacy), encoding="utf-8")
        _, catalog_folder = validate_body_masks(review_file)
        self.legacy = catalog_folder / "validated_body_masks.json"
        self.pixel_review, self.pixel_folder = prepare_pixel_review(self.legacy)
        self.review_file = self.pixel_folder / "pixel_masks_review.json"

    def _write_review(self, review):
        self.review_file.write_text(
            json.dumps(review, ensure_ascii=False), encoding="utf-8"
        )

    def _approve(self, *, exclude=None):
        review = json.loads(self.review_file.read_text(encoding="utf-8"))
        for index, row in enumerate(review["rows"]):
            if index == exclude:
                row["excluded"] = True
                row["exclusion_reason"] = "CROPPED_COMPONENT"
                row["status"] = "EXCLUDED"
                continue
            for part in ("reference", "test"):
                H = row[part + "_size_wh"][1]
                W = row[part + "_size_wh"][0]
                mask = np.zeros((H, W), np.uint8)
                cv2.rectangle(mask, (W // 4, H // 4),
                              (W * 3 // 4, H * 3 // 4), 255, -1)
                file = self.pixel_folder / row["mask_" + part + "_path"]
                _save_mask(file, mask)
                row["mask_" + part + "_sha256"] = image_hash(file)
                row["edited_" + part] = True
            row["approved"] = True
            row["review_notes"] = "Operador conferiu pixels e terminais."
        self._write_review(review)
        return review

    def test_candidates_cannot_be_auto_approved_and_cover_all_lights(self):
        self.assertEqual(self.pixel_review["schema"], REVIEW_SCHEMA)
        self.assertEqual(self.pixel_review["total_images"], 6)
        self.assertEqual(
            {row["lighting_mode"] for row in self.pixel_review["rows"]},
            {"SIDE", "TOP", "MID"}
        )
        self.assertTrue(all(row["approved"] is False for row in self.pixel_review["rows"]))
        self.assertEqual(len(list((self.pixel_folder / "masks").glob("*.png"))), 12)
        with self.assertRaisesRegex(ValueError, "sem revisão"):
            validate_pixel_review(self.review_file)

    def test_full_review_emits_binary_masks_with_provenance_and_no_training(self):
        sources = {
            path: sha256(path.read_bytes()).hexdigest()
            for path in (self.root / "public" / "ok_archive").glob("*.png")
        }
        self._approve()
        result, folder = validate_pixel_review(self.review_file)
        self.assertEqual(result["schema"], VALIDATED_SCHEMA)
        self.assertEqual(result["total_reviewed"], 6)
        self.assertEqual(result["total_approved"], 6)
        self.assertEqual(result["total_excluded"], 0)
        self.assertEqual(len(list((folder / "masks_validated").glob("*.png"))), 12)
        self.assertEqual(len(list((folder / "preview_validated").glob("*.png"))), 6)
        self.assertFalse(result["production_approved"])
        self.assertFalse(result["production_modified"])
        self.assertFalse(result["training_performed"])
        self.assertEqual(sources, {
            path: sha256(path.read_bytes()).hexdigest() for path in sources
        })

    def test_explicit_exclusion_preserves_coverage(self):
        self._approve(exclude=0)
        result, folder = validate_pixel_review(self.review_file)
        self.assertEqual(result["total_reviewed"], 6)
        self.assertEqual(result["total_approved"], 5)
        self.assertEqual(result["total_excluded"], 1)
        self.assertEqual(result["excluded"][0]["reason"], "CROPPED_COMPONENT")

    def test_potential_crop_edge_must_fail(self):
        self._approve()
        review = json.loads(self.review_file.read_text(encoding="utf-8"))
        row = review["rows"][0]
        path = self.pixel_folder / row["mask_reference_path"]
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        frame[-1, frame.shape[1] // 2] = 255
        _save_mask(path, frame)
        row["mask_reference_sha256"] = image_hash(path)
        self._write_review(review)
        with self.assertRaisesRegex(ValueError, "borda do recorte"):
            validate_pixel_review(self.review_file)

    def test_white_printing_only_fails_minimum_coverage(self):
        frame = np.zeros((180, 120, 3), np.uint8)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.rectangle(mask, (45, 72), (70, 105), 255, -1)
        with self.assertRaisesRegex(ValueError, "Área da máscara"):
            check_pixel_mask(mask, frame)

    def test_external_tampering_of_mask_rejected_even_after_approval(self):
        self._approve()
        row = json.loads(self.review_file.read_text(encoding="utf-8"))["rows"][0]
        mask = self.pixel_folder / row["mask_reference_path"]
        mask.write_bytes(mask.read_bytes() + b"not authorized")
        with self.assertRaisesRegex(ValueError, "Hash da máscara"):
            validate_pixel_review(self.review_file)

    def test_duplicate_or_unreviewed_record_fails_closed(self):
        review = self._approve()
        review["rows"][-1] = dict(review["rows"][0])
        self._write_review(review)
        with self.assertRaisesRegex(ValueError, "Cobertura"):
            validate_pixel_review(self.review_file)

    def test_unedited_rectangle_placeholder_cannot_be_accepted(self):
        review = self._approve()
        row = review["rows"][0]
        row["proposal_reference_type"] = "RECTANGLE_PLACEHOLDER_EDIT_REQUIRED"
        row["edited_reference"] = False
        self._write_review(review)
        with self.assertRaisesRegex(ValueError, "Placeholder"):
            validate_pixel_review(self.review_file)

    def test_original_manifest_change_rejected(self):
        self._approve()
        manifest = self.run / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Manifesto de origem alterado"):
            validate_pixel_review(self.review_file)
