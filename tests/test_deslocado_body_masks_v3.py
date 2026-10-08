"""DESLOCADO v3: revisão obrigatória do corpo inteiro antes de proxies/treino.

Cobertura dos OK legados e SIDE/TOP/MID, anti-marcação, hashes,
aprovado somente por humano e ausência de qualquer integração operacional.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from src.services.deslocado_body_masks_v3 import (
    SCHEMA, RESULT_SCHEMA, check_box, propose_box,
    prepare_body_masks, validate_body_masks,
)
from tests.test_deslocado_cnn import DeslocadoFixtures


class FullBodyReviewTests(DeslocadoFixtures):
    def setUp(self):
        super().setUp()
        self._preparation,self.run=self.prepared()
        self.manifest=self.run/"manifest.json"
        self.rois,self.review_folder=prepare_body_masks(self.manifest)
        self.review_file=self.review_folder/"body_masks_review.json"

    def _approve(self):
        row=json.loads(self.review_file.read_text(encoding="utf-8"))
        for entry in row["rows"]:
            entry["approved"]=True
            entry["review_notes"]="Corpo e terminais identificados visualmente; excluir pads"
        self.review_file.write_text(json.dumps(row,ensure_ascii=False),
                                    encoding="utf-8")
        return row

    def test_previews_for_every_legacy_and_multilight_ok(self):
        self.assertEqual(self.rois["schema"], SCHEMA)
        self.assertEqual(self.rois["total_images"],6)
        self.assertEqual(self.rois["events"],4)
        self.assertEqual(self.rois["lighting_distribution"],
                         {"SIDE":4,"TOP":1,"MID":1})
        self.assertTrue(all(not row["approved"] for row in self.rois["rows"]))
        self.assertEqual(len(list((self.review_folder/"preview_proposals").glob("*.png"))),6)
        image=cv2.imread(str(next(
            (self.review_folder/"preview_proposals").glob("*.png")
        )))
        self.assertIsNotNone(image)
        self.assertGreater(image.shape[1],image.shape[0])
        self.assertFalse(self.rois["training_performed"])
        self.assertFalse(self.rois["production_modified"])

    def test_default_approval_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Falta revisão humana"):
            validate_body_masks(self.review_file)

    def test_verified_boxes_have_immutable_provenance_and_previews(self):
        original={
            p:sha256(p.read_bytes()).hexdigest()
            for p in (self.root/"public"/"ok_archive").glob("*.png")
        }
        self._approve()
        report,output=validate_body_masks(self.review_file)
        self.assertEqual(report["schema"],RESULT_SCHEMA)
        self.assertEqual(report["total_approved"],6)
        self.assertEqual(report["per_lighting"],{"SIDE":4,"TOP":1,"MID":1})
        self.assertEqual(len(list((output/"preview_validated").glob("*.png"))),6)
        self.assertTrue((output/"validated_body_masks.json").is_file())
        self.assertTrue(all(row["approved_by_operator"] for row in report["rows"]))
        self.assertFalse(report["production_approved"])
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["production_modified"])
        self.assertEqual(original,{
            p:sha256(p.read_bytes()).hexdigest() for p in original
        })

    def test_character_only_box_and_out_of_bounds_are_rejected(self):
        frame=np.zeros((200,120,3),np.uint8)
        with self.assertRaisesRegex(ValueError,"pequeno/grande"):
            check_box([51,78,18,38],frame)
        with self.assertRaisesRegex(ValueError,"ultrapassa"):
            check_box([30,30,150,80],frame)
        with self.assertRaisesRegex(ValueError,"inteiros"):
            check_box([20.0,10,50,120],frame)
        self.assertEqual(len(propose_box(frame)),4)

    def test_approval_of_marking_only_box_fails_even_with_true(self):
        raw=self._approve()
        raw["rows"][0]["body_box_test_xywh"]=[49,40,15,20]
        self.review_file.write_text(json.dumps(raw,ensure_ascii=False))
        with self.assertRaisesRegex(ValueError,"pequeno/grande"):
            validate_body_masks(self.review_file)

    def test_duplicate_or_missing_image_prevents_validation(self):
        raw=self._approve()
        raw["rows"][-1]=dict(raw["rows"][0])
        self.review_file.write_text(json.dumps(raw,ensure_ascii=False))
        with self.assertRaisesRegex(ValueError,"Duplicações"):
            validate_body_masks(self.review_file)

    def test_tampered_source_and_extraction_cannot_be_approved(self):
        self._approve()
        item=self.rois["rows"][0]
        pair=self.run/item["test_path"]
        pair.write_bytes(pair.read_bytes()+b"alterado")
        with self.assertRaisesRegex(ValueError,"alterado"):
            validate_body_masks(self.review_file)

    def test_stale_manifest_prevents_validation(self):
        self._approve()
        self.manifest.write_bytes(self.manifest.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"Manifesto foi modificado"):
            validate_body_masks(self.review_file)

    def test_mismatched_pair_boxes_cannot_be_approved(self):
        raw=self._approve()
        # Geometria discordante com dois intervalos tecnicamente válidos,
        # mas com larguras relativas incongruentes.
        raw["rows"][0]["body_box_reference_xywh"]=[30,16,40,40]
        raw["rows"][0]["body_box_test_xywh"]=[10,10,83,74]
        self.review_file.write_text(json.dumps(raw,ensure_ascii=False))
        with self.assertRaisesRegex(ValueError,"Dimensões relativas"):
            validate_body_masks(self.review_file)

    def test_ref_and_test_annotations_are_distinct(self):
        raw=self._approve()
        raw["rows"][0]["body_box_reference_xywh"]=[17,13,75,68]
        raw["rows"][0]["body_box_test_xywh"]=[15,15,73,68]
        self.review_file.write_text(json.dumps(raw,ensure_ascii=False))
        report,_=validate_body_masks(self.review_file)
        self.assertNotEqual(
            report["rows"][0]["body_box_reference_xywh"],
            report["rows"][0]["body_box_test_xywh"],
        )


if __name__=="__main__":
    unittest.main()
