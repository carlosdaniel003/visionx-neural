"""Offline-only cross-category CNN candidate: source SHA, leakage and gates."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.services.faltando_neural_dataset import AOIPairExtractor, prepare_faltando
from src.services.aoi_universal_quality_archive import (
    split_universal_events, universal_archive_events,
)
from src.scripts.train_aoi_universal_quality_offline import (
    SCHEMA, _metrics, _selection_key, _trial_readonly_snapshot,
    _pinned_initialization, train_general_quality,
)
from tests.test_faltando_neural_dataset import FakeMonitor


class UniversalQualityOfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for label in ("ok", "ng"):
            (self.root/"public"/(label+"_archive")).mkdir(parents=True)
        self.items = []
        rng = np.random.default_rng(20261016)
        categories = ["FALTANDO", "INVERTIDO", "EMBORCADO",
                      "DESLOCADO", "MUITO_ADESIVO"]
        for i in range(40):
            category = categories[i % len(categories)]
            label = "NG" if i % 2 == 0 or i % 7 == 0 else "OK"
            # One shared timestamp across DIFFERENT AOI categories:
            minute = 1 if i == 2 else (i + 1)
            filename = f"2026-10-16_{minute:04d}_{category}_SIDE.png"
            folder = self.root/"public"/(label.lower()+"_archive")
            frame = rng.integers(10, 240, size=(40, 80, 3), dtype=np.uint8)
            valid, png = cv2.imencode(".png", frame)
            self.assertTrue(valid)
            target = folder/filename
            target.write_bytes(png.tobytes())
            self.items.append((target, category.replace("_", " "), label))
        # Separate explicit-annotation contract from provisional AOI labels.
        reviewed = [
            (p, category, "FALTANDO" if ix < 9 else "INVERTIDO")
            for ix, (p, category, label) in enumerate(
                (row for row in self.items if row[2] == "NG")
            ) if ix < 13
        ]
        ignored = next(row[0] for row in self.items if row[2] == "OK")
        cases = []
        for path, _, defect in reviewed:
            cases.append({
                "source": path.relative_to(self.root).as_posix(),
                "source_sha256": sha256(path.read_bytes()).hexdigest(),
                "visual_category": defect, "note": "Teste de revisão humana",
            })
        cases.append({
            "source": ignored.relative_to(self.root).as_posix(),
            "source_sha256": sha256(ignored.read_bytes()).hexdigest(),
            "visual_category": "IGNORAR", "note": "Não usar",
        })
        docs = self.root/"docs"
        docs.mkdir()
        (docs/"faltando_v2_operator_triage_20261016.json").write_text(
            json.dumps({
                "schema": "visionx.faltando_v2_operator_contact_sheet_triage.v1",
                "production_training_approved": False,
                "individual_qualification_file_created": False,
                "bulk_archive_labels_modified": False,
                "cases": cases,
            }), encoding="utf-8",
        )
        self.manifest, self.run = self._prepare()

    def _prepare(self):
        _, run = prepare_faltando(
            self.root, extractor=AOIPairExtractor(FakeMonitor()),
            include_all_aoi_categories=True,
        )
        return run/"manifest.json", run

    def test_all_categories_and_real_sha_triage_without_editing_png(self):
        before = {str(p): sha256(p.read_bytes()).hexdigest()
                  for p, _, _ in self.items}
        rows, provenance = universal_archive_events(self.manifest)
        self.assertEqual(provenance["source_pairs"], 40)
        self.assertEqual(provenance["human_ignored_sources"], 1)
        self.assertEqual(provenance["human_reviewed_general_ng"], 13)
        self.assertEqual(len(rows), 39)
        self.assertEqual({r["category_hint"] for r in rows}, {
            "FALTANDO", "INVERTIDO", "EMBORCADO",
            "DESLOCADO", "MUITO ADESIVO",
        })
        self.assertTrue(all(not r["event_link_verified"] for r in rows))
        self.assertEqual(
            {r["visual_defect_type"] for r in rows if r["visual_defect_type"]},
            {"FALTANDO", "INVERTIDO"},
        )
        self.assertEqual(before, {
            source: sha256(Path(source).read_bytes()).hexdigest()
            for source in before
        })
        self.assertFalse((self.run/"qualification.json").exists())

    def test_split_is_deterministic_and_no_time_or_hash_overlap(self):
        rows, _ = universal_archive_events(self.manifest)
        train, val, test, detail = split_universal_events(rows)
        self.assertEqual(len(train)+len(val)+len(test), 39)
        self.assertEqual(set(r["id"] for r in train) &
                         set(r["id"] for r in test), set())
        self.assertEqual(
            detail["folds"]["test"]["records"], len(test)
        )
        group_locations = {}
        for fold, events in (("train", train), ("validation", val), ("test", test)):
            for e in events:
                key = e["split_group"]
                if key in group_locations:
                    self.assertEqual(group_locations[key], fold)
                group_locations[key] = fold
        train2, val2, test2, _ = split_universal_events(rows)
        self.assertEqual(
            [[r["id"] for r in fold] for fold in (train, val, test)],
            [[r["id"] for r in fold] for fold in (train2, val2, test2)],
        )
        self.assertFalse(detail["physical_event_association_verified"])

    def test_fail_closed_without_explicit_provisional_opt_in(self):
        with self.assertRaisesRegex(ValueError, "provisórios"):
            train_general_quality(self.root, self.manifest, epochs=1)
        self.assertFalse(
            (self.root/"reports/aoi_universal_quality").exists()
        )

    def test_sha_tamper_blocks_training(self):
        p, _, _ = self.items[0]
        p.write_bytes(p.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "modificado"):
            universal_archive_events(self.manifest)

    def test_candidate_training_real_gradients_and_never_promotes(self):
        report = train_general_quality(
            self.root, self.manifest,
            epochs=1, image_size=64, learning_rate=.00005,
            allow_provisional_archive_labels=True,
        )
        self.assertEqual(report["schema"],
                         "visionx.aoi_general_quality_experiment_report.v1")
        self.assertFalse(report["epoch_selection_uses_test"])
        self.assertFalse(report["promoted"])
        self.assertFalse(report["production_approval_eligible"])
        self.assertFalse(report["blind_test_claim"])
        self.assertFalse(report["physical_defect_type_predicted"])
        self.assertEqual(len(report["epoch_history"]), 2)
        self.assertGreater(report["epoch_history"][1]["mean_loss"], 0.)
        self.assertTrue(report["trained_on_provisional_labels"])
        checkpoint = torch.load(
            self.root/report["source_candidate_relative_path"],
            map_location="cpu", weights_only=True,
        )
        self.assertEqual(checkpoint["schema"], SCHEMA)
        self.assertFalse(checkpoint["production_approved"])
        self.assertFalse(checkpoint["automatic_ok_authorized"])
        self.assertTrue(checkpoint["candidate_only"])
        self.assertFalse(checkpoint["aoi_category_used_as_model_input"])
        self.assertNotEqual(
            checkpoint["schema"],
            "visionx.faltando_comparative_cnn.v2",
        )
        self.assertFalse(
            (self.root/"reports/neural_online/live_active.json").exists()
        )

    def test_online_pointer_blocks_by_default_without_modification(self):
        pointer = self.root/"reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"schema":"unknown"}', encoding="utf-8")
        before = pointer.read_bytes()
        with self.assertRaisesRegex(ValueError, "Ponteiro online ativo"):
            train_general_quality(
                self.root, self.manifest, epochs=1,
                allow_provisional_archive_labels=True,
                initialize_from_pinned_champion=True,
            )
        self.assertEqual(pointer.read_bytes(), before)

    def test_readonly_override_rejects_untrusted_pointer(self):
        pointer = self.root/"reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"schema":"untrusted"}', encoding="utf-8")
        before = pointer.read_bytes()
        with self.assertRaisesRegex(ValueError, "outra implantação"):
            train_general_quality(
                self.root, self.manifest, epochs=1,
                allow_provisional_archive_labels=True,
                allow_verified_trial_pointer_readonly=True,
                initialize_from_pinned_champion=True,
            )
        self.assertEqual(pointer.read_bytes(), before)

    def test_audited_trial_snapshot_is_readonly_and_tracks_change(self):
        pointer = self.root/"reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"schema":"trial"}', encoding="utf-8")
        before = pointer.read_bytes()
        with patch(
            "src.scripts.train_aoi_universal_quality_offline._read_trial_pointer",
            return_value={"validated": True},
        ) as audited:
            stamp = _trial_readonly_snapshot(self.root, opt_in=True)
            self.assertEqual(stamp, sha256(before).hexdigest())
            self.assertEqual(pointer.read_bytes(), before)
            pointer.write_text('{"schema":"trial","changed":true}', encoding="utf-8")
            self.assertNotEqual(
                _trial_readonly_snapshot(self.root, opt_in=True), stamp,
            )
        self.assertEqual(audited.call_count, 2)

    def test_initialization_uses_original_pinned_file_not_active_pointer(self):
        from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
        from src.core.neural.faltando_live import PINNED_RELATIVE_CHECKPOINT
        original = self.root/PINNED_RELATIVE_CHECKPOINT
        original.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": MODEL_SCHEMA_V2, "experimental": True,
            "production_approved": False, "lights": LIGHTS,
            "state_dict": FaltandoCNNV2().state_dict(),
        }, original)
        digest = sha256(original.read_bytes()).hexdigest()
        pointer = self.root/"reports/neural_online/live_active.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text('{"schema":"active-different-weight"}', encoding="utf-8")
        previous = pointer.read_bytes()
        with patch(
            "src.scripts.train_aoi_universal_quality_offline.PINNED_CHECKPOINT_SHA256",
            digest,
        ):
            path, model, actual = _pinned_initialization(self.root)
        self.assertEqual(path, original)
        self.assertEqual(actual, digest)
        self.assertEqual(model["schema"], MODEL_SCHEMA_V2)
        self.assertEqual(pointer.read_bytes(), previous)

    def test_category_metrics_do_not_infer_unknown_ng_recall(self):
        records = [
            {"id": "a", "category_hint": "FALTANDO", "label": "NG"},
            {"id": "b", "category_hint": "INVERTIDO", "label": "OK"},
        ]
        cases = [
            {"event_id_or_group_hint": "a", "predicted": "OK"},
            {"event_id_or_group_hint": "b", "predicted": "OK"},
        ]
        summary = _metrics(records, {"cases": cases})
        self.assertIsNone(
            summary["by_aoi_category"]["INVERTIDO"]["ng_recall"]
        )
        self.assertIsNone(summary["macro_category_balanced_accuracy"])
        with self.assertRaisesRegex(ValueError, "ambas as classes"):
            _selection_key(summary)
