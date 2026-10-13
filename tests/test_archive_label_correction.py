"""Etapa 2: retificação offline de 9 imagens OK falsas, sem tocar pesos CNN."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_operator_record, make_record,
)
from src.services.archive_label_correction import (
    apply_correction, prepare_correction, unfinished_corrections,
)
from src.services.image_archive_dedup import image_fingerprint


class ArchiveLabelCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for sub in (
            "public/ok_archive", "public/ng_archive",
            "public/dataset/nao_anomalia/FALTANDO",
            "public/dataset/nao_anomalia/INVERTIDO",
            "public/dataset/anomalia/FALTANDO",
            "public/dataset/anomalia/INVERTIDO",
            "reports/neural_online/events",
        ):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.captures = []
        self.dataset = []
        self.reference_pairs = {}
        self.store = PersistentArchiveKNN(self.root)
        proposed = []
        modes = ("SIDE", "TOP", "MID")
        for number, (event, category, timestamp) in enumerate((
            ("ev1", "FALTANDO", "1249"),
            ("ev2", "INVERTIDO", "1249"),
            ("ev3", "FALTANDO", "1253"),
        )):
            for offset, mode in enumerate(modes):
                index = number * 3 + offset
                source = np.full(
                    (80, 130, 3), (15 + index * 8, 23 + index, 40), dtype=np.uint8
                )
                reference = np.full(
                    (44, 46, 3), (100 + index, 20, 30), dtype=np.uint8
                )
                test = np.full(
                    (44, 46, 3), (70 + index, 30, 80), dtype=np.uint8
                )
                filename = f"2026-10-13_{timestamp}_{category}_{mode}.png"
                archive = self.root / "public" / "ok_archive" / filename
                self.assertTrue(cv2.imwrite(str(archive), source))
                self.captures.append(archive)
                stem = f"memory_OK_{mode}_{index:02d}"
                base = self.root / "public" / "dataset" / "nao_anomalia" / category
                paths = {
                    "source": base / (stem + "_source.png"),
                    "reference": base / (stem + "_reference.png"),
                    "test": base / (stem + "_test.png"),
                    "json": base / (stem + ".json"),
                }
                self.assertTrue(cv2.imwrite(str(paths["source"]), source))
                self.assertTrue(cv2.imwrite(str(paths["reference"]), reference))
                self.assertTrue(cv2.imwrite(str(paths["test"]), test))
                payload = {
                    "schema": "visionx.memory.v3",
                    "label": "OK",
                    "decision": {"operator_label": "OK", "ai_label": "NG",
                                 "disagreement": True},
                    "analysis": {"operator_label": "OK"},
                    "aoi_info": {"category": category, "lighting_mode": mode,
                                 "board": "P22-22200 L13", "parts": "D1~5"},
                    "multilight": {"event_id": event, "lighting_mode": mode},
                    "storage": {
                        "source_image_file": paths["source"].name,
                        "reference_image_file": paths["reference"].name,
                        "test_image_file": paths["test"].name,
                        "test_image_fingerprint": image_fingerprint(test),
                        "visual_deduplicated": False,
                    },
                    "image_file": paths["test"].name,
                }
                paths["json"].write_text(json.dumps(payload), encoding="utf-8")
                self.dataset.append(paths)
                self.reference_pairs[(event, mode)] = (reference, test)
                signature = build_anomaly_signature(
                    reference, test, {}, {"category": category, "lighting_mode": mode},
                )
                proposed.append(make_operator_record(
                    label="OK", source="xp_keyboard", event_id=event,
                    category=category, lighting=mode, reference=reference,
                    test=test, anomaly_signature=signature,
                ))
                from hashlib import sha256
                proposed.append(make_record(
                    label="OK", source_path="public/ok_archive/" + archive.name,
                    source_sha256=sha256(archive.read_bytes()).hexdigest(),
                    category=category, lighting=mode, reference=reference,
                    test=test, anomaly_signature=signature,
                ))
        self.store.add(proposed)

        # Evento CNN antigo do mesmo event_id; as três ROIs precisam coincidir.
        from hashlib import sha256
        entries = []
        for mode in modes:
            reference, test = self.reference_pairs[("ev1", mode)]
            rec = {"lighting_mode": mode}
            for role, image in (("reference", reference), ("test", test)):
                target = self.root / "reports/neural_online/events" / f"c1_{mode}_{role}.png"
                self.assertTrue(cv2.imwrite(str(target), image))
                rec[role] = target.name
                rec[role + "_sha256"] = sha256(target.read_bytes()).hexdigest()
            entries.append(rec)
        (self.root / "reports/neural_online/events/c1.json").write_text(
            json.dumps({"label": "OK", "category": "FALTANDO",
                        "source_aoi_event_id": "ev1", "images": entries}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_does_not_modify_originals(self):
        before = [(p, p.read_bytes()) for p in self.captures]
        plan = prepare_correction(self.root, self.captures)
        self.assertEqual(len(plan["items"]), 9)
        self.assertEqual(plan["memories_ok"], 9)
        self.assertEqual(len(plan["online"]), 1)
        self.assertEqual(len(plan["events"]), 3)
        for path, contents in before:
            self.assertEqual(path.read_bytes(), contents)

    def test_success_moves_nine_and_relabels_dataset_knn_and_journal(self):
        plan = prepare_correction(self.root, self.captures)
        result = apply_correction(plan)
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(result["images_corrected"], 9)
        self.assertEqual(result["events_corrected"], 3)
        self.assertEqual(result["cnn_online_journal_corrected"], 1)
        self.assertEqual(result["knn"]["old_ok_removed"], 9)
        for path in self.captures:
            self.assertFalse(path.exists())
            self.assertTrue((self.root / "public/ng_archive" / path.name).is_file())
        for row in self.dataset:
            old = row["json"]
            new = self.root / "public/dataset/anomalia" / old.parent.name / (
                old.name.replace("memory_OK_", "memory_NG_", 1)
            )
            self.assertFalse(old.exists())
            self.assertTrue(new.is_file())
            metadata = json.loads(new.read_text(encoding="utf-8"))
            self.assertEqual(metadata["label"], "NG")
            self.assertEqual(metadata["decision"]["operator_label"], "NG")
            self.assertEqual(metadata["analysis"]["operator_label"], "NG")
            self.assertFalse(metadata["decision"]["disagreement"])
            for role in ("source", "reference", "test"):
                changed = self.root / "public/dataset/anomalia" / old.parent.name / (
                    row[role].name.replace("memory_OK_", "memory_NG_", 1)
                )
                self.assertTrue(changed.is_file())
        self.assertEqual(
            json.loads((self.root / "reports/neural_online/events/c1.json")
                       .read_text(encoding="utf-8"))["label"], "NG"
        )
        memory = PersistentArchiveKNN(self.root)
        self.assertEqual(len(memory.records()), 9)
        self.assertTrue(all(x["label"] == "NG" for x in memory.records()))
        self.assertTrue(Path(result["backup_folder"]).is_dir())
        self.assertTrue((self.root / "data/knn_archive_memory/memory.json").is_file())

    def test_duplicate_or_incomplete_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exatamente 9"):
            prepare_correction(self.root, self.captures[:8])
        with self.assertRaisesRegex(ValueError, "exatamente 9"):
            prepare_correction(self.root, self.captures[:8] + [self.captures[0]])
        self.assertTrue(all(p.is_file() for p in self.captures))

    def test_missing_dataset_link_blocks_all_writes(self):
        self.dataset[0]["source"].unlink()
        with self.assertRaises(ValueError):
            prepare_correction(self.root, self.captures)
        self.assertTrue(all(p.is_file() for p in self.captures))

    def test_memory_failure_restores_every_original_file(self):
        before = {p: p.read_bytes() for p in self.captures}
        before_memory = self.store.path.read_bytes()
        plan = prepare_correction(self.root, self.captures)
        with patch.object(PersistentArchiveKNN, "replace_verified_labels",
                          side_effect=OSError("falha simulada do SSD")):
            with self.assertRaisesRegex(RuntimeError, "Backup restaurado"):
                apply_correction(plan)
        self.assertEqual(self.store.path.read_bytes(), before_memory)
        for p, content in before.items():
            self.assertTrue(p.exists())
            self.assertEqual(p.read_bytes(), content)
            self.assertFalse((self.root / "public/ng_archive" / p.name).exists())
        self.assertTrue(all(p["json"].exists() for p in self.dataset))
        statuses = list((self.root / "reports/label_corrections").glob("*/status.json"))
        self.assertEqual(len(statuses), 1)
        self.assertEqual(json.loads(statuses[0].read_text(encoding="utf-8"))["state"],
                         "ROLLED_BACK")

    def test_previous_incomplete_transaction_blocks_rewrite(self):
        transaction = self.root / "reports/label_corrections/incomplete"
        transaction.mkdir(parents=True)
        (transaction / "status.json").write_text(
            '{"state":"APPLYING"}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "interrompida"):
            prepare_correction(self.root, self.captures)
        self.assertEqual(unfinished_corrections(self.root), [transaction])


if __name__ == "__main__":
    unittest.main()
