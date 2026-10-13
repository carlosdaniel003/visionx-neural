"""Memória KNN permanece após exclusão de PNGs e rejeita conflitos."""
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_record,
)
from src.core.verified_memory_router import VerifiedKNNMemory
from src.services.startup_regression.knn_archive_memory_cli import (
    synchronize_archive_knn, write_report,
)


def images(label):
    reference = np.zeros((40, 50, 3), dtype=np.uint8)
    reference[4:20, 5:25] = (75, 90, 20)
    test = reference.copy()
    if label == "NG":
        test[10:19, 12:25] = 0
    else:
        test[10:19, 12:25] = (75, 90, 20)
    return reference, test


def info():
    return {
        "board": "PCB-ABC", "parts": "U2~5",
        "category": "FALTANDO", "lighting_mode": "SIDE",
        "value": "0 <= 2 <= 80 FALTANDO",
    }


def entry(label, *, path=None, override_pair=None, lighting="SIDE"):
    ref, tst = override_pair or images(label)
    sig = build_anomaly_signature(ref, tst, {}, info())
    source = path or (
        "public/ok_archive/example.png" if label == "OK"
        else "public/ng_archive/example.png"
    )
    return make_record(
        label=label, source_path=source,
        source_sha256=sha256(source.encode()).hexdigest(),
        category="FALTANDO", lighting=lighting,
        reference=ref, test=tst,
        anomaly_signature=sig, aoi_info=info(),
    )


class DurableMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = PersistentArchiveKNN(self.root)

    def test_import_reboot_delete_source_then_exact_ok_ng(self):
        records = [entry("OK"), entry("NG")]
        self.assertEqual(self.store.add(records)["new"], 2)
        for row in records:
            original = self.root / row["sources"][0]["source_path"]
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b"placeholder-original")
            original.unlink()
        reboot = PersistentArchiveKNN(self.root)
        for label in ("OK", "NG"):
            r, t = images(label)
            result = reboot.lookup(r, t, info())
            self.assertEqual(result["status"], "KNOWN")
            self.assertEqual(result["label"], label)
        # Production exact matcher must also recognize independent snapshot.
        router = VerifiedKNNMemory(archive_store=reboot)
        class EmptyKNN:
            signatures_ok = []
            signatures_ng = []
        self.assertEqual(router.lookup(EmptyKNN(), *images("NG"), info())["label"], "NG")
        self.assertEqual(router.lookup(EmptyKNN(), *images("OK"), info())["label"], "OK")

    def test_duplicate_import_is_idempotent_and_does_not_forget(self):
        ok = entry("OK")
        self.store.add([ok])
        self.assertEqual(self.store.add([ok])["new"], 0)
        self.assertFalse(self.store.add([])["changed"])
        self.assertEqual(len(PersistentArchiveKNN(self.root).records()), 1)

    def test_new_sources_are_appended_without_removing_old(self):
        first = entry("OK")
        second = entry(
            "OK", path="public/ok_archive/second.png"
        )
        self.store.add([first])
        stats = self.store.add([second])
        self.assertTrue(stats["changed"])
        self.assertEqual(stats["new"], 0)
        self.assertEqual(len(self.store.records()[0]["sources"]), 2)
        self.assertTrue(self.store.path.with_suffix(".json.bak").exists())

    def test_conflicting_archive_labels_force_conflict(self):
        same_pair = images("OK")
        self.store.add([
            entry("OK", override_pair=same_pair),
            entry("NG", override_pair=same_pair),
        ])
        outcome = self.store.lookup(*same_pair, info())
        self.assertEqual(outcome["status"], "CONFLICT")
        router = VerifiedKNNMemory(archive_store=self.store)
        self.assertEqual(router.lookup(None, *same_pair, info())["status"], "CONFLICT")

    def test_corrupt_snapshot_is_not_silently_replaced(self):
        self.store.add([entry("NG")])
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["records"][0]["label"] = "OK"
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        reboot = PersistentArchiveKNN(self.root)
        with self.assertRaisesRegex(ValueError, "inválido|danificado"):
            reboot.records()
        with self.assertRaisesRegex(ValueError, "inválido|danificado"):
            reboot.add([entry("OK")])

    def test_delete_archive_then_empty_sync_does_not_erase_snapshot(self):
        self.store.add([entry("NG")])
        inventory = {
            "schema": "visionx.archive_inventory.v1",
            "summary": {"png_count": 0}, "images": [],
            "cross_label_conflicts": [],
        }
        report = synchronize_archive_knn(
            self.root, inventory=inventory,
            extractor=lambda frame: images(frame) + (info(),),
            frame_loader=lambda _root, i: i["expected_label"],
            store=self.store,
        )
        self.assertEqual(report["snapshot"]["total"], 1)
        self.assertEqual(len(PersistentArchiveKNN(self.root).records()), 1)
        self.assertFalse(report["summary"]["archive_98pct_target_met"])

    def test_learn_archive_reopen_and_recall_both_labels(self):
        rows = []
        for label in ("OK", "NG"):
            source = f"public/{'ok_archive' if label == 'OK' else 'ng_archive'}/x_{label}.png"
            rows.append({
                "path": source, "file_sha256": sha256(source.encode()).hexdigest(),
                "expected_label": label, "status": "VALID_PNG",
                "lighting_mode": "SIDE", "category_hint": "FALTANDO",
            })
        inventory = {
            "schema": "visionx.archive_inventory.v1",
            "summary": {"png_count": 2}, "images": rows,
            "cross_label_conflicts": [],
        }
        result = synchronize_archive_knn(
            self.root, inventory=inventory,
            extractor=lambda label: images(label) + (info(),),
            frame_loader=lambda _root, row: row["expected_label"],
            store=self.store,
        )
        self.assertEqual(result["summary"]["recalled_correct"], 2)
        self.assertTrue(result["summary"]["archive_98pct_target_met"])
        self.assertEqual(result["snapshot"]["by_label"], {"OK": 1, "NG": 1})
        path_j, path_t = write_report(
            result, self.root / "reports" / "startup_regression"
        )
        self.assertTrue(path_j.exists())
        self.assertIn("NG: 1/1", path_t.read_text(encoding="utf-8"))

    def test_match_only_exact_and_same_light(self):
        self.store.add([entry("OK")])
        ref, test = images("OK")
        changed = test.copy()
        changed[3, 3] = (1, 0, 0)
        self.assertEqual(self.store.lookup(ref, changed, info())["status"], "NEW")
        self.assertEqual(self.store.lookup(
            ref, test, {**info(), "lighting_mode": "TOP"}
        )["status"], "NEW")


if __name__ == "__main__":
    unittest.main()
