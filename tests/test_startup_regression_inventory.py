"""Regressões do inventário — sem acessar AOI, UI ou diretórios reais."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.services.startup_regression.archive_inventory import (
    filename_hints,
    inventory_archives,
)
from src.services.startup_regression.archive_inventory_report import (
    human_summary,
    write_reports,
)


class ArchiveInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ok_dir = self.root / "public" / "ok_archive"
        self.ng_dir = self.root / "public" / "ng_archive"
        self.ok_dir.mkdir(parents=True)
        self.ng_dir.mkdir(parents=True)

    def png(self, folder, name, value=100):
        image = np.full((30, 45, 3), value, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        path = folder / name
        path.write_bytes(encoded.tobytes())
        return path

    def manifest(self, folder, name, *, event_id="evt-real-1",
                 expected_label="NG", frames=None):
        payload = {
            "schema": "visionx.archive_regression.v1",
            "event_id": event_id,
            "expected_label": expected_label,
            "aoi_info": {
                "board": "BOARD-1", "parts": "U2~5",
                "category": "FALTANDO", "value": "FALTANDO",
            },
            "frames": frames or {},
        }
        path = folder / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def frames(paths):
        return {
            light: {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for light, path in paths.items()
        }

    def test_historic_and_new_filename_patterns(self):
        historic = filename_hints(
            Path("2026-10-01_07-35-16-002_INVERTIDO.png")
        )
        self.assertEqual(historic["lighting_mode"], "SIDE")
        self.assertEqual(historic["lighting_source"], "LEGACY_DEFAULT")
        self.assertEqual(historic["category_hint"], "INVERTIDO")

        adhesive = filename_hints(
            Path("2026-10-06_1042_MUITO_ADESIVO.png")
        )
        self.assertEqual(adhesive["category_hint"], "MUITO ADESIVO")

        top = filename_hints(Path("2026-10-08_0715_FALTANDO_TOP_2.png"))
        self.assertEqual(top["lighting_mode"], "TOP")
        self.assertEqual(top["lighting_source"], "EXPLICIT_SUFFIX")
        self.assertEqual(top["category_hint"], "FALTANDO")

    def test_legacy_inventory_uses_archive_labels_without_judging(self):
        p_ok = self.png(self.ok_dir, "2026-10-01_07-35-16-002_INVERTIDO.png")
        p_ng = self.png(self.ng_dir, "2026-10-01_07-36-23-677_FALTANDO.png", 200)
        original_ok = p_ok.read_bytes()
        original_ng = p_ng.read_bytes()

        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["png_count"], 2)
        self.assertEqual(report["summary"]["valid_png"], 2)
        self.assertEqual(report["summary"]["legacy_side_count"], 2)
        self.assertFalse(report["is_operational_gate"])
        self.assertFalse(report["analysis_performed"])
        self.assertEqual(
            {r["expected_label"] for r in report["images"]},
            {"OK", "NG"},
        )
        self.assertTrue(all(
            r["ocr_status"] == "NOT_EVALUATED_STAGE_1"
            and r["decision_status"] == "NOT_EVALUATED_STAGE_1"
            for r in report["images"]
        ))
        self.assertEqual(p_ok.read_bytes(), original_ok)
        self.assertEqual(p_ng.read_bytes(), original_ng)
        self.assertEqual(len(list(self.ok_dir.iterdir())), 1)
        self.assertEqual(len(list(self.ng_dir.iterdir())), 1)

    def test_explicit_multilight_without_manifest_remains_unlinked(self):
        for light, value in zip(("SIDE", "TOP", "MID"), (70, 80, 90)):
            self.png(self.ng_dir, f"2026-10-08_0715_FALTANDO_{light}.png", value)

        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["explicit_unlinked_count"], 3)
        self.assertEqual(report["summary"]["linked_event_count"], 0)
        self.assertTrue(all(x["event_id"] is None for x in report["images"]))
        self.assertEqual(
            sum(i["code"] == "MULTILIGHT_WITHOUT_MANIFEST"
                for i in report["issues"]),
            3,
        )

    def test_multilight_manifest_links_only_verified_frames(self):
        paths = {
            light: self.png(self.ng_dir, f"2026-10-08_0715_FALTANDO_{light}.png", val)
            for light, val in zip(("SIDE", "TOP", "MID"), (50, 100, 150))
        }
        self.manifest(self.ng_dir, "evt.json", frames=self.frames(paths))
        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["linked_event_count"], 1)
        self.assertEqual(report["summary"]["manifest_linked_png_count"], 3)
        self.assertEqual(report["summary"]["explicit_unlinked_count"], 0)
        self.assertTrue(all(x["event_id"] == "evt-real-1" for x in report["images"]))
        self.assertEqual(report["manifests"][0]["status"], "LINKED")
        self.assertEqual(report["status"], "INVENTORIED")

    def test_incomplete_manifest_is_not_accepted_as_complete(self):
        paths = {
            light: self.png(self.ng_dir, f"2026-10-08_0715_FALTANDO_{light}.png", val)
            for light, val in (("SIDE", 80), ("TOP", 160))
        }
        self.manifest(self.ng_dir, "evt.json", frames=self.frames(paths))

        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["linked_event_count"], 0)
        self.assertEqual(report["manifests"][0]["status"], "INVALID_MANIFEST")
        self.assertIn("INVALID_MANIFEST", [i["code"] for i in report["issues"]])

    def test_cross_label_pixel_conflict_and_same_label_duplicates(self):
        self.png(self.ok_dir, "2026-10-01_0800_FALTANDO.png", 111)
        self.png(self.ng_dir, "2026-10-01_0801_FALTANDO.png", 111)
        self.png(self.ng_dir, "2026-10-01_0802_FALTANDO.png", 111)
        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["pixel_duplicate_groups"], 1)
        self.assertEqual(report["summary"]["cross_label_conflict_groups"], 1)
        self.assertEqual(len(report["cross_label_conflicts"][0]["paths"]), 3)

    def test_corrupted_png_is_reported_not_skipped(self):
        (self.ng_dir / "broken_FALTANDO.png").write_bytes(b"not-png")
        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["invalid_png"], 1)
        self.assertEqual(report["summary"]["png_count"], 1)
        self.assertIn("INVALID_PNG", [i["code"] for i in report["issues"]])

    def test_unknown_file_and_unknown_category_are_visible(self):
        self.png(self.ng_dir, "unknown_format.png", 55)
        (self.ng_dir / "screenshot.jpg").write_bytes(b"hello")
        (self.ok_dir / "desktop.ini").write_text("system")
        report = inventory_archives(self.root)

        self.assertIn("UNKNOWN_CATEGORY_HINT", [i["code"] for i in report["issues"]])
        self.assertIn("UNSUPPORTED_FILE", [i["code"] for i in report["issues"]])
        self.assertEqual(report["summary"]["unsupported_file_count"], 1)

    def test_missing_archives_are_not_created(self):
        self.ok_dir.rmdir()
        self.ng_dir.rmdir()
        report = inventory_archives(self.root)

        self.assertFalse(self.ok_dir.exists())
        self.assertFalse(self.ng_dir.exists())
        self.assertEqual(report["summary"]["png_count"], 0)
        self.assertIn("MISSING_ARCHIVE_DIR", [i["code"] for i in report["issues"]])
        self.assertIn("NO_COVERAGE", [i["code"] for i in report["issues"]])

    def test_manifest_cannot_escape_archive_folder(self):
        paths = {
            light: self.png(self.ng_dir, f"2026-10-08_0715_FALTANDO_{light}.png", val)
            for light, val in zip(("SIDE", "TOP", "MID"), (50, 100, 150))
        }
        frames = self.frames(paths)
        frames["MID"]["path"] = "../ok_archive/another.png"
        self.manifest(self.ng_dir, "evt.json", frames=frames)
        report = inventory_archives(self.root)

        self.assertEqual(report["summary"]["linked_event_count"], 0)
        self.assertIn("MANIFEST_UNSAFE_PATH", [i["code"] for i in report["issues"]])

    def test_reports_are_outside_archive_and_contain_full_json(self):
        self.png(self.ng_dir, "2026-10-01_08-11_FALTANDO.png", 44)
        report = inventory_archives(self.root)
        report_dir = self.root / "reports" / "startup_regression"
        json_path, text_path = write_reports(report, report_dir)

        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), report)
        self.assertIn("SOMENTE DIAGNÓSTICO", human_summary(report))
        self.assertTrue(text_path.exists())
        self.assertEqual(len(list(self.ng_dir.iterdir())), 1)

        with self.assertRaises(ValueError):
            write_reports(report, self.ng_dir)


if __name__ == "__main__":
    unittest.main()
