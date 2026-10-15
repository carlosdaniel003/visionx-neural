"""Regressões do benchmark OCR: comparações nunca aprovam OCR alternativo."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.services import aoi_ocr_benchmark as benchmark
from src.services import screen_monitor


GOOD = (
    "Board P22-22200 (PRINCIPAL) Li3\n"
    "Parts RI75 Block 1\n"
    "Value 12 <= 0 <= 60 FALTANDO"
)
BAD = (
    "Board P22-22200 (PRINCIPAL) Li3\n"
    "Parts R175 Block 1\n"
    "Value 12 <= 1 <= 60 FALTANDO"
)
EXPECTED = {
    "board": "",
    "parts": "",
    "value": "12 <= 0 <= 60 FALTANDO",
    "category": "FALTANDO",
}


class FakeOCR:
    def __init__(self):
        self.binary_shapes = []

    def image_to_string(self, binary, config=""):
        self.binary_shapes.append(binary.shape[:2])
        # Entradas com escala 2 e margem superior cortada perdem o campo Parts.
        return BAD if binary.shape[0] < 500 else GOOD


class OCRBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = np.full((450, 900, 3), 160, dtype=np.uint8)
        self.path = self.root / "frame.png"
        self.assertTrue(cv2.imwrite(str(self.path), self.frame))
        self.fake = FakeOCR()

    @staticmethod
    def monitor_factory():
        monitor = screen_monitor.ScreenMonitor()
        monitor._find_color_bar = lambda _mask: (120, 220, 240, 14)
        monitor._find_sibling_bar = lambda _mask, _blue: (510, 220, 240, 14)
        return monitor

    def run_benchmark(self, inputs, repeats=2):
        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", self.fake
        ):
            return benchmark.compare_variants(
                inputs, repeats=repeats,
                monitor_factory=self.monitor_factory,
                tesseract_module=self.fake,
            )

    def test_control_uses_original_ocr_and_fast_candidate_cannot_be_promoted(self):
        result = self.run_benchmark([(self.path, EXPECTED)])
        self.assertEqual(result["case_count"], 1)
        self.assertEqual(result["manifest_expected_count"], 1)
        control = result["summary"]["baseline_original"]
        self.assertEqual(control["mismatch_with_human_labels_cases"], 0)
        self.assertTrue(control["manifest_reference_agreement"])
        optimized = result["summary"]["scale_2"]
        self.assertEqual(optimized["different_from_baseline_cases"], 1)
        self.assertEqual(optimized["mismatch_with_human_labels_cases"], 1)
        self.assertFalse(optimized["manifest_reference_agreement"])
        self.assertFalse(optimized["eligible_for_automatic_activation"])
        self.assertEqual(
            result["cases"][0]["measurements"][0]["fields"], EXPECTED
        )
        self.assertTrue(result["no_production_change"])
        self.assertFalse(result["auto_promotion"])
        self.assertEqual(result["cases"][0]["sha256"],
                         benchmark.sha256(self.path.read_bytes()).hexdigest())

    def test_without_human_labels_matching_baseline_is_not_human_approval(self):
        result = self.run_benchmark([(self.path, None)], repeats=1)
        self.assertEqual(result["manifest_expected_count"], 0)
        candidate = result["summary"]["scale_3"]
        self.assertTrue(candidate["same_fields_as_baseline"])
        self.assertFalse(candidate["manifest_reference_agreement"])
        self.assertFalse(candidate["eligible_for_automatic_activation"])

    def test_little_solder_unknown_category_is_measured_not_approved(self):
        # Frame real public/debug_ocr/frame_completo.png contém Little Solder:
        # OCR reconhece Board/Parts/Value, mas categoria não é operacional.
        text = chr(10).join((
            "Board P24-03804 (PRINCIPAL) L05",
            "Parts JF271 Block 1",
            "Value 40 <= 25.602 <= 98 Little Solder",
        ))
        self.fake.image_to_string = lambda binary, config="": text
        report = self.run_benchmark([(self.path, None)], repeats=3)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["unsupported_baseline_category_cases"], 1)
        case = report["cases"][0]
        self.assertFalse(case["baseline_category_supported"])
        control = case["measurements"][0]["fields"]
        self.assertEqual(control["board"], "")
        self.assertEqual(control["parts"], "")
        self.assertEqual(
            control["value"], "40 <= 25.602 <= 98 Little Solder"
        )
        self.assertEqual(control["category"], "Unknown")
        for item in report["summary"].values():
            self.assertEqual(item["unsupported_baseline_category_cases"], 1)
            self.assertFalse(item["manifest_reference_agreement"])
            self.assertFalse(item["eligible_for_automatic_activation"])

    def test_unknown_category_cannot_be_human_approved_by_manifest(self):
        text = chr(10).join((
            "Board P24-03804 (PRINCIPAL) L05",
            "Parts JF271 Block 1",
            "Value 40 <= 25.602 <= 98 Little Solder",
        ))
        self.fake.image_to_string = lambda binary, config="": text
        expected = {
            "board": "",
            "parts": "",
            "value": "40 <= 25.602 <= 98 Little Solder",
            "category": "Unknown",
        }
        report = self.run_benchmark([(self.path, expected)], repeats=1)
        self.assertEqual(
            report["summary"]["baseline_original"]["mismatch_with_human_labels_cases"],
            0,
        )
        self.assertFalse(
            report["summary"]["baseline_original"]["manifest_reference_agreement"]
        )

    def test_reference_disagrees_with_original_and_does_not_approve_variant(self):
        expected = dict(EXPECTED, value="12 <= 1 <= 60 FALTANDO")
        report = self.run_benchmark([(self.path, expected)], repeats=1)
        self.assertEqual(
            report["summary"]["baseline_original"]["mismatch_with_human_labels_cases"],
            1,
        )
        candidate = report["summary"]["scale_2"]
        self.assertEqual(candidate["mismatch_with_human_labels_cases"], 0)
        # Mesmo candidato igual à referência declarada requer revisão do
        # erro do baseline; não promover em lote sem auditoria independente.
        self.assertFalse(candidate["manifest_reference_agreement"])
        self.assertFalse(candidate["eligible_for_automatic_activation"])

    def test_scale_three_pixel_pipeline_matches_original_production(self):
        image = np.arange(120 * 240 * 3, dtype=np.uint8).reshape(120, 240, 3)
        fake = FakeOCR()
        monitor = self.monitor_factory()
        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", fake
        ):
            monitor._ocr_fast(image)
        ref_binary_shape = fake.binary_shapes[0]
        candidate = benchmark.prepare_variant_binary(
            image, benchmark.Variant("scale_3", scale=3)
        )
        self.assertEqual(candidate.shape[:2], ref_binary_shape)
        # Confirma a identidade de pixels, não apenas o formato da imagem.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_big = cv2.resize(
            gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC,
        )
        _, reference = cv2.threshold(
            gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        np.testing.assert_array_equal(candidate, reference)

    def test_manifest_human_labels_and_invalid_cases_fail_closed(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps([
            {"path": "frame.png", "expected": EXPECTED}
        ]), encoding="utf-8")
        sources = benchmark._image_paths(SimpleNamespace(
            manifest=str(manifest), images=None,
        ))
        self.assertEqual(sources, [(self.path.resolve(strict=True), EXPECTED)])

        manifest.write_text(json.dumps([
            {"path": "frame.png", "expected": {"value": "fake"}}
        ]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected exige"):
            benchmark._image_paths(SimpleNamespace(
                manifest=str(manifest), images=None,
            ))

    def test_manifest_rejects_escape_and_duplicate_frames(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps([
            {"path": "../outside.png"}
        ]), encoding="utf-8")
        with self.assertRaises((OSError, ValueError)):
            benchmark._image_paths(SimpleNamespace(
                manifest=str(manifest), images=None,
            ))
        with self.assertRaisesRegex(ValueError, "duplicada"):
            benchmark._image_paths(SimpleNamespace(
                manifest=None, images=[str(self.path), str(self.path)],
            ))

    def test_control_that_changes_across_repeats_is_not_trusted(self):
        class Flaky(FakeOCR):
            def __init__(self):
                super().__init__()
                self.large_calls = 0

            def image_to_string(self, binary, config=""):
                if binary.shape[0] >= 500:
                    self.large_calls += 1
                    return GOOD if self.large_calls <= 2 else BAD
                return BAD

        fake = Flaky()
        # Duas medições de controle distintas devem reprovar a experiência,
        # não fabricar equivalência entre variantes.
        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", fake
        ):
            with self.assertRaisesRegex(ValueError, "instável"):
                benchmark.compare_variants(
                    [(self.path, None)], 3,
                    monitor_factory=self.monitor_factory,
                    tesseract_module=fake,
                )

    def test_cli_never_overwrites_previous_report(self):
        output = self.root / "result.json"
        args = [
            "--images", str(self.path), "--repeats", "1",
            "--output", str(output),
        ]
        actual_compare = benchmark.compare_variants

        def mocked_compare(sources, repeats):
            return actual_compare(
                sources, repeats,
                monitor_factory=self.monitor_factory,
                tesseract_module=self.fake,
            )

        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", self.fake
        ), patch.object(benchmark, "locate_bars", return_value=(
            (120, 220, 240, 14), (510, 220, 240, 14),
        )), patch.object(
            benchmark, "compare_variants", side_effect=mocked_compare,
        ):
            self.assertEqual(benchmark.main(args), 0)
            first = output.read_bytes()
            with self.assertRaises(FileExistsError):
                benchmark.main(args)
        self.assertEqual(output.read_bytes(), first)
        self.assertTrue(self.path.is_file())

    def test_invalid_original_ocr_never_passes_by_vacuous_equality(self):
        blank = FakeOCR()
        blank.image_to_string = lambda binary, config="": ""
        with patch.object(screen_monitor, "HAS_TESSERACT", True), patch.object(
            screen_monitor, "pytesseract", blank
        ):
            with self.assertRaisesRegex(ValueError, "Baseline tem campo"):
                benchmark.compare_variants(
                    [(self.path, None)], 1,
                    monitor_factory=self.monitor_factory,
                    tesseract_module=blank,
                )

    def test_invalid_repeat_count_rejected_without_inspection(self):
        with self.assertRaisesRegex(ValueError, "repeats"):
            self.run_benchmark([(self.path, EXPECTED)], repeats=0)
        with self.assertRaisesRegex(ValueError, "repeats"):
            self.run_benchmark([(self.path, EXPECTED)], repeats=21)


if __name__ == "__main__":
    unittest.main()
