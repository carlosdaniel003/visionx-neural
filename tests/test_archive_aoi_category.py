"""Contrato único da categoria OCR em importação KNN e replay histórico."""
import unittest

from src.services.archive_aoi_category import resolve_archive_aoi_category


class ArchiveAOICategoryTests(unittest.TestCase):
    def test_current_screen_monitor_has_category_but_empty_value(self):
        for category in ("FALTANDO", "INVERTIDO", "DESLOCADO", "EMBORCADO", "MUITO ADESIVO"):
            with self.subTest(category=category):
                observed, value = resolve_archive_aoi_category(
                    {"category": category, "value": "", "parts": "", "board": ""},
                    category,
                )
                self.assertEqual(observed.replace(" ", ""), category.replace(" ", ""))
                self.assertEqual(value, "")

    def test_legacy_value_is_supported_when_category_is_missing(self):
        actual, value = resolve_archive_aoi_category(
            {"category": "Unknown", "value": "0 <= 60 <= 80 FALTANDO"},
            "FALTANDO",
        )
        self.assertEqual(actual, "FALTANDO")
        self.assertEqual(value, "0 <= 60 <= 80 FALTANDO")

    def test_filename_never_supplies_missing_ocr_category(self):
        for info in (
            {"category": "Unknown", "value": ""},
            {"category": "", "value": ""},
            {},
        ):
            with self.subTest(info=info):
                with self.assertRaisesRegex(ValueError, "Categoria AOI ausente"):
                    resolve_archive_aoi_category(info, "FALTANDO")

    def test_real_category_mismatch_still_blocks(self):
        for info in (
            {"category": "INVERTIDO", "value": ""},
            {"category": "FALTANDO", "value": "INVERTIDO"},
        ):
            with self.subTest(info=info):
                with self.assertRaisesRegex(ValueError, "não confere|Conflito"):
                    resolve_archive_aoi_category(info, "FALTANDO")

    def test_unknown_filename_hint_cannot_be_promoted_to_label(self):
        with self.assertRaisesRegex(ValueError, "arquivo não reconhecida"):
            resolve_archive_aoi_category(
                {"category": "FALTANDO", "value": ""},
                "UNKNOWN",
            )


if __name__ == "__main__":
    unittest.main()
