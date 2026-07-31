import unittest

from src.core.strict_category_memory import canonical_memory_category
from src.utils.text_normalizer import CATEGORIES, normalize_aoi_text


class AOICategoryNormalizationTests(unittest.TestCase):
    def test_required_categories_are_canonical(self):
        self.assertIn("DESLOCADO", CATEGORIES)
        self.assertIn("EMBORCADO", CATEGORIES)
        self.assertIn("FALTANDO", CATEGORIES)
        self.assertIn("INVERTIDO", CATEGORIES)

    def test_double_d_displaced_ocr_is_normalized(self):
        category, _ = normalize_aoi_text("25 <= 88 DDESLOCADO")
        self.assertEqual(category, "DESLOCADO")

    def test_displaced_aliases(self):
        for text in ("DESLOCADO", "SHIFTED", "MISALIGNED", "OFFSET"):
            with self.subTest(text=text):
                self.assertEqual(normalize_aoi_text(text)[0], "DESLOCADO")

    def test_tombstone_aliases(self):
        for text in ("EMBORCADO", "TOMBSTONE", "TOMBSTONED", "STANDING"):
            with self.subTest(text=text):
                self.assertEqual(normalize_aoi_text(text)[0], "EMBORCADO")

    def test_existing_categories_are_preserved(self):
        expected = {
            "MISSING": "FALTANDO",
            "REVERSE": "INVERTIDO",
            "MUCH ADHESIVE": "MUITO ADESIVO",
        }
        for text, category in expected.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_aoi_text(text)[0], category)


class MemoryCategoryIsolationTests(unittest.TestCase):
    def test_memory_uses_same_canonical_category(self):
        aliases = {
            "DDESLOCADO": "DESLOCADO",
            "SHIFTED": "DESLOCADO",
            "EMBORCADO": "EMBORCADO",
            "TOMBSTONE": "EMBORCADO",
            "MISSING": "FALTANDO",
            "REVERSE": "INVERTIDO",
        }
        for value, expected in aliases.items():
            with self.subTest(value=value):
                self.assertEqual(canonical_memory_category(value), expected)

    def test_displaced_and_tombstone_memories_never_mix(self):
        self.assertNotEqual(
            canonical_memory_category("DDESLOCADO"),
            canonical_memory_category("EMBORCADO"),
        )
        self.assertNotEqual(
            canonical_memory_category("DESLOCADO"),
            canonical_memory_category("FALTANDO"),
        )
        self.assertNotEqual(
            canonical_memory_category("EMBORCADO"),
            canonical_memory_category("INVERTIDO"),
        )


if __name__ == "__main__":
    unittest.main()
