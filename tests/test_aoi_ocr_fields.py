import unittest

import numpy as np

from src.services.aoi_ocr_fields import (
    looks_like_component_reference,
    normalize_board_ocr,
    normalize_parts_ocr,
    normalize_value_ocr,
    recover_parts_from_text_zone,
)


class _FakeOutput:
    DICT = object()


class _FakeTesseract:
    Output = _FakeOutput

    def __init__(self):
        self.image_calls = []

    def image_to_data(self, _image, config="", output_type=None):
        # Linha equivalente a: Parts [RI~5 Block 74
        return {
            "text": ["Parts", "[RI~5", "Block", "74"],
            "left": [10, 90, 190, 250],
            "top": [20, 20, 20, 20],
            "width": [55, 70, 45, 25],
            "height": [20, 20, 20, 20],
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
        }

    def image_to_string(self, _image, config=""):
        self.image_calls.append(config)
        if "whitelist=0123456789~-" in config:
            return "3~5"
        return "RI~5"


class AOIOCRFieldNormalizationTests(unittest.TestCase):
    def test_board_removes_false_cell_border(self):
        self.assertEqual(
            normalize_board_ocr("[P22-22200 (PRINCIPAL) L13"),
            "P22-22200 (PRINCIPAL) L13",
        )

    def test_value_recovers_numeric_comparison_prefix(self):
        self.assertEqual(
            normalize_value_ocr("[io <= 2 <= 80 FALTANDO"),
            "10 <= 2 <= 80 FALTANDO",
        )

    def test_value_recovers_decimal_case_from_real_aoi(self):
        self.assertEqual(
            normalize_value_ocr("fo <= $4.872 <= 10 FALTANDO"),
            "0 <= 54.872 <= 10 FALTANDO",
        )

    def test_parts_recovers_u2_range_from_contextual_numeric_suffix(self):
        self.assertEqual(
            normalize_parts_ocr("u2~s"),
            "U2~5",
        )
        self.assertTrue(looks_like_component_reference("U2~5"))

    def test_value_does_not_replace_letters_outside_numeric_prefix(self):
        self.assertEqual(
            normalize_value_ocr("FALTANDO"),
            "FALTANDO",
        )

    def test_parts_removes_false_cell_border_and_spaces(self):
        self.assertEqual(normalize_parts_ocr("[R3 ~ 5"), "R3~5")

    def test_component_reference_contract(self):
        self.assertTrue(looks_like_component_reference("R3~5"))
        self.assertTrue(looks_like_component_reference("C120"))
        self.assertFalse(looks_like_component_reference("RI~5"))


class AOIPartsDirectedOCRTests(unittest.TestCase):
    def test_invalid_general_ocr_uses_numeric_directed_reading(self):
        image = np.full((100, 320, 3), 220, dtype=np.uint8)
        fake = _FakeTesseract()

        recovered = recover_parts_from_text_zone(
            image,
            "[RI~5",
            fake,
        )

        self.assertEqual(recovered, "R3~5")
        self.assertEqual(len(fake.image_calls), 2)

    def test_valid_parts_skips_extra_tesseract_calls(self):
        image = np.full((100, 320, 3), 220, dtype=np.uint8)
        fake = _FakeTesseract()

        recovered = recover_parts_from_text_zone(
            image,
            "R3~5",
            fake,
        )

        self.assertEqual(recovered, "R3~5")
        self.assertEqual(fake.image_calls, [])


if __name__ == "__main__":
    unittest.main()
