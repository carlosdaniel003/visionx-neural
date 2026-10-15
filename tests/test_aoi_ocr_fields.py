import unittest

import numpy as np

from src.services.aoi_ocr_timing import AOIOCRTiming

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

    def test_board_normalizes_only_ambiguous_final_l_number(self):
        self.assertEqual(
            normalize_board_ocr("P22-22200 (PRINCIPAL) Li3"),
            "P22-22200 (PRINCIPAL) L13",
        )
        self.assertEqual(
            normalize_board_ocr("P22-22200 (PRINCIPAL) L13"),
            "P22-22200 (PRINCIPAL) L13",
        )
        self.assertEqual(
            normalize_board_ocr("P22-22200 (Li3) PRINCIPAL"),
            "P22-22200 (Li3) PRINCIPAL",
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

    def test_human_confirmed_e1_range_repairs_repeated_tilde_and_l_one(self):
        # Evidência visual SIDE/TOP/MID: E1~9; OCR exibiu EL~~~9.
        for raw in ("EL~~~9", "[EL~~~9", "EI~~~9", "E|~~~9",
                    "E1~~~9", "EL~~9"):
            with self.subTest(raw=raw):
                result = normalize_parts_ocr(raw)
                self.assertEqual(result, "E1~9")
                self.assertTrue(looks_like_component_reference(result))

    def test_e1_repair_is_strict_and_does_not_invent_other_components(self):
        # Não fazer correções globais L->1, I->1 ou ~~ -> ~.
        unchanged = (
            "EL~9", "EI~9", "RL~~~9", "FICE~S", "RI75",
            "EL9", "E1~9", "R3~~~5", "EIL~~~9",
        )
        for raw in unchanged:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_parts_ocr(raw), raw)
        self.assertFalse(looks_like_component_reference("EL~9"))
        self.assertFalse(looks_like_component_reference("FICE~S"))

    def test_component_reference_contract(self):
        self.assertTrue(looks_like_component_reference("R3~5"))
        self.assertTrue(looks_like_component_reference("C120"))
        self.assertFalse(looks_like_component_reference("RI~5"))


class _NoPartsAnchorTesseract(_FakeTesseract):
    def image_to_data(self, _image, config="", output_type=None):
        return {"text": []}


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

    def test_double_ambiguity_diode_fallback_without_anchor(self):
        image = np.full((100, 320, 3), 220, dtype=np.uint8)
        fake = _NoPartsAnchorTesseract()
        recovered = recover_parts_from_text_zone(image, "[Di~s", fake)
        self.assertEqual(recovered, "D1~5")
        self.assertEqual(fake.image_calls, [])

    def test_other_ambiguous_designator_is_not_forced_to_one(self):
        image = np.full((100, 320, 3), 220, dtype=np.uint8)
        fake = _NoPartsAnchorTesseract()
        recovered = recover_parts_from_text_zone(image, "RI~5", fake)
        self.assertEqual(recovered, "RI~5")

    def test_directed_ocr_timing_has_three_calls_without_changing_label(self):
        image = np.full((100, 320, 3), 220, dtype=np.uint8)
        fake = _FakeTesseract()
        timing = AOIOCRTiming()
        actual = recover_parts_from_text_zone(
            image, "[RI~5", fake, timing=timing
        )
        reference = recover_parts_from_text_zone(
            image, "[RI~5", _FakeTesseract()
        )
        self.assertEqual(actual, reference)
        self.assertEqual(actual, "R3~5")
        profile = timing.snapshot()
        self.assertEqual(profile["parts_reason"], "directed_candidate_valid")
        self.assertEqual(
            [row["phase"] for row in profile["calls"]],
            ["parts_anchor_data", "parts_full", "parts_numeric"],
        )
        self.assertEqual(profile["call_count"], 3)
        self.assertGreaterEqual(profile["tesseract_calls_ms"], 0)

    def test_valid_component_timing_skips_every_tesseract_subprocess(self):
        timing = AOIOCRTiming()
        result = recover_parts_from_text_zone(
            np.ones((100, 320, 3), dtype=np.uint8),
            "R3~5", _FakeTesseract(), timing=timing,
        )
        self.assertEqual(result, "R3~5")
        self.assertEqual(timing.snapshot()["call_count"], 0)
        self.assertEqual(
            timing.snapshot()["parts_reason"], "valid_general_reference"
        )

    def test_failing_tesseract_phase_is_profiled_then_propagated(self):
        timing = AOIOCRTiming()
        with self.assertRaises(ValueError):
            timing.call("general_blue", lambda: (_ for _ in ()).throw(
                ValueError("tesseract falhou")
            ))
        self.assertEqual(timing.snapshot()["call_count"], 1)
        self.assertEqual(timing.snapshot()["calls"][0]["status"], "error")

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
