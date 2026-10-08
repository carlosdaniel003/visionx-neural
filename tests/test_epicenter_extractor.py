import unittest

import cv2
import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor


GREEN = (0, 255, 0)
BACKGROUND = (28, 30, 34)


class EpicenterExtractorRegressionTests(unittest.TestCase):
    def test_radar_ignores_global_frame_and_selects_central_inner_roi(self):
        reference = np.full((180, 240, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()

        # A moldura global ocupa mais de 85% da imagem e deve ser ignorada.
        cv2.rectangle(reference, (5, 5), (234, 174), GREEN, 2)
        cv2.rectangle(test, (5, 5), (234, 174), GREEN, 2)

        # A ROI real está dentro da moldura global e próxima ao centro.
        expected = (102, 68, 38, 44)
        x, y, width, height = expected
        cv2.rectangle(
            reference,
            (x, y),
            (x + width - 1, y + height - 1),
            GREEN,
            2,
        )
        cv2.rectangle(
            test,
            (x, y),
            (x + width - 1, y + height - 1),
            GREEN,
            2,
        )
        test[y + 5 : y + height - 5, x + 5 : x + width - 5] = (220, 220, 220)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"w": 230, "h": 170},
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        for current, target in zip(selected, expected):
            self.assertLessEqual(abs(int(current) - int(target)), 4)
        self.assertEqual(focus_reference.shape, focus_test.shape)
        self.assertEqual(focus_reference.shape[:2], (selected[3], selected[2]))
        self.assertGreater(float(np.mean(focus_test)), float(np.mean(focus_reference)))

    def test_radar_prefers_candidate_closest_to_image_center(self):
        reference = np.full((200, 260, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()

        cv2.rectangle(reference, (12, 60), (49, 103), GREEN, 2)
        cv2.rectangle(reference, (111, 77), (151, 124), GREEN, 2)
        cv2.rectangle(test, (12, 60), (49, 103), GREEN, 2)
        cv2.rectangle(test, (111, 77), (151, 124), GREEN, 2)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={},
        )

        self.assertEqual(len(epicenters), 1)
        x, y, width, height = epicenters[0]
        self.assertLess(abs((x + width / 2) - 130), 8)
        self.assertLess(abs((y + height / 2) - 100), 8)


    def test_tall_narrow_aoi_roi_is_not_mistaken_for_global_frame(self):
        reference = np.full((540, 345, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()

        # Geometria equivalente ao caso real capturado da AOI: a moldura ocupa
        # quase todo o recorte, enquanto o epicentro é estreito e passa de 90%
        # da altura.
        cv2.rectangle(reference, (21, 21), (326, 539), GREEN, 2)
        cv2.rectangle(test, (21, 21), (326, 539), GREEN, 2)
        cv2.rectangle(reference, (133, 48), (213, 539), GREEN, 2)
        cv2.rectangle(test, (133, 48), (213, 539), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[(134, 49, 81, 491)],
            global_box_info={"w": 307, "h": 519},
        )

        self.assertEqual(len(epicenters), 1)
        x, y, width, height = epicenters[0]
        self.assertGreater(height / 540.0, 0.85)
        self.assertLess(width / 345.0, 0.50)
        self.assertEqual(focus_reference.shape, focus_test.shape)
        self.assertGreater(focus_reference.size, 0)

    def test_global_frame_alone_remains_rejected(self):
        reference = np.full((540, 345, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()
        cv2.rectangle(reference, (21, 21), (326, 539), GREEN, 2)
        cv2.rectangle(test, (21, 21), (326, 539), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"w": 307, "h": 519},
        )

        self.assertEqual(epicenters, [])
        self.assertEqual(focus_reference.size, 0)
        self.assertEqual(focus_test.size, 0)

    def test_legacy_fallback_is_preserved_when_green_radar_finds_nothing(self):
        reference = np.full((180, 240, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()
        old_epicenters = [
            (90, 70, 44, 38),
            (104, 82, 24, 22),
        ]

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=old_epicenters,
            global_box_info={"w": 230, "h": 170},
        )

        self.assertEqual(epicenters[0], (90, 70, 44, 38))
        self.assertEqual(focus_reference.shape[:2], (38, 44))
        self.assertEqual(focus_reference.shape, focus_test.shape)


    def test_inner_epicenter_precedes_centered_global_when_outer_not_over_85_percent_height(self):
        """Reproduz coordenadas da captura real C6~2 em DESLOCADO."""
        reference = np.full((276, 570, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()
        for image in (reference, test):
            cv2.rectangle(image, (25, 25), (548, 254), GREEN, 2)
            cv2.rectangle(image, (76, 56), (264, 221), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[(78, 57, 189, 166)],
            global_box_info={"x": 25, "y": 25, "w": 525, "h": 230, "detected": True},
        )

        self.assertEqual(len(epicenters), 1)
        x, y, w, h = epicenters[0]
        self.assertAlmostEqual(x, 76, delta=4)
        self.assertAlmostEqual(y, 56, delta=4)
        self.assertAlmostEqual(w, 189, delta=5)
        self.assertAlmostEqual(h, 166, delta=5)
        self.assertEqual(focus_reference.shape, focus_test.shape)
        self.assertLess(w * h, 0.35 * 525 * 230)

    def test_three_nested_green_frames_select_deepest_real_epicenter(self):
        """Regressão do evento real 2eac69d: uma moldura intermediária
        corresponde perfeitamente ao TESTE, mas não é o epicentro.
        """
        reference = np.full((276, 570, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()
        for image in (reference, test):
            cv2.rectangle(image, (2, 2), (548, 274), GREEN, 2)
            cv2.rectangle(image, (25, 25), (545, 253), GREEN, 2)
            cv2.rectangle(image, (76, 56), (264, 221), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference, test,
            old_epicenters=[(25, 25, 522, 229), (79, 57, 188, 166)],
            global_box_info={"x": 2, "y": 2, "w": 549, "h": 274, "detected": True},
        )

        self.assertEqual(len(epicenters), 1)
        x, y, width, height = epicenters[0]
        self.assertAlmostEqual(x, 76, delta=4)
        self.assertAlmostEqual(y, 56, delta=4)
        self.assertAlmostEqual(width, 189, delta=5)
        self.assertAlmostEqual(height, 166, delta=5)
        self.assertEqual(focus_reference.shape, focus_test.shape)
        self.assertLess(width * height, 0.35 * 522 * 230)

    def test_outer_frame_without_inner_is_not_accepted_at_83_percent_height(self):
        reference = np.full((276, 570, 3), BACKGROUND, dtype=np.uint8)
        cv2.rectangle(reference, (25, 25), (548, 254), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference, reference.copy(), old_epicenters=[],
            global_box_info={"x": 25, "y": 25, "w": 525, "h": 230, "detected": True},
        )
        self.assertEqual(epicenters, [])
        self.assertEqual(focus_reference.size, 0)
        self.assertEqual(focus_test.size, 0)

    def test_crossing_frames_clipped_at_bottom_recover_inner_roi(self):
        """Caso real 08/10: os dois topos se cruzam e os fundos saem da tela."""
        reference = np.full((540, 394, 3), BACKGROUND, dtype=np.uint8)
        test = reference.copy()
        for image in (reference, test):
            cv2.rectangle(image, (15, 27), (377, 550), GREEN, 2)
            cv2.rectangle(image, (29, 14), (365, 550), GREEN, 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={},
        )

        self.assertEqual(len(epicenters), 1)
        x, y, width, height = epicenters[0]
        self.assertAlmostEqual(x, 29, delta=4)
        self.assertAlmostEqual(y, 14, delta=4)
        self.assertAlmostEqual(width, 338, delta=6)
        self.assertGreater(height, 510)
        self.assertEqual(focus_reference.shape, focus_test.shape)

    def test_small_interruptions_in_crossing_frames_can_be_recovered(self):
        reference = np.full((540, 394, 3), BACKGROUND, dtype=np.uint8)
        cv2.rectangle(reference, (15, 27), (377, 550), GREEN, 2)
        cv2.rectangle(reference, (29, 14), (365, 550), GREEN, 2)
        reference[150:156, 28:32] = BACKGROUND
        reference[260:266, 364:368] = BACKGROUND

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=[],
            global_box_info={},
        )
        self.assertEqual(len(epicenters), 1)
        self.assertAlmostEqual(epicenters[0][0], 29, delta=4)

    def test_outer_frame_and_unrelated_line_do_not_invent_epicenter(self):
        reference = np.full((540, 394, 3), BACKGROUND, dtype=np.uint8)
        cv2.rectangle(reference, (15, 27), (377, 550), GREEN, 2)
        cv2.line(reference, (28, 80), (28, 525), GREEN, 2)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=[],
            global_box_info={},
        )
        self.assertEqual(epicenters, [])


if __name__ == "__main__":
    unittest.main()