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


if __name__ == "__main__":
    unittest.main()