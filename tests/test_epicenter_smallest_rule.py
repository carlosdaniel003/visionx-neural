import unittest

import cv2
import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor


class EpicenterSmallestRuleTests(unittest.TestCase):
    def test_two_green_rectangles_always_select_the_smaller(self):
        reference = np.full((180, 240, 3), 35, dtype=np.uint8)
        test = reference.copy()

        outer = (24, 18, 180, 142)
        inner = (152, 55, 28, 70)
        cv2.rectangle(
            reference,
            (outer[0], outer[1]),
            (outer[0] + outer[2] - 1, outer[1] + outer[3] - 1),
            (21, 149, 22),
            2,
        )
        cv2.rectangle(
            reference,
            (inner[0], inner[1]),
            (inner[0] + inner[2] - 1, inner[1] + inner[3] - 1),
            (12, 203, 12),
            2,
        )

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[outer, inner],
            global_box_info={
                "x": outer[0],
                "y": outer[1],
                "w": outer[2],
                "h": outer[3],
            },
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        self.assertEqual(selected, (153, 56, 26, 68))
        self.assertEqual(focus_reference.shape[:2], (68, 26))
        self.assertEqual(focus_reference.shape, focus_test.shape)

    def test_order_of_candidates_does_not_change_result(self):
        image = np.full((140, 200, 3), 40, dtype=np.uint8)
        outer = (18, 14, 160, 110)
        inner = (96, 72, 34, 26)

        first, _, _ = EpicenterExtractor.extract_focus(
            image,
            image.copy(),
            old_epicenters=[outer, inner],
            global_box_info={"x": 18, "y": 14, "w": 160, "h": 110},
        )
        second, _, _ = EpicenterExtractor.extract_focus(
            image,
            image.copy(),
            old_epicenters=[inner, outer],
            global_box_info={"x": 18, "y": 14, "w": 160, "h": 110},
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0], (97, 73, 32, 24))

    def test_visual_fallback_discards_larger_rectangle(self):
        reference = np.full((180, 240, 3), 30, dtype=np.uint8)
        test = reference.copy()
        outer = (20, 18, 190, 140)
        inner = (145, 48, 35, 78)

        cv2.rectangle(
            reference,
            (outer[0], outer[1]),
            (outer[0] + outer[2] - 1, outer[1] + outer[3] - 1),
            (13, 124, 15),
            2,
        )
        cv2.rectangle(
            reference,
            (inner[0], inner[1]),
            (inner[0] + inner[2] - 1, inner[1] + inner[3] - 1),
            (74, 202, 60),
            2,
        )

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={
                "x": outer[0],
                "y": outer[1],
                "w": outer[2],
                "h": outer[3],
            },
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        self.assertLess(selected[2] * selected[3], outer[2] * outer[3] * 0.25)
        self.assertGreater(selected[0], 120)


if __name__ == "__main__":
    unittest.main()
