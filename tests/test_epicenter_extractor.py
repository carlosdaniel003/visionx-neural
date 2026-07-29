import unittest

import cv2
import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor


GREEN = (0, 255, 0)
BACKGROUND = (24, 27, 31)


def canvas(width=220, height=180):
    return np.full((height, width, 3), BACKGROUND, dtype=np.uint8)


def draw_nested_rectangles(
    image,
    outer=(24, 18, 174, 144),
    inner=(91, 67, 34, 31),
    outer_thickness=3,
    inner_thickness=2,
):
    outer_x, outer_y, outer_width, outer_height = outer
    inner_x, inner_y, inner_width, inner_height = inner
    cv2.rectangle(
        image,
        (outer_x, outer_y),
        (outer_x + outer_width - 1, outer_y + outer_height - 1),
        GREEN,
        outer_thickness,
    )
    cv2.rectangle(
        image,
        (inner_x, inner_y),
        (inner_x + inner_width - 1, inner_y + inner_height - 1),
        GREEN,
        inner_thickness,
    )
    return image


def assert_box_near(test_case, actual, expected, tolerance=5):
    test_case.assertIsNotNone(actual)
    for current, target in zip(actual, expected):
        test_case.assertLessEqual(abs(int(current) - int(target)), tolerance)


class EpicenterExtractorTests(unittest.TestCase):
    def test_smaller_concentric_box_beats_large_centered_box(self):
        reference = draw_nested_rectangles(canvas())
        test = reference.copy()

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"x": 24, "y": 18, "w": 174, "h": 144},
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        assert_box_near(self, selected, (91, 67, 34, 31), tolerance=5)
        self.assertLess(selected[2] * selected[3], 2500)
        self.assertEqual(focus_reference.shape[:2], (selected[3], selected[2]))
        self.assertEqual(focus_reference.shape, focus_test.shape)

    def test_thick_lines_do_not_create_false_duplicate_epicenter(self):
        reference = draw_nested_rectangles(
            canvas(),
            outer=(20, 14, 180, 150),
            inner=(92, 66, 36, 34),
            outer_thickness=6,
            inner_thickness=5,
        )

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=[],
            global_box_info={"x": 20, "y": 14, "w": 180, "h": 150},
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        assert_box_near(self, selected, (92, 66, 36, 34), tolerance=7)
        self.assertLess(selected[2], 60)
        self.assertLess(selected[3], 60)

    def test_inner_diamond_is_selected_inside_global_rectangle(self):
        reference = canvas()
        cv2.rectangle(reference, (22, 15), (198, 164), GREEN, 3)
        diamond = np.asarray(
            [[110, 55], [139, 84], [110, 113], [81, 84]],
            dtype=np.int32,
        )
        cv2.polylines(reference, [diamond], True, GREEN, 3, cv2.LINE_AA)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=[],
            global_box_info={"x": 22, "y": 15, "w": 177, "h": 150},
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        self.assertLess(selected[2] * selected[3], 5000)
        selected_center = (
            selected[0] + selected[2] / 2.0,
            selected[1] + selected[3] / 2.0,
        )
        self.assertLess(abs(selected_center[0] - 110), 8)
        self.assertLess(abs(selected_center[1] - 84), 8)

    def test_small_box_visible_only_in_test_image_is_still_selected(self):
        reference = canvas()
        test = canvas()
        cv2.rectangle(reference, (26, 18), (194, 160), GREEN, 3)
        cv2.rectangle(test, (26, 18), (194, 160), GREEN, 3)
        cv2.rectangle(test, (95, 70), (124, 98), GREEN, 2)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"x": 26, "y": 18, "w": 169, "h": 143},
        )

        self.assertEqual(len(epicenters), 1)
        assert_box_near(self, epicenters[0], (95, 70, 30, 29), tolerance=5)

    def test_legacy_fallback_chooses_smallest_valid_box(self):
        reference = canvas()
        old_epicenters = [
            (25, 18, 170, 140),
            (94, 71, 31, 27),
        ]

        epicenters, focus_reference, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=old_epicenters,
            global_box_info={"x": 25, "y": 18, "w": 170, "h": 140},
        )

        self.assertEqual(epicenters[0], (94, 71, 31, 27))
        self.assertEqual(focus_reference.shape[:2], (27, 31))

    def test_global_fallback_preserves_absolute_position(self):
        reference = canvas()
        epicenters, focus_reference, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=[],
            global_box_info={"x": 68, "y": 49, "w": 52, "h": 44},
        )

        self.assertEqual(epicenters[0], (68, 49, 52, 44))
        self.assertEqual(focus_reference.shape[:2], (44, 52))


if __name__ == "__main__":
    unittest.main()
