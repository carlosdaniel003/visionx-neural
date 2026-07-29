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

    def test_green_reflections_and_text_do_not_beat_real_frame(self):
        reference = draw_nested_rectangles(
            canvas(width=305, height=516),
            outer=(35, 33, 240, 465),
            inner=(106, 378, 98, 51),
            outer_thickness=3,
            inner_thickness=2,
        )
        test = reference.copy()

        # Simulam reflexos, letras e pequenos blobs verdes dentro do componente.
        cv2.rectangle(reference, (63, 42), (158, 76), GREEN, -1)
        cv2.rectangle(test, (119, 34), (132, 43), GREEN, -1)
        cv2.rectangle(test, (137, 412), (143, 418), GREEN, -1)
        cv2.line(reference, (120, 457), (144, 457), GREEN, 5)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[(106, 378, 98, 51)],
            global_box_info={"x": 35, "y": 33, "w": 240, "h": 465},
        )

        self.assertEqual(len(epicenters), 1)
        selected = epicenters[0]
        assert_box_near(self, selected, (106, 378, 98, 51), tolerance=6)
        self.assertGreater(selected[2], 80)
        self.assertGreater(selected[3], 35)
        self.assertEqual(focus_reference.shape, focus_test.shape)

    def test_largest_reliable_inner_frame_beats_tiny_deep_blob(self):
        reference = draw_nested_rectangles(
            canvas(width=305, height=516),
            outer=(35, 33, 240, 465),
            inner=(106, 376, 96, 49),
        )
        test = draw_nested_rectangles(
            canvas(width=305, height=516),
            outer=(35, 26, 240, 464),
            inner=(106, 378, 98, 51),
        )
        cv2.rectangle(test, (136, 411), (144, 419), GREEN, -1)
        cv2.rectangle(test, (119, 34), (133, 43), GREEN, -1)

        epicenters, _, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"x": 35, "y": 33, "w": 240, "h": 465},
        )

        self.assertEqual(len(epicenters), 1)
        assert_box_near(self, epicenters[0], (106, 377, 97, 50), tolerance=7)
        self.assertGreater(focus_test.shape[1], 80)
        self.assertGreater(focus_test.shape[0], 35)

    def test_content_crop_removes_green_frame(self):
        reference = draw_nested_rectangles(
            canvas(),
            outer=(20, 15, 180, 150),
            inner=(80, 60, 62, 45),
            outer_thickness=3,
            inner_thickness=3,
        )
        test = reference.copy()
        test[70:95, 94:128] = (220, 220, 220)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[],
            global_box_info={"x": 20, "y": 15, "w": 180, "h": 150},
        )

        self.assertEqual(len(epicenters), 1)
        green_reference = EpicenterExtractor._green_mask(focus_reference)
        green_test = EpicenterExtractor._green_mask(focus_test)
        self.assertLess(float(np.mean(green_reference > 0)), 0.08)
        self.assertLess(float(np.mean(green_test > 0)), 0.08)
        self.assertGreater(float(np.mean(focus_test)), float(np.mean(focus_reference)))

    def test_legacy_fallback_chooses_largest_valid_inner_box(self):
        reference = canvas()
        old_epicenters = [
            (94, 71, 31, 27),
            (105, 75, 12, 11),
        ]

        epicenters, focus_reference, _ = EpicenterExtractor.extract_focus(
            reference,
            reference.copy(),
            old_epicenters=old_epicenters,
            global_box_info={"x": 25, "y": 18, "w": 170, "h": 140},
        )

        assert_box_near(self, epicenters[0], (94, 71, 31, 27), tolerance=3)
        self.assertGreater(focus_reference.shape[1], 20)
        self.assertGreater(focus_reference.shape[0], 18)

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
