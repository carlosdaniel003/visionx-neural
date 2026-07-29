import unittest

import cv2
import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor


RGB_GREEN_SAMPLES = (
    (22, 149, 21),
    (15, 124, 13),
    (86, 188, 105),
    (60, 202, 74),
    (12, 203, 12),
    (153, 246, 133),
    (88, 203, 66),
)
BGR_GREEN_SAMPLES = tuple((blue, green, red) for red, green, blue in RGB_GREEN_SAMPLES)


def scene(width=260, height=220):
    # Placa verde natural deliberadamente próxima da faixa antiga.
    image = np.full((height, width, 3), (74, 103, 69), dtype=np.uint8)
    cv2.rectangle(image, (38, 24), (221, 196), (38, 42, 46), -1)
    cv2.rectangle(image, (62, 43), (198, 181), (52, 57, 61), -1)
    return image


def multicolor_rectangle(image, box, thickness=2):
    x, y, width, height = box
    colors = BGR_GREEN_SAMPLES
    cv2.line(image, (x, y), (x + width - 1, y), colors[0], thickness)
    cv2.line(
        image,
        (x, y + height - 1),
        (x + width - 1, y + height - 1),
        colors[3],
        thickness,
    )
    cv2.line(image, (x, y), (x, y + height - 1), colors[5], thickness)
    cv2.line(
        image,
        (x + width - 1, y),
        (x + width - 1, y + height - 1),
        colors[6],
        thickness,
    )


def near(actual, expected, tolerance=7):
    return all(abs(int(a) - int(e)) <= tolerance for a, e in zip(actual, expected))


class EpicenterGreenSignatureTests(unittest.TestCase):
    def test_all_supplied_green_samples_are_accepted(self):
        image = np.zeros((40, len(BGR_GREEN_SAMPLES) * 12, 3), dtype=np.uint8)
        for index, color in enumerate(BGR_GREEN_SAMPLES):
            image[:, index * 12 : (index + 1) * 12] = color
        mask = EpicenterExtractor._green_mask(image)
        for index in range(len(BGR_GREEN_SAMPLES)):
            patch = mask[:, index * 12 : (index + 1) * 12]
            self.assertGreater(float(np.mean(patch > 0)), 0.75)

    def test_gabarito_green_roi_beats_colored_test_annotation(self):
        reference = scene()
        test = scene()
        outer = (48, 31, 165, 157)
        expected = (86, 139, 67, 34)
        multicolor_rectangle(reference, outer, 3)
        multicolor_rectangle(reference, expected, 2)
        multicolor_rectangle(test, outer, 3)

        # A indicação NG no teste pode estar em outra posição e em outra cor.
        cv2.rectangle(test, (169, 57), (207, 96), (0, 210, 255), 2)
        cv2.rectangle(test, (168, 47), (220, 58), (0, 140, 255), -1)
        cv2.rectangle(test, (87, 139), (153, 173), (255, 70, 20), 2)

        epicenters, focus_reference, focus_test = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[expected, (169, 57, 39, 40)],
            global_box_info={"x": outer[0], "y": outer[1], "w": outer[2], "h": outer[3]},
        )

        self.assertEqual(len(epicenters), 1)
        self.assertTrue(near(epicenters[0], expected, tolerance=8))
        self.assertEqual(focus_reference.shape, focus_test.shape)
        self.assertLess(epicenters[0][1], 180)

    def test_filled_green_board_regions_are_not_frames(self):
        reference = scene()
        test = scene()
        outer = (45, 27, 171, 165)
        expected = (75, 145, 74, 30)
        multicolor_rectangle(reference, outer, 3)
        multicolor_rectangle(reference, expected, 2)
        multicolor_rectangle(test, outer, 3)

        # Manchas da placa e reflexos com cores próximas dos exemplos fornecidos.
        cv2.rectangle(reference, (92, 55), (143, 108), BGR_GREEN_SAMPLES[2], -1)
        cv2.circle(reference, (179, 123), 13, BGR_GREEN_SAMPLES[4], -1)
        cv2.line(reference, (64, 82), (201, 82), BGR_GREEN_SAMPLES[1], 4)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[expected, (92, 55, 52, 54), (166, 110, 27, 27)],
            global_box_info={"x": outer[0], "y": outer[1], "w": outer[2], "h": outer[3]},
        )

        self.assertEqual(len(epicenters), 1)
        self.assertTrue(near(epicenters[0], expected, tolerance=8))
        self.assertGreater(epicenters[0][2], 55)
        self.assertLess(epicenters[0][3], 45)

    def test_no_test_candidate_can_override_reference_roi(self):
        reference = scene()
        test = scene()
        outer = (43, 25, 174, 169)
        expected = (71, 132, 61, 39)
        distracting = (159, 48, 46, 72)
        multicolor_rectangle(reference, outer, 3)
        multicolor_rectangle(reference, expected, 2)
        multicolor_rectangle(test, outer, 3)
        multicolor_rectangle(test, distracting, 3)

        epicenters, _, _ = EpicenterExtractor.extract_focus(
            reference,
            test,
            old_epicenters=[expected, distracting],
            global_box_info={"x": outer[0], "y": outer[1], "w": outer[2], "h": outer[3]},
        )

        self.assertEqual(len(epicenters), 1)
        self.assertTrue(near(epicenters[0], expected, tolerance=8))
        self.assertFalse(near(epicenters[0], distracting, tolerance=8))


if __name__ == "__main__":
    unittest.main()
