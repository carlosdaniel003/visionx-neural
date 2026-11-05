"""Regressão: MSS devolvia gabarito com 600px e teste com 459px.

A comparação usa limites medidos da interface AOI nos dois painéis,
não rótulos de arquivo, scores CNN nem uma altura fixa.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.services.aoi_paired_photo_geometry import reconcile_paired_photos
from src.services.screen_monitor import ScreenMonitor


def aoi_frame(*, missing_blue_gray_border: bool) -> np.ndarray:
    frame = np.full((1080, 1920, 3), (200, 200, 200), dtype=np.uint8)
    blue = (101, 130, 521, 20)
    red = (781, 130, 522, 20)
    bx, by, bw, bh = blue
    rx, ry, rw, rh = red
    cv2.rectangle(frame, (bx, by), (bx+bw-1, by+bh-1), (255, 0, 0), -1)
    cv2.rectangle(frame, (rx, ry), (rx+rw-1, ry+rh-1), (0, 0, 255), -1)

    # A mesma fotografia real tem 459 px nos dois painéis.
    frame[by+bh:by+bh+459, bx:bx+bw] = (17, 88, 44)
    frame[ry+rh:ry+rh+459, rx:rx+rw] = (48, 38, 91)
    # Em uma captura exibida, o cinza inferior do gabarito pode não
    # passar na máscara (ou ter sido ocultado pela janela).
    if missing_blue_gray_border:
        frame[by+bh+459:by+bh+600, bx:bx+bw] = (70, 70, 70)
    return frame


class GeometryReconciliationTests(unittest.TestCase):
    def test_recover_gabarito_from_sibling_bottom_without_hardcoded_height(self):
        monitor = ScreenMonitor()
        monitor._replay_no_debug = True
        observed = []
        monitor.layout_detected.connect(
            lambda sample, test, info: observed.append(
                (sample.copy(), test.copy(), info)
            )
        )
        original = aoi_frame(missing_blue_gray_border=False)
        displayed = aoi_frame(missing_blue_gray_border=True)
        with patch.object(
            monitor, "_extract_text_info",
            return_value={"category": "FALTANDO"},
        ):
            monitor.process_external_image(original)
            self.assertEqual(len(observed), 1)
            base_sample, base_test, _ = observed.pop()
            self.assertEqual(base_sample.shape, (459, 521, 3))
            self.assertEqual(base_test.shape, (459, 522, 3))

            monitor.process_external_image(displayed)
            self.assertEqual(len(observed), 1)
            sample, test, _ = observed.pop()
        self.assertEqual(sample.shape, base_sample.shape)
        self.assertEqual(test.shape, base_test.shape)
        self.assertTrue(np.array_equal(sample, base_sample))
        self.assertTrue(np.array_equal(test, base_test))
        self.assertEqual(
            monitor.last_crop_geometry["status"],
            "ALIGNED_FROM_SIBLING_BOUNDARY",
        )
        self.assertEqual(
            monitor.last_crop_geometry["corrected_panels"], ["SAMPLE"],
        )

    def test_same_algorithm_with_other_panel_missing_boundary(self):
        frame = aoi_frame(missing_blue_gray_border=False)
        known = frame[150:609, 101:622].copy()
        images = {"SAMPLE": known, "TEST": frame[150:750, 781:1303].copy()}
        geometries = {
            "SAMPLE": {
                "boundary_source": "DETECTED",
                "bar_bottom": 150,
                "x1": 101, "x2": 622,
                "relative_top": 0, "relative_bottom": 459,
            },
            "TEST": {
                "boundary_source": "UNBOUNDED",
                "bar_bottom": 150,
                "x1": 781, "x2": 1303,
            },
        }
        result, audit = reconcile_paired_photos(frame, images, geometries)
        self.assertEqual(result["TEST"].shape, (459, 522, 3))
        self.assertEqual(audit["corrected_panels"], ["TEST"])

    def test_no_valid_source_boundary_keeps_original_without_fabricating(self):
        frame = aoi_frame(missing_blue_gray_border=True)
        original = {
            "SAMPLE": frame[150:750, 101:622].copy(),
            "TEST": frame[150:750, 781:1303].copy(),
        }
        result, audit = reconcile_paired_photos(
            frame, original, {
                "SAMPLE": {"boundary_source": "UNBOUNDED"},
                "TEST": {"boundary_source": "UNBOUNDED"},
            },
        )
        self.assertEqual(audit["status"], "BOTH_UNBOUNDED")
        self.assertTrue(np.array_equal(result["SAMPLE"], original["SAMPLE"]))
        self.assertEqual(result["TEST"].shape[0], 600)


if __name__ == "__main__":
    unittest.main()
