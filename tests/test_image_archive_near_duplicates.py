"""Regressão da seleção visual de imagens novas no arquivo OK/NG."""
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.services.image_archive_dedup import (
    ArchiveVisualIndex, archive_category_from_filename,
    archive_visual_descriptor, near_duplicate_visual,
)
from src.services.ng_image_archive import NGImageArchiveQueue
from src.services.ok_image_archive import OKImageArchiveQueue


def screenshot(seed=1) -> np.ndarray:
    """Padrão AOI sintético; nunca arquivar imagem real de produção em testes."""
    img = np.full((840, 1165, 3), 186, dtype=np.uint8)
    cv2.rectangle(img, (15, 250), (585, 830), (24, 24, 24), -1)
    cv2.rectangle(img, (596, 250), (1158, 830), (22, 22, 22), -1)
    for offset in (0, 590):
        for x in (70, 340):
            cv2.rectangle(img, (x+offset, 330), (x+offset+160, 460), (30, 80, 220), -1)
            cv2.rectangle(img, (x+offset, 650), (x+offset+160, 810), (30, 80, 220), -1)
        cv2.rectangle(img, (155+offset, 510), (340+offset, 640), (16, 16, 16), -1)
        cv2.rectangle(img, (30+offset, 280), (530+offset, 818), (50, 210, 30), 2)
    cv2.putText(img, "FALTANDO", (80, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 2)
    if seed:
        noise = np.random.default_rng(seed).integers(
            -7, 8, size=img.shape, dtype=np.int16
        )
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class ArchiveVisualNearDuplicateTests(unittest.TestCase):
    def test_very_close_screens_are_similar_not_byte_identical(self):
        a, b = screenshot(1), screenshot(2)
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(near_duplicate_visual(
            archive_visual_descriptor(a), archive_visual_descriptor(b)
        ))

    def test_changed_component_focal_region_is_not_suppressed(self):
        original = screenshot(1)
        changed = original.copy()
        cv2.rectangle(changed, (670, 490), (1000, 700), (235, 235, 235), -1)
        self.assertFalse(near_duplicate_visual(
            archive_visual_descriptor(original), archive_visual_descriptor(changed)
        ))

    def test_invalid_small_or_uniform_image_never_claimed_visually_similar(self):
        self.assertIsNone(archive_visual_descriptor(np.zeros((50, 80, 3), dtype=np.uint8)))
        self.assertFalse(near_duplicate_visual(None, None))
        uniform = np.zeros((840, 1165, 3), dtype=np.uint8)
        self.assertFalse(near_duplicate_visual(
            archive_visual_descriptor(uniform), archive_visual_descriptor(uniform)
        ))

    def test_archive_name_suffix_and_lighting_scope(self):
        self.assertEqual(archive_category_from_filename(
            Path("2026-10-21_1600_FALTANDO_3.png")), "FALTANDO_SIDE")
        self.assertEqual(archive_category_from_filename(
            Path("2026-10-21_1600_FALTANDO_TOP_2.png")), "FALTANDO_TOP")
        self.assertEqual(archive_category_from_filename(
            Path("legacy.png")), "")

    def test_index_requires_same_category_and_lighting(self):
        a, b = screenshot(1), screenshot(2)
        index = ArchiveVisualIndex()
        index.register(a, "FALTANDO_SIDE")
        self.assertTrue(index.contains_similar(b, "FALTANDO_SIDE"))
        self.assertTrue(index.contains_similar(b, "FALTANDO"))
        self.assertFalse(index.contains_similar(b, "FALTANDO_TOP"))
        self.assertFalse(index.contains_similar(b, "INVERTIDO_SIDE"))


class ImageArchiveQueueVisualTests(unittest.TestCase):
    def test_ng_and_ok_workers_skip_close_events_and_preserve_distinct(self):
        for queue_type in (NGImageArchiveQueue, OKImageArchiveQueue):
            with self.subTest(queue=queue_type.__name__), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                worker = queue_type(output)
                stamp = datetime(2026, 10, 21, 16, 0)
                worker.submit(screenshot(1), "FALTANDO_SIDE", stamp)
                worker.submit(screenshot(2), "FALTANDO_SIDE", stamp.replace(minute=1))
                worker.wait_until_idle()
                self.assertEqual(len(list(output.glob("*.png"))), 1)

                changed = screenshot(1)
                cv2.rectangle(changed, (670, 490), (1000, 700), (235, 235, 235), -1)
                worker.submit(changed, "FALTANDO_SIDE", stamp.replace(minute=2))
                worker.wait_until_idle()
                self.assertEqual(len(list(output.glob("*.png"))), 2)

    def test_restart_sees_legacy_category_and_blocks_close_new_event(self):
        for queue_type in (NGImageArchiveQueue, OKImageArchiveQueue):
            with self.subTest(queue=queue_type.__name__), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                self.assertTrue(cv2.imwrite(
                    str(folder / "2026-10-21_1600_FALTANDO_3.png"), screenshot(1)
                ))
                worker = queue_type(folder)
                worker.submit(screenshot(2), "FALTANDO_SIDE",
                              datetime(2026, 10, 21, 16, 8))
                worker.wait_until_idle()
                self.assertEqual(len(list(folder.glob("*.png"))), 1)

    def test_similar_pixels_with_different_category_or_lighting_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            queue = NGImageArchiveQueue(folder)
            stamp = datetime(2026, 10, 21, 16, 5)
            for i, category in enumerate(("FALTANDO_SIDE", "INVERTIDO_SIDE",
                                          "FALTANDO_TOP")):
                queue.submit(screenshot(i + 1), category, stamp)
            queue.wait_until_idle()
            self.assertEqual(len(list(folder.glob("*.png"))), 3)

    def test_ng_and_ok_are_never_compared_across_independent_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            ng = NGImageArchiveQueue(folder / "ng")
            ok = OKImageArchiveQueue(folder / "ok")
            for queue in (ng, ok):
                queue.submit(screenshot(1), "FALTANDO_SIDE",
                             datetime(2026, 10, 21, 16, 0))
                queue.wait_until_idle()
            self.assertEqual(len(list((folder / "ng").glob("*.png"))), 1)
            self.assertEqual(len(list((folder / "ok").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
