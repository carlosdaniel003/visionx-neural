"""Reexibição de PNG do acervo no MSS: registration != par RGB exato."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.services.archive_rendered_source_match import (
    compare_rendered_source, match_rendered_archive,
)
from src.ui.neural_telemetry_model import memory_panel_text, memory_seen_state
from src.ui.inspection_memory_feedback import memory_feedback_state
from src.services.capture_debug_payload import decision_record


def original_frame(seed=4):
    rng = np.random.default_rng(seed)
    img = np.full((840, 1165, 3), 70, dtype=np.uint8)
    cv2.rectangle(img, (100, 200), (1065, 800), (14, 22, 22), -1)
    for _ in range(120):
        x = int(rng.integers(120, 1040))
        y = int(rng.integers(230, 780))
        col = tuple(map(int, rng.integers(20, 245, size=3)))
        cv2.circle(img, (x, y), int(rng.integers(3, 13)), col, -1)
    for i in range(30):
        cv2.putText(
            img, f"X{i:02d}",
            (int(rng.integers(110, 920)), int(rng.integers(225, 780))),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 130, 40), 1,
        )
    return img


def displayed(frame):
    screen = np.full((1080, 1920, 3), 218, dtype=np.uint8)
    width = 1110
    height = round(frame.shape[0] * width / frame.shape[1])
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    screen[110:110+height, 450:450+width] = resized
    return screen


class ArchiveRenderedSourceTests(unittest.TestCase):
    def test_scaled_archived_png_is_identified_without_pixel_exact_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ok = root / "public" / "ok_archive"
            ng = root / "public" / "ng_archive"
            ok.mkdir(parents=True)
            ng.mkdir(parents=True)
            filename = "2026-10-02_1029_FALTANDO.png"
            original = original_frame()
            self.assertTrue(cv2.imwrite(str(ok / filename), original))
            screen = displayed(original)
            result = match_rendered_archive(
                screen, "FALTANDO", ["public/ok_archive/" + filename],
                root=root,
            )
            self.assertEqual(result["status"], "RENDERED_ARCHIVE_SOURCE")
            self.assertEqual(result["label"], "OK")
            self.assertFalse(result["exact_aoi_pair"])
            self.assertFalse(result["frame_pixel_identical"])
            self.assertEqual(result["path"], "public/ok_archive/" + filename)
            self.assertGreaterEqual(result["evidence"]["inliers"], 18)

            decision = {
                "is_defect": True,
                "verdict": "DEFEITO REAL",
                "detail": {
                    "recognition_route": "NEW_CNN",
                    "cnn_v3_primary": True,
                    "cnn_v3_status": "INFERENCE_OK",
                    "cnn_v3_ng_score_uncalibrated": 1.0,
                    "side_archive_visual_presence": {
                        "status": "NO_SIMILAR",
                        "rendered_archive_match": result,
                    },
                },
            }
            memo = memory_seen_state(decision)
            self.assertEqual(memo["status"], "JA_VI")
            self.assertFalse(memo["exact_pair_seen"])
            self.assertFalse(memo["archive_exact_frame_seen"])
            self.assertTrue(memo["archive_rendered_source_seen"])
            self.assertIn(filename, memory_panel_text(decision)[1])
            self.assertIn("ARQUIVO REEXIBIDO", memory_feedback_state(decision)[0])
            self.assertTrue(decision["is_defect"])  # Nenhuma decisão fabricada.
            self.assertEqual(decision_record(decision)["cnn_v3"]["ng_score_uncalibrated"], 1.0)

    def test_focal_change_cannot_reuse_archived_source(self):
        original = original_frame()
        changed = displayed(original)
        cv2.rectangle(changed, (850, 410), (930, 485), (250, 250, 250), -1)
        self.assertIsNone(compare_rendered_source(original, changed))

    def test_unrelated_frame_not_equivalent(self):
        self.assertIsNone(compare_rendered_source(
            original_frame(4), displayed(original_frame(90)),
        ))

    def test_wrong_category_and_source_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ok = root / "public" / "ok_archive"
            ok.mkdir(parents=True)
            frame = original_frame()
            self.assertTrue(cv2.imwrite(
                str(ok / "2026-10-02_1029_FALTANDO.png"), frame,
            ))
            screen = displayed(frame)
            relative = "public/ok_archive/2026-10-02_1029_FALTANDO.png"
            self.assertEqual(
                match_rendered_archive(screen, "INVERTIDO", [relative], root=root)["status"],
                "NO_MATCH",
            )
            self.assertEqual(
                match_rendered_archive(screen, "FALTANDO", ["../wrong.png"], root=root)["status"],
                "NO_MATCH",
            )

    def test_conflicting_archive_labels_do_not_choose_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ok = root / "public" / "ok_archive"
            ng = root / "public" / "ng_archive"
            ok.mkdir(parents=True)
            ng.mkdir(parents=True)
            frame = original_frame()
            filename = "2026-10-02_1029_FALTANDO.png"
            cv2.imwrite(str(ok / filename), frame)
            cv2.imwrite(str(ng / filename), frame)
            matched = match_rendered_archive(
                displayed(frame), "FALTANDO",
                ["public/ok_archive/" + filename,
                 "public/ng_archive/" + filename],
                root=root,
            )
            self.assertEqual(matched["status"], "CONFLICT")
            self.assertEqual(matched["label"], "")


if __name__ == "__main__":
    unittest.main()
