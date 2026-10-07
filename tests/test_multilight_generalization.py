import unittest

import numpy as np

from src.core.multilight_fusion import fuse_multilight
from src.services.image_archive_candidates import archive_image_candidates


def _analysis(
    mode,
    *,
    final_score,
    physical_score,
    is_defect,
    review=False,
):
    return {
        "lighting_mode": mode,
        "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
        "is_defect": bool(is_defect),
        "confidence": 0.88,
        "active_engines": ["ssim_expert.py", "knn_expert.py"],
        "detail": {
            "final_score": float(final_score),
            "physical_score": float(physical_score),
            "fusion_rule": "local",
            "dominant_engine": "ssim",
            "operator_review_required": bool(review),
            "decision_trace": {
                "operator_review_required": bool(review),
            },
        },
    }


class GeneralMultiLightFusionTests(unittest.TestCase):
    def test_strong_single_light_can_confirm_non_adhesive_defect(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                final_score=0.20,
                physical_score=0.30,
                is_defect=False,
            ),
            "TOP": _analysis(
                "TOP",
                final_score=0.91,
                physical_score=0.88,
                is_defect=True,
            ),
            "MID": _analysis(
                "MID",
                final_score=0.18,
                physical_score=0.24,
                is_defect=False,
            ),
        }

        result = fuse_multilight(analyses, "FALTANDO")

        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertTrue(result["is_defect"])
        self.assertTrue(result["multilight_final"])
        self.assertEqual(result["multilight_category"], "FALTANDO")
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "multilight_strong_single",
        )
        self.assertEqual(
            result["detail"]["multilight_dominant_mode"],
            "TOP",
        )

    def test_single_moderate_positive_requires_review(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                final_score=0.30,
                physical_score=0.25,
                is_defect=False,
            ),
            "TOP": _analysis(
                "TOP",
                final_score=0.67,
                physical_score=0.61,
                is_defect=True,
            ),
            "MID": _analysis(
                "MID",
                final_score=0.24,
                physical_score=0.20,
                is_defect=False,
            ),
        }

        result = fuse_multilight(analyses, "DESLOCADO")

        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(result["is_defect"])
        self.assertTrue(result["production_review_required"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "multilight_single_positive_review",
        )

    def test_two_positive_lights_are_corroborated(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                final_score=0.62,
                physical_score=0.58,
                is_defect=True,
            ),
            "TOP": _analysis(
                "TOP",
                final_score=0.64,
                physical_score=0.55,
                is_defect=True,
            ),
            "MID": _analysis(
                "MID",
                final_score=0.22,
                physical_score=0.20,
                is_defect=False,
            ),
        }

        result = fuse_multilight(analyses, "INVERTIDO")

        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "multilight_corroborated",
        )

    def test_three_clear_lights_are_false_failure(self):
        analyses = {
            mode: _analysis(
                mode,
                final_score=0.20,
                physical_score=0.18,
                is_defect=False,
            )
            for mode in ("SIDE", "TOP", "MID")
        }

        result = fuse_multilight(analyses, "FALTANDO")

        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertFalse(result["is_defect"])
        self.assertFalse(result["production_review_required"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "multilight_all_clear",
        )


class GeneralMultiLightArchiveTests(unittest.TestCase):
    class Panel:
        pass

    def _panel(self):
        panel = self.Panel()
        panel.adhesive_multilight_last_event_id = "evt-general-001"
        panel.adhesive_multilight_last_source_frames = {
            "SIDE": np.full(
                (10, 12, 3),
                (10, 20, 30),
                dtype=np.uint8,
            ),
            "TOP": np.full(
                (10, 12, 3),
                (40, 50, 60),
                dtype=np.uint8,
            ),
            "MID": np.full(
                (10, 12, 3),
                (70, 80, 90),
                dtype=np.uint8,
            ),
        }
        return panel

    def test_non_adhesive_multilight_returns_three_separate_archive_images(self):
        panel = self._panel()
        primary = np.full(
            (10, 12, 3),
            (1, 2, 3),
            dtype=np.uint8,
        )

        resolved = archive_image_candidates(
            panel,
            event_id="evt-general-001",
            category="FALTANDO",
            primary_image=primary,
        )

        self.assertEqual(len(resolved), 3)
        self.assertEqual(
            [category for _image, category in resolved],
            ["FALTANDO_SIDE", "FALTANDO_TOP", "FALTANDO_MID"],
        )
        self.assertTrue(
            np.array_equal(
                resolved[0][0],
                panel.adhesive_multilight_last_source_frames["SIDE"],
            )
        )
        self.assertTrue(
            np.array_equal(
                resolved[1][0],
                panel.adhesive_multilight_last_source_frames["TOP"],
            )
        )
        self.assertTrue(
            np.array_equal(
                resolved[2][0],
                panel.adhesive_multilight_last_source_frames["MID"],
            )
        )

    def test_incomplete_multilight_never_archives_partial_three_image_set(self):
        panel = self._panel()
        panel.adhesive_multilight_last_source_frames.pop("MID")
        primary = np.full(
            (10, 12, 3),
            (1, 2, 3),
            dtype=np.uint8,
        )

        resolved = archive_image_candidates(
            panel,
            event_id="evt-general-001",
            category="FALTANDO",
            primary_image=primary,
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0][1], "FALTANDO")
        self.assertTrue(np.array_equal(resolved[0][0], primary))


if __name__ == "__main__":
    unittest.main()
