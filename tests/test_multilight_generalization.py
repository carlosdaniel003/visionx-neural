import unittest

import numpy as np

from src.core.multilight_fusion import fuse_multilight
from src.services.capture_debug_payload import decision_record
from src.services.image_archive_candidates import archive_image_candidates
from src.ui.network_xp_debug import (
    format_multilight_debug_report,
    multilight_copy_image_snapshot,
)


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

    def test_real_top_missing_false_positive_requires_review_when_three_ok_witnesses_disagree(self):
        analyses = {
            "SIDE": _analysis("SIDE", final_score=0.0, physical_score=0.88, is_defect=False),
            "TOP": _analysis("TOP", final_score=1.0, physical_score=1.0, is_defect=True),
            "MID": _analysis("MID", final_score=0.1316, physical_score=0.85, is_defect=False),
        }
        for mode, sim in (("SIDE", 0.9234), ("TOP", 0.9075), ("MID", 0.8976)):
            analyses[mode]["detail"]["best_match_label"] = "OK"
            analyses[mode]["detail"]["best_similarity"] = sim
        top = analyses["TOP"]["detail"]
        top["dominant_engine"] = "missing"
        top["missing_hard_absence"] = True
        top["missing_global_envelope_coarse_similarity"] = 0.77256
        top["missing_global_envelope_background_exposure"] = 0.0

        result = fuse_multilight(analyses, "FALTANDO")
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(result["is_defect"])
        self.assertTrue(result["production_review_required"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "multilight_missing_physical_disagreement",
        )
        self.assertTrue(result["detail"]["multilight_physical_disagreement"])
        self.assertIn("Revisão humana", result["reason"])
        self.assertTrue(result["detail"]["missing_hard_absence"])
        self.assertTrue(
            result["detail"]["decision_trace"]["raw_hard_missing_evidence"]
        )
        self.assertFalse(
            result["detail"]["decision_trace"]["hard_missing_evidence"]
        )
        debug = decision_record(result, {"category": "FALTANDO"})
        self.assertTrue(debug["raw_hard_missing_evidence"])
        self.assertFalse(debug["hard_missing_evidence"])
        self.assertTrue(debug["operator_review_required"])

    def test_real_single_top_missing_remains_ng_when_context_does_not_corrob_or_memory_is_weak(self):
        for context, ok_similarity in ((0.15, 0.91), (0.77, 0.62)):
            with self.subTest(context=context, ok_similarity=ok_similarity):
                analyses = {
                    "SIDE": _analysis("SIDE", final_score=0.0, physical_score=0.15, is_defect=False),
                    "TOP": _analysis("TOP", final_score=1.0, physical_score=1.0, is_defect=True),
                    "MID": _analysis("MID", final_score=0.02, physical_score=0.12, is_defect=False),
                }
                for mode in ("SIDE", "TOP", "MID"):
                    analyses[mode]["detail"]["best_match_label"] = "OK"
                    analyses[mode]["detail"]["best_similarity"] = ok_similarity
                top = analyses["TOP"]["detail"]
                top["dominant_engine"] = "missing"
                top["missing_hard_absence"] = True
                top["missing_global_envelope_coarse_similarity"] = context
                top["missing_global_envelope_background_exposure"] = 0.0

                result = fuse_multilight(analyses, "FALTANDO")
                self.assertEqual(result["verdict"], "DEFEITO REAL")
                self.assertEqual(result["detail"]["fusion_rule"], "multilight_strong_single")
                self.assertFalse(result["detail"]["multilight_physical_disagreement"])

    def test_missing_disagreement_does_not_change_other_categories(self):
        analyses = {
            "SIDE": _analysis("SIDE", final_score=0.0, physical_score=0.12, is_defect=False),
            "TOP": _analysis("TOP", final_score=1.0, physical_score=1.0, is_defect=True),
            "MID": _analysis("MID", final_score=0.1, physical_score=0.14, is_defect=False),
        }
        for mode in analyses:
            analyses[mode]["detail"]["best_match_label"] = "OK"
            analyses[mode]["detail"]["best_similarity"] = 0.94
        top = analyses["TOP"]["detail"]
        top.update({
            "dominant_engine": "missing",
            "missing_hard_absence": True,
            "missing_global_envelope_coarse_similarity": 0.85,
            "missing_global_envelope_background_exposure": 0.0,
        })
        result = fuse_multilight(analyses, "DESLOCADO")
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(result["detail"]["fusion_rule"], "multilight_strong_single")

    def test_event_691e95_side_false_hard_missing_has_independent_witnesses(self):
        analyses = {
            "SIDE": _analysis("SIDE", final_score=1.0, physical_score=1.0, is_defect=True),
            "TOP": _analysis("TOP", final_score=0.0, physical_score=1.0, is_defect=False),
            "MID": _analysis("MID", final_score=0.0, physical_score=0.0, is_defect=False),
        }
        for mode, sim in (("SIDE", .9124937), ("TOP", .9259532), ("MID", .9414898)):
            analyses[mode]["detail"].update({"best_match_label": "OK", "best_similarity": sim})
        analyses["SIDE"]["detail"].update({
            "dominant_engine": "missing",
            "missing_hard_absence": True,
            "missing_global_envelope_coarse_similarity": .528,
            "missing_global_envelope_background_exposure": .2156,
        })
        analyses["TOP"]["detail"].update({
            "dominant_engine": "knn", "missing_hard_absence": False,
            "missing_global_envelope_support": True,
            "missing_active": True, "missing_is_defect": True,
            "missing_score": 1.0, "missing_tolerance": .36,
        })
        analyses["MID"]["detail"].update({
            "dominant_engine": "knn", "missing_hard_absence": False,
            "missing_active": True, "missing_is_defect": False,
            "missing_score": .12, "missing_tolerance": .36,
        })
        fused = fuse_multilight(analyses, "FALTANDO")
        self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(fused["is_defect"])
        self.assertTrue(fused["production_review_required"])
        self.assertEqual(fused["detail"]["fusion_rule"], "multilight_missing_physical_disagreement")
        self.assertEqual(fused["detail"]["multilight_dominant_mode"], "SIDE")
        self.assertEqual(fused["detail"]["multilight_dominant_local_engine"], "missing")
        self.assertTrue(fused["detail"]["decision_trace"]["raw_hard_missing_evidence"])
        self.assertFalse(fused["detail"]["decision_trace"]["hard_missing_evidence"])
        self.assertEqual(fused["detail"]["decision_trace"]["weights"], {"physical": 0.0, "knn": 0.0})
        self.assertIn("SIDE sinalizou", fused["reason"])

    def test_side_hard_missing_without_independent_witness_stays_defect(self):
        analyses = {
            "SIDE": _analysis("SIDE", final_score=1., physical_score=1., is_defect=True),
            "TOP": _analysis("TOP", final_score=0., physical_score=.9, is_defect=False),
            "MID": _analysis("MID", final_score=0., physical_score=.02, is_defect=False),
        }
        for mode in analyses:
            analyses[mode]["detail"].update({"best_match_label": "OK", "best_similarity": .93})
        analyses["SIDE"]["detail"].update({
            "dominant_engine": "missing", "missing_hard_absence": True,
        })
        analyses["TOP"]["detail"].update({
            "missing_global_envelope_support": True, "missing_hard_absence": False,
        })
        analyses["MID"]["detail"].update({
            "missing_active": True, "missing_is_defect": True,
            "missing_score": .87, "missing_tolerance": .36,
        })
        fused = fuse_multilight(analyses, "FALTANDO")
        self.assertEqual(fused["verdict"], "DEFEITO REAL")
        self.assertEqual(fused["detail"]["fusion_rule"], "multilight_strong_single")

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

    def test_adhesive_keeps_specialized_fusion_policy(self):
        def adhesive(mode, score, physical, defect):
            return {
                "lighting_mode": mode,
                "verdict": "DEFEITO REAL" if defect else "FALHA FALSA",
                "is_defect": bool(defect),
                "confidence": 0.90,
                "active_engines": ["ssim_expert.py", "knn_expert.py"],
                "detail": {
                    "adhesive_score": float(score),
                    "adhesive_is_defect": bool(defect),
                    "adhesive_tolerance": 0.32,
                    "physical_score": float(physical),
                    "best_match_label": "NG" if defect else "OK",
                    "best_similarity": 0.90,
                    "adhesive_reason": f"adesivo {mode}",
                    "decision_trace": {
                        "operator_review_required": False,
                    },
                },
            }

        analyses = {
            "SIDE": adhesive("SIDE", 0.05, 0.20, False),
            "TOP": adhesive("TOP", 0.88, 0.90, True),
            "MID": adhesive("MID", 0.10, 0.20, False),
        }

        result = fuse_multilight(analyses, "MUITO ADESIVO")

        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "adhesive_multilight_strong_auxiliary",
        )
        self.assertEqual(
            result["detail"]["multilight_dominant_mode"],
            "TOP",
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


class GeneralMultiLightDebugTests(unittest.TestCase):
    class Panel:
        pass

    def _panel(self):
        panel = self.Panel()
        event_id = "evt-debug-general"
        panel.capture_debug_last_record = {
            "event_id": event_id,
            "source": "windows_xp",
            "validation": {"valid": True},
        }
        panel.adhesive_multilight_last_event_id = event_id
        panel.adhesive_multilight_last_category = "FALTANDO"
        panel.adhesive_multilight_last_analyses = {
            mode: _analysis(
                mode,
                final_score=score,
                physical_score=score,
                is_defect=mode == "TOP",
            )
            for mode, score in (
                ("SIDE", 0.20),
                ("TOP", 0.91),
                ("MID", 0.18),
            )
        }
        panel.adhesive_multilight_last_final_analysis = fuse_multilight(
            panel.adhesive_multilight_last_analyses,
            "FALTANDO",
        )
        panel.adhesive_multilight_last_source_frames = {
            "SIDE": np.full((10, 20, 3), (10, 20, 30), dtype=np.uint8),
            "TOP": np.full((10, 24, 3), (40, 50, 60), dtype=np.uint8),
            "MID": np.full((10, 22, 3), (70, 80, 90), dtype=np.uint8),
        }
        return panel

    def test_non_adhesive_debug_contains_three_independent_analyses(self):
        report = format_multilight_debug_report(self._panel())

        self.assertIn("ANÁLISES MULTILIGHT - FALTANDO", report)
        self.assertIn("Categoria multilight: FALTANDO", report)
        self.assertIn("ILUMINAÇÃO SIDE", report)
        self.assertIn("ILUMINAÇÃO TOP", report)
        self.assertIn("ILUMINAÇÃO MID", report)
        self.assertIn("JULGAMENTO FINAL MULTILIGHT", report)
        self.assertIn("Iluminação dominante: TOP", report)

    def test_non_adhesive_copy_image_snapshot_is_side_top_mid_composite(self):
        composite = multilight_copy_image_snapshot(self._panel())

        self.assertIsNotNone(composite)
        self.assertEqual(composite.shape, (54, 82, 3))
        self.assertTrue(
            np.array_equal(composite[44, 0], np.array([10, 20, 30]))
        )
        self.assertTrue(
            np.array_equal(composite[44, 28], np.array([40, 50, 60]))
        )
        self.assertTrue(
            np.array_equal(composite[44, 60], np.array([70, 80, 90]))
        )


if __name__ == "__main__":
    unittest.main()
