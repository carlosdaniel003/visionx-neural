import unittest
from unittest.mock import patch

import numpy as np

import src.core.adhesive_multilight_analysis as module
from src.core.adhesive_multilight_analysis import (
    analyze_lighting,
    build_lighting_context,
)


class _FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        self.calls.append(
            {
                "aoi_info": dict(aoi_info),
                "raw_anomalies": list(raw_anomalies),
                "global_box_info": dict(global_box_info),
                "aoi_epicenters": list(aoi_epicenters),
            }
        )
        return {
            "is_defect": True,
            "confidence": 0.8,
            "verdict": "DEFEITO REAL",
            "active_engines": ["ssim_expert.py"],
            "detail": {"local_score": 0.7},
        }


class AdhesiveLightingContextTests(unittest.TestCase):
    def test_context_reuses_same_geometry_for_visual_and_experts(self):
        sample = np.full((100, 120, 3), 30, dtype=np.uint8)
        test = sample.copy()
        large_box = {
            "x": 10, "y": 15, "w": 80, "h": 70, "detected": True
        }
        small_box = (30, 35, 25, 20)
        focus_gab = sample[35:55, 30:55].copy()
        focus_ng = test[35:55, 30:55].copy()

        with patch.object(
            module,
            "detect_anomalies",
            return_value=(
                [small_box],
                [small_box],
                large_box,
                np.array([]),
                np.array([]),
            ),
        ), patch.object(
            module.EpicenterExtractor,
            "extract_focus",
            return_value=([small_box], focus_gab, focus_ng),
        ):
            context = build_lighting_context(sample, test)

        self.assertTrue(context["valid"])
        self.assertEqual(context["global_box_info"], large_box)
        self.assertEqual(context["real_epicenters"], [small_box])
        self.assertTrue(np.array_equal(context["focus_ng"], focus_ng))


class AdhesiveLightingExpertAnalysisTests(unittest.TestCase):
    def test_auxiliary_analysis_is_marked_non_final(self):
        sample = np.full((100, 120, 3), 30, dtype=np.uint8)
        test = sample.copy()
        info = {
            "category": "MUITO ADESIVO",
            "value": "MUCH ADHESIVE",
            "board": "BOARD-1",
        }
        context = {
            "valid": True,
            "raw_anomalies": [(1, 2, 3, 4)],
            "global_box_info": {
                "x": 1, "y": 2, "w": 3, "h": 4, "detected": True
            },
            "real_epicenters": [(1, 2, 3, 4)],
        }
        orchestrator = _FakeOrchestrator()

        result = analyze_lighting(
            orchestrator,
            sample,
            test,
            info,
            "TOP",
            context=context,
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["lighting_mode"], "TOP")
        self.assertTrue(result["multilight_visual_analysis"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertEqual(result["detail"]["lighting_mode"], "TOP")
        self.assertFalse(
            result["detail"]["eligible_for_final_decision"]
        )
        self.assertEqual(info.get("lighting_mode"), None)
        self.assertEqual(
            orchestrator.calls[0]["aoi_info"]["category"],
            "MUITO ADESIVO",
        )
        self.assertEqual(
            orchestrator.calls[0]["aoi_info"]["lighting_mode"],
            "TOP",
        )

    def test_top_and_mid_produce_separate_analysis_objects(self):
        sample = np.full((80, 80, 3), 40, dtype=np.uint8)
        test = sample.copy()
        context = {
            "valid": True,
            "raw_anomalies": [],
            "global_box_info": {},
            "real_epicenters": [],
        }
        orchestrator = _FakeOrchestrator()

        top = analyze_lighting(
            orchestrator, sample, test, {}, "TOP", context=context
        )
        mid = analyze_lighting(
            orchestrator, sample, test, {}, "MID", context=context
        )

        self.assertIsNot(top, mid)
        self.assertEqual(top["lighting_mode"], "TOP")
        self.assertEqual(mid["lighting_mode"], "MID")
        self.assertEqual(len(orchestrator.calls), 2)


if __name__ == "__main__":
    unittest.main()
