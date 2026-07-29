import unittest

import cv2
import numpy as np

from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.roi_input_contract import install_roi_input_contract


class ContractOrchestrator:
    def __init__(self):
        self.experts = {
            "ssim": SSIMExpert(),
            "silk": SilkExpert(),
        }

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        x, y, width, height = aoi_epicenters[0]
        reference = full_gab[y : y + height, x : x + width].copy()
        test = full_test[y : y + height, x : x + width].copy()
        ssim_result = self.experts["ssim"].analyze(
            reference,
            test,
            full_gab,
            full_test,
            x,
            y,
            width,
            height,
            canonical_focus=True,
            focus_box=(x, y, width, height),
        )
        silk_result = self.experts["silk"].analyze(
            full_gab,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )
        detail = {}
        detail.update(ssim_result)
        detail.update(silk_result)
        detail.update(
            {
                "semantic_focus_box": (x, y, width, height),
                "semantic_focus_test": test.copy(),
            }
        )
        return {
            "is_defect": True,
            "active_engines": [
                "ssim_expert.py",
                "silk_expert.py",
                "semantic_expert.py",
            ],
            "detail": detail,
        }


install_roi_input_contract(ContractOrchestrator, SSIMExpert, SilkExpert)


def inspection_pair():
    reference = np.full((120, 180, 3), (32, 38, 44), dtype=np.uint8)
    test = reference.copy()
    cv2.rectangle(reference, (45, 28), (142, 98), (44, 105, 184), -1)
    cv2.rectangle(test, (45, 28), (142, 98), (44, 105, 184), -1)
    cv2.circle(reference, (92, 64), 24, (25, 25, 28), -1)
    cv2.circle(test, (78, 69), 25, (25, 25, 28), -1)
    cv2.line(reference, (104, 45), (135, 51), (230, 230, 225), 6)
    cv2.line(test, (89, 50), (137, 59), (230, 230, 225), 6)
    return reference, test


class ROIInputContractTests(unittest.TestCase):
    def test_all_raw_inputs_match_canonical_test_roi(self):
        reference, test = inspection_pair()
        box = (52, 34, 88, 60)
        orchestrator = ContractOrchestrator()
        analysis = orchestrator.inspect(
            reference,
            test,
            raw_anomalies=[box],
            aoi_info={"category": "FALTANDO"},
            global_box_info={},
            aoi_epicenters=[box],
        )

        self.assertTrue(analysis["roi_consistent"])
        audit = analysis["detail"]["roi_audit"]
        self.assertEqual(audit["canonical_box"], list(box))
        self.assertTrue(audit["all_boxes_match"])
        self.assertTrue(audit["all_raw_inputs_match"])
        self.assertTrue(
            audit["engines"]["texture_ssim"]["test_input"]["exact_match"]
        )
        self.assertTrue(
            audit["engines"]["structural_xor"]["test_input"]["exact_match"]
        )
        self.assertTrue(
            audit["engines"]["semantic"]["test_input"]["exact_match"]
        )

    def test_missing_category_never_uses_adhesive_heatmap(self):
        reference = np.full((64, 96, 3), (35, 75, 160), dtype=np.uint8)
        test = reference.copy()
        test[20:55, 38:75] = (20, 35, 105)
        expert = SSIMExpert()
        expert._visionx_current_category = "FALTANDO"
        result = expert.analyze(
            reference,
            test,
            reference,
            test,
            0,
            0,
            96,
            64,
            canonical_focus=True,
            focus_box=(0, 0, 96, 64),
        )

        self.assertEqual(result["ssim_input_category"], "FALTANDO")
        self.assertEqual(result["heat_map_mode"], "generic_ssim")
        self.assertEqual(result["adhesive_evidence"], 0.0)
        self.assertEqual(result["adhesive_coverage"], 0.0)

    def test_structural_debug_keeps_raw_roi_and_aligned_copy_separate(self):
        reference, test = inspection_pair()
        box = (52, 34, 88, 60)
        expert = SilkExpert()
        result = expert.analyze(
            reference,
            test,
            {},
            {"category": "FALTANDO"},
            [box],
        )
        expected_test = test[34:94, 52:140]

        self.assertEqual(tuple(result["roi_box"]), box)
        self.assertTrue(np.array_equal(result["roi_test_raw"], expected_test))
        self.assertEqual(result["roi_test_aligned"].shape, expected_test.shape)
        self.assertIn("dx", result)
        self.assertIn("dy", result)


if __name__ == "__main__":
    unittest.main()
