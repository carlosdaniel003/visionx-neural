import unittest

import numpy as np

from src.core.moe_orchestrator import MoEOrchestrator


class RecordingSSIM:
    def __init__(self):
        self.calls = []

    def analyze(self, crop_gab, crop_test, *args, **kwargs):
        self.calls.append((crop_gab.copy(), crop_test.copy(), args, kwargs))
        return {
            "local_score": 0.8,
            "ctx_score": 0.4,
            "ssim": 0.2,
            "pct_changed": 0.7,
            "global_boxes": [],
            "crop_gab": crop_gab,
            "crop_test": crop_test,
            "focus_source": "epicenter_extractor",
        }


class MoESSIMFocusRoutingTests(unittest.TestCase):
    def test_orchestrator_uses_exact_epicenter_instead_of_best_raw_anomaly(self):
        orchestrator = MoEOrchestrator.__new__(MoEOrchestrator)
        recorder = RecordingSSIM()
        orchestrator.experts = {"ssim": recorder}
        orchestrator.routing_table = {"Synthetic": ["ssim"]}

        full_gab = np.zeros((80, 100, 3), dtype=np.uint8)
        full_test = np.zeros_like(full_gab)
        for row in range(80):
            full_gab[row, :, 0] = row
            full_test[row, :, 0] = row
        full_test[20:45, 30:70, 2] = 255

        epicenter = (30, 20, 40, 25)
        raw_anomalies = [(2, 3, 8, 9), (75, 60, 10, 10)]

        result = orchestrator.inspect(
            full_gab,
            full_test,
            raw_anomalies,
            {"category": "Synthetic"},
            {"w": 100, "h": 80},
            [epicenter],
        )

        self.assertEqual(len(recorder.calls), 1)
        crop_gab, crop_test, _args, kwargs = recorder.calls[0]
        self.assertTrue(np.array_equal(crop_gab, full_gab[20:45, 30:70]))
        self.assertTrue(np.array_equal(crop_test, full_test[20:45, 30:70]))
        self.assertTrue(kwargs["canonical_focus"])
        self.assertEqual(kwargs["focus_box"], epicenter)
        self.assertEqual(result["all_boxes"]["ssim_local"], epicenter)
        self.assertEqual(result["detail"]["focus_source"], "epicenter_extractor")


if __name__ == "__main__":
    unittest.main()
