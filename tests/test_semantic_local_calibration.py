import unittest

from src.core.experts.semantic_calibration import (
    calibrate_semantic_result,
    semantic_evidence_components,
)


def build_debug(peak, top_values, grid_mean, dominant):
    grid = [[grid_mean for _ in range(4)] for _ in range(4)]
    grid[3][2] = peak
    return {
        "spatial": {
            "peak_cell": {"row": 3, "column": 2, "value": peak},
            "top_cells": [
                {"combined_delta": value}
                for value in top_values
            ],
            "combined_delta_grid": grid,
        },
        "groups": {
            "edge_density": {"relative_divergence": dominant},
            "brightness": {"relative_divergence": dominant * 0.7},
            "hue_histogram": {"relative_divergence": dominant * 0.8},
        },
    }


class SemanticLocalCalibrationTests(unittest.TestCase):
    def test_localized_defect_can_cross_threshold_despite_low_global_loss(self):
        debug = build_debug(
            peak=0.85,
            top_values=[0.85, 0.62, 0.55],
            grid_mean=0.12,
            dominant=0.68,
        )
        evidence = semantic_evidence_components(0.27, debug)

        self.assertGreater(evidence["calibrated_score"], 0.45)
        self.assertGreater(evidence["calibrated_score"], 0.27)
        self.assertGreater(evidence["local_evidence"], 0.60)

    def test_mild_local_noise_remains_below_threshold(self):
        debug = build_debug(
            peak=0.25,
            top_values=[0.25, 0.15, 0.10],
            grid_mean=0.12,
            dominant=0.20,
        )
        evidence = semantic_evidence_components(0.05, debug)

        self.assertLess(evidence["calibrated_score"], 0.45)

    def test_identical_embeddings_remain_zero(self):
        debug = build_debug(
            peak=0.0,
            top_values=[0.0, 0.0, 0.0],
            grid_mean=0.0,
            dominant=0.0,
        )
        evidence = semantic_evidence_components(0.0, debug)

        self.assertEqual(evidence["calibrated_score"], 0.0)
        self.assertEqual(evidence["local_evidence"], 0.0)

    def test_result_exposes_global_and_local_scores(self):
        result = {
            "semantic_loss": 0.27,
            "score": 0.27,
            "is_defect": False,
            "semantic_debug": build_debug(
                peak=0.85,
                top_values=[0.85, 0.62, 0.55],
                grid_mean=0.12,
                dominant=0.68,
            ),
        }

        calibrated = calibrate_semantic_result(result)

        self.assertTrue(calibrated["is_defect"])
        self.assertEqual(calibrated["semantic_global_loss"], 0.27)
        self.assertGreater(calibrated["semantic_local_evidence"], 0.60)
        self.assertIn("calibration", calibrated["semantic_debug"])
        self.assertIn("global", calibrated["reason"])
        self.assertIn("local", calibrated["reason"])


if __name__ == "__main__":
    unittest.main()
