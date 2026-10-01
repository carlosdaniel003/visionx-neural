import unittest

from src.core.experts.dual_scale_presence import DualScalePresenceAnalyzer


class DualScalePresenceGeometryTests(unittest.TestCase):
    def test_real_faltando_focus_is_small_fraction_of_global_component(self):
        ratio = DualScalePresenceAnalyzer._local_global_ratio(
            (215, 90, 140, 108),
            (286, 570, 3),
            {"w": 547, "h": 261},
        )

        self.assertAlmostEqual(ratio, 0.106, delta=0.002)
        self.assertLess(
            ratio,
            DualScalePresenceAnalyzer.MAX_LOCAL_GLOBAL_AREA_RATIO,
        )

    def test_context_expands_real_faltando_focus_without_leaving_crop(self):
        box = DualScalePresenceAnalyzer._context_box(
            (215, 90, 140, 108),
            (286, 570, 3),
            {"w": 547, "h": 261},
        )

        self.assertEqual(box, (110, 9, 350, 270))


class DualScalePresenceEvidenceTests(unittest.TestCase):
    def test_real_faltando_context_vector_confirms_absence(self):
        # Reproduzido do evento 912c92754da3433d8a2a0980052e2b78:
        # ROI local parecia conforme, mas a expansão contextual mostrou o
        # desaparecimento do corpo inteiro do componente.
        metrics = {
            "score": 0.7951582189074781,
            "coverage": 0.3879047619047619,
            "residual_mean": 0.6884945631027222,
            "residual_p90": 0.8399999737739563,
            "structure_loss": 0.3270075908909309,
            "edge_mismatch": 0.39177139945652173,
            "direct_similarity": 0.5621884657095184,
            "appearance_loss": 0.4378115342904816,
            "best_similarity": 0.1976355016231537,
        }
        support = {
            "supported": True,
            "structural": 0.56,
            "semantic": 0.56,
        }

        hard, reason = DualScalePresenceAnalyzer._context_hard_absence(
            metrics,
            support,
        )

        self.assertTrue(hard)
        self.assertIn("contexto maior", reason)

    def test_same_context_without_independent_physical_support_stays_ambiguous(self):
        metrics = {
            "score": 0.79,
            "coverage": 0.39,
            "residual_mean": 0.69,
            "appearance_loss": 0.44,
            "structure_loss": 0.33,
            "edge_mismatch": 0.39,
            "best_similarity": 0.20,
        }
        support = {
            "supported": False,
            "structural": 0.21,
            "semantic": 0.22,
        }

        hard, _reason = DualScalePresenceAnalyzer._context_hard_absence(
            metrics,
            support,
        )

        self.assertFalse(hard)

    def test_nearby_component_match_blocks_contextual_absence(self):
        metrics = {
            "score": 0.80,
            "coverage": 0.40,
            "residual_mean": 0.70,
            "appearance_loss": 0.45,
            "structure_loss": 0.35,
            "edge_mismatch": 0.40,
            "best_similarity": 0.62,
        }
        support = {
            "supported": True,
            "structural": 0.58,
            "semantic": 0.61,
        }

        hard, _reason = DualScalePresenceAnalyzer._context_hard_absence(
            metrics,
            support,
        )

        self.assertFalse(hard)

    def test_extreme_context_can_confirm_without_global_support(self):
        metrics = {
            "score": 0.92,
            "coverage": 0.62,
            "residual_mean": 0.72,
            "appearance_loss": 0.61,
            "structure_loss": 0.35,
            "edge_mismatch": 0.44,
            "best_similarity": 0.12,
        }
        support = {
            "supported": False,
            "structural": 0.20,
            "semantic": 0.25,
        }

        hard, reason = DualScalePresenceAnalyzer._context_hard_absence(
            metrics,
            support,
        )

        self.assertTrue(hard)
        self.assertIn("colapso visual extremo", reason)


if __name__ == "__main__":
    unittest.main()
