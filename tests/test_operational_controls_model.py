import unittest

from src.ui.operational_controls_model import (
    available_action_count,
    operational_state,
)


class OperationalControlsModelTests(unittest.TestCase):
    def test_idle_test_mode_exposes_capture_lights_and_dataset_admin(self):
        state = operational_state(
            mode="Modo Teste",
            is_locked=False,
            has_analysis=False,
        )

        self.assertEqual(state.name, "idle")
        self.assertTrue(state.capture_enabled)
        self.assertTrue(state.lighting_enabled)
        self.assertTrue(state.dataset_clear_enabled)
        self.assertFalse(state.approve_enabled)
        self.assertFalse(state.reject_enabled)
        self.assertEqual(available_action_count(state), 5)

    def test_processing_blocks_every_control(self):
        state = operational_state(
            mode="Modo Teste",
            is_locked=True,
            has_analysis=False,
        )

        self.assertEqual(state.name, "processing")
        self.assertEqual(available_action_count(state), 0)
        self.assertFalse(state.capture_enabled)
        self.assertFalse(state.lighting_enabled)

    def test_test_review_exposes_operator_decisions(self):
        state = operational_state(
            mode="Modo Teste",
            is_locked=True,
            has_analysis=True,
        )

        self.assertEqual(state.name, "review_test")
        self.assertTrue(state.capture_enabled)
        self.assertTrue(state.discard_enabled)
        self.assertTrue(state.approve_enabled)
        self.assertTrue(state.reject_enabled)
        self.assertFalse(state.dataset_clear_enabled)
        self.assertEqual(available_action_count(state), 7)

    def test_shadow_review_keeps_screen_decisions_blocked(self):
        state = operational_state(
            mode="Modo Sombra",
            is_locked=True,
            has_analysis=True,
        )

        self.assertEqual(state.name, "review_shadow")
        self.assertTrue(state.discard_enabled)
        self.assertFalse(state.approve_enabled)
        self.assertFalse(state.reject_enabled)
        self.assertFalse(state.dataset_clear_enabled)

    def test_production_auto_blocks_manual_interference(self):
        state = operational_state(
            mode="Modo Produção",
            is_locked=True,
            has_analysis=True,
        )

        self.assertEqual(state.name, "production_auto")
        self.assertEqual(available_action_count(state), 0)

    def test_dataset_clear_is_never_available_outside_test_mode(self):
        for mode in ("Modo Produção", "Modo Sombra"):
            state = operational_state(
                mode=mode,
                is_locked=False,
                has_analysis=False,
            )
            self.assertFalse(state.dataset_clear_enabled)


if __name__ == "__main__":
    unittest.main()
