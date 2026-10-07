import unittest
from pathlib import Path


class OperationalControlsVisualContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("src/ui/operational_controls.py").read_text(
            encoding="utf-8"
        )

    def test_each_operational_role_has_a_specific_selector(self):
        for selector in (
            "captureActionButton",
            "discardActionButton",
            "approveActionButton",
            "rejectActionButton",
            "cameraLightButton",
            "datasetDangerButton",
        ):
            self.assertIn(selector, self.source)

    def test_visual_contract_contains_hover_pressed_and_disabled_states(self):
        self.assertIn(":hover", self.source)
        self.assertIn(":pressed", self.source)
        self.assertIn(":disabled", self.source)

    def test_active_lighting_and_state_bar_are_explicit(self):
        self.assertIn('activeLight="true"', self.source)
        self.assertIn("operationStateBar", self.source)
        self.assertIn("operationStateBadge", self.source)

    def test_state_bar_has_compact_and_wide_reflow(self):
        self.assertIn("QGridLayout", self.source)
        self.assertIn("def apply_responsive_layout", self.source)
        self.assertIn("if compact:", self.source)
        self.assertIn("layout.addWidget(hint, 1, 0, 1, 2)", self.source)
        self.assertIn("layout.addWidget(hint, 0, 1)", self.source)


if __name__ == "__main__":
    unittest.main()
