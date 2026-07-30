import unittest
from pathlib import Path

from src.ui.production_confidence_gate import (
    PRODUCTION_AUTO_CONFIDENCE_THRESHOLD,
    normalized_confidence,
    production_decision_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class ProductionConfidencePolicyTests(unittest.TestCase):
    def test_exactly_99_percent_allows_automatic_decision(self):
        policy = production_decision_policy(
            {"confidence": 0.99, "is_defect": True}
        )
        self.assertEqual(PRODUCTION_AUTO_CONFIDENCE_THRESHOLD, 0.99)
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "NG")

    def test_below_99_percent_requires_operator(self):
        policy = production_decision_policy(
            {"confidence": 0.9899, "is_defect": False}
        )
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["operator_shortcuts"], {"0": "OK", "1": "NG"})

    def test_above_99_percent_remains_automatic(self):
        policy = production_decision_policy(
            {"confidence": 1.0, "is_defect": False}
        )
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "OK")

    def test_invalid_confidence_never_becomes_automatic(self):
        self.assertEqual(normalized_confidence({"confidence": "invalid"}), 0.0)
        self.assertFalse(
            production_decision_policy({"confidence": "invalid"})["auto_allowed"]
        )

    def test_confidence_is_clamped(self):
        self.assertEqual(normalized_confidence({"confidence": 2.0}), 1.0)
        self.assertEqual(normalized_confidence({"confidence": -1.0}), 0.0)


class ProductionConfidenceIntegrationContractTests(unittest.TestCase):
    def test_gate_is_installed_after_anomaly_learning(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_anomaly_learning(ControlPanel)"),
            source.index(
                "install_production_confidence_gate(ControlPanel, OperationalControlsPresenter)"
            ),
        )

    def test_low_confidence_blocks_auto_save_and_exposes_zero_one(self):
        source = (
            ROOT / "src" / "ui" / "production_confidence_gate.py"
        ).read_text(encoding="utf-8")
        self.assertIn('source == "auto"', source)
        self.assertIn('if not policy["auto_allowed"]', source)
        self.assertIn("self.production_review_pending = True", source)
        self.assertIn('"0 - Aprovar como OK"', source)
        self.assertIn('"1 - Confirmar defeito NG"', source)
        self.assertIn('{"0", "OK"}', source)
        self.assertIn('{"1", "NG"}', source)

    def test_production_hides_decision_buttons_until_review(self):
        source = (
            ROOT / "src" / "ui" / "production_confidence_gate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("decision_buttons_visible = (not is_production) or pending", source)
        self.assertIn("panel.btn_save_ok.setVisible(decision_buttons_visible)", source)
        self.assertIn("panel.btn_save_ng.setVisible(decision_buttons_visible)", source)


if __name__ == "__main__":
    unittest.main()
