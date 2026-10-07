import unittest
from pathlib import Path

from src.ui.production_confidence_gate import (
    PRODUCTION_AUTO_CONFIDENCE_THRESHOLD,
    normalized_confidence,
    production_decision_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class ProductionPolicyTests(unittest.TestCase):
    def test_false_failure_allows_auto_ok_without_99_percent_gate(self):
        policy = production_decision_policy(
            {
                "confidence": 0.42,
                "is_defect": False,
                "verdict": "FALHA FALSA",
            }
        )

        self.assertIsNone(PRODUCTION_AUTO_CONFIDENCE_THRESHOLD)
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertIsNone(policy["threshold"])

    def test_real_defect_always_requires_operator_in_v1(self):
        policy = production_decision_policy(
            {
                "confidence": 1.0,
                "is_defect": True,
                "verdict": "DEFEITO REAL",
            }
        )

        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertEqual(policy["verdict"], "DEFEITO REAL")

    def test_mandatory_review_never_sends_zero_or_one_automatically(self):
        policy = production_decision_policy(
            {
                "confidence": 1.0,
                "is_defect": False,
                "verdict": "REVISÃO OBRIGATÓRIA",
                "production_review_required": True,
            }
        )

        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(policy["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(
            policy["operator_shortcuts"],
            {"0": "OK", "1": "NG"},
        )

    def test_legacy_analysis_without_verdict_uses_is_defect_fallback(self):
        self.assertEqual(
            production_decision_policy(
                {"is_defect": False, "confidence": 0.1}
            )["verdict"],
            "FALHA FALSA",
        )
        self.assertEqual(
            production_decision_policy(
                {"is_defect": True, "confidence": 0.1}
            )["verdict"],
            "DEFEITO REAL",
        )

    def test_confidence_remains_telemetry_only_and_is_clamped(self):
        self.assertEqual(normalized_confidence({"confidence": 2.0}), 1.0)
        self.assertEqual(normalized_confidence({"confidence": -1.0}), 0.0)
        self.assertEqual(
            normalized_confidence({"confidence": "invalid"}),
            0.0,
        )


class ProductionGateIntegrationContractTests(unittest.TestCase):
    @staticmethod
    def gate_source() -> str:
        return (
            ROOT / "src" / "ui" / "production_confidence_gate.py"
        ).read_text(encoding="utf-8")

    def test_gate_is_installed_after_anomaly_learning(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_anomaly_learning(ControlPanel)"),
            source.index(
                "install_production_confidence_gate(ControlPanel, OperationalControlsPresenter)"
            ),
        )

    def test_manual_review_freezes_capture_and_exposes_zero_one(self):
        source = self.gate_source()
        self.assertIn("def enter_production_review", source)
        self.assertIn("self.production_review_pending = True", source)
        self.assertIn('"0 - Aprovar como OK"', source)
        self.assertIn('"1 - Confirmar defeito NG"', source)
        self.assertIn('command in {"0", "OK"}', source)
        self.assertIn('command in {"1", "NG"}', source)
        self.assertIn('"INTERVENÇÃO NECESSÁRIA"', source)

    def test_production_hides_decision_buttons_until_intervention(self):
        source = self.gate_source()
        self.assertIn(
            "decision_buttons_visible = (not is_production) or pending",
            source,
        )
        self.assertIn(
            "panel.btn_save_ok.setVisible(decision_buttons_visible)",
            source,
        )
        self.assertIn(
            "panel.btn_save_ng.setVisible(decision_buttons_visible)",
            source,
        )

    def test_confidence_threshold_is_not_used_to_allow_auto_ok(self):
        source = self.gate_source()
        self.assertIn("PRODUCTION_AUTO_CONFIDENCE_THRESHOLD = None", source)
        self.assertIn('verdict == "FALHA FALSA"', source)
        self.assertNotIn("confidence + 1e-12 >=", source)

    def test_real_defect_preserves_its_verdict_during_manual_wait(self):
        source = self.gate_source()
        self.assertIn(
            '== "REVISÃO OBRIGATÓRIA"',
            source,
        )
        self.assertIn('analysis["production_review_required"]', source)

    def test_human_resolution_notifies_autonomy_controller(self):
        source = self.gate_source()
        self.assertIn(
            '"production_operator_decision_completed"',
            source,
        )
        self.assertIn("callback(user_decision, source=source)", source)
        self.assertIn(
            'and not bool(getattr(self, "is_locked", False))',
            source,
        )


if __name__ == "__main__":
    unittest.main()
