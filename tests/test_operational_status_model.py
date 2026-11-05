"""Estado visual da AOI: conexão, análise e decisões são coisas distintas."""
import unittest

from src.ui.operational_status_model import (
    STATUS_COLORS, inspection_tone, last_decision_status, network_tone,
)


class StatusSemanticsTests(unittest.TestCase):
    def test_listening_is_not_connected(self):
        self.assertEqual(network_tone("Ouvindo AOI (Porta 5001)"), "waiting")
        self.assertEqual(network_tone("Conexão AOI: aguardando tráfego (porta 5001)"),
                         "waiting")
        self.assertNotEqual(network_tone("Porta aberta"), "connected")

    def test_live_network_and_failure_do_not_share_success_color(self):
        self.assertEqual(network_tone("AOI conectada, quadro recebido"), "connected")
        self.assertEqual(network_tone("AOI desconectada"), "error")
        self.assertEqual(network_tone("Erro de transporte"), "error")
        self.assertEqual(network_tone("Estado desconhecido"), "neutral")
        self.assertNotEqual(STATUS_COLORS["waiting"], STATUS_COLORS["connected"])

    def test_brain_waiting_not_processing_even_with_active_flag(self):
        self.assertEqual(inspection_tone("Aguardando próxima peça", is_active=True),
                         "waiting")
        self.assertEqual(inspection_tone("Analisando imagem", is_active=True),
                         "active")
        self.assertEqual(inspection_tone("Erro ao interpretar recorte", is_active=True),
                         "error")
        self.assertEqual(inspection_tone("Sistema ocioso", is_active=False),
                         "waiting")

    def test_auto_decision_is_not_human_confirmation(self):
        title, tone = last_decision_status("OK", "production_auto")
        self.assertIn("automática, sem confirmação física", title)
        self.assertEqual(tone, "ok")
        title, tone = last_decision_status("NG", "button")
        self.assertIn("operador ODIN", title)
        self.assertEqual(tone, "ng")

    def test_discard_does_not_claim_physical_quality(self):
        title, tone = last_decision_status(
            "OK (Concordou / Descartado)", "button"
        )
        self.assertTrue(title.startswith("Última ação"))
        self.assertEqual(tone, "neutral")
        title, tone = last_decision_status("UNKNOWN", "auto")
        self.assertTrue(title.startswith("Último registro"))
        self.assertEqual(tone, "neutral")


if __name__ == "__main__":
    unittest.main()
