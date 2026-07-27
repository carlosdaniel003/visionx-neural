import unittest

from src.ui.iconography_model import (
    ALL_ICON_NAMES,
    sanitize_visual_text,
    status_icon_name,
)


class SvgIconographyModelTests(unittest.TestCase):
    def test_visual_text_removes_emoji_and_keeps_words(self):
        message = "\U0001f9e0 Processando \u26a0\ufe0f imagem \u2705"
        self.assertEqual(
            sanitize_visual_text(message),
            "Processando imagem",
        )

    def test_visual_text_removes_joined_and_directional_symbols(self):
        message = "\U0001f4be  Última peça   OK  \u2192 pronta"
        self.assertEqual(
            sanitize_visual_text(message),
            "Última peça OK → pronta",
        )

    def test_status_icons_follow_message_semantics(self):
        self.assertEqual(status_icon_name("network", "Conectado"), "network")
        self.assertEqual(status_icon_name("network", "Erro de conexão"), "defect")
        self.assertEqual(status_icon_name("brain", "Processando", True), "processor")
        self.assertEqual(status_icon_name("brain", "Iluminação TOP"), "light-right")
        self.assertEqual(status_icon_name("brain", "Dataset limpo"), "database")
        self.assertEqual(status_icon_name("history", "OK salvo"), "approve")
        self.assertEqual(status_icon_name("history", "NG confirmado"), "defect")

    def test_expected_icon_catalog_is_complete(self):
        expected = {
            "capture",
            "discard",
            "approve",
            "defect",
            "light-left",
            "light-down",
            "light-right",
            "database-delete",
            "network",
            "processor",
            "history",
            "idle",
            "warning",
            "database",
        }
        self.assertEqual(ALL_ICON_NAMES, expected)


if __name__ == "__main__":
    unittest.main()
