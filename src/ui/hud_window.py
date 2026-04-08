# C:\Users\cdaniel\visionx-neural\src\ui\hud_window.py
"""
Módulo responsável pela interface overlay (HUD) transparente.
Não contém lógica de detecção, apenas reage a sinais para desenhar na tela.
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from src.config.settings import settings

class HUDWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.target_rect = None
        self.log_text = "VisionX Neural: Inicializando..." # Variável do Log na tela
        self._setup_window()

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showMaximized()

    def update_target(self, x: int, y: int, w: int, h: int):
        self.target_rect = QRect(x, y, w, h)
        self.update()

    def clear_target(self):
        if self.target_rect is not None:
            self.target_rect = None
            self.update()

    def update_log(self, message: str):
        """Atualiza a mensagem de texto flutuante no topo da tela."""
        self.log_text = message
        self.update() # Força o redesenho da tela para atualizar o texto

    def paintEvent(self, event):
        if self.target_rect is None and not self.log_text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Desenha o Log de Radar no topo esquerdo da tela
        painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        # Fundo escuro sutil para o texto
        painter.fillRect(40, 30, 600, 35, QColor(0, 0, 0, 150))
        # Texto em Ciano
        painter.setPen(QPen(QColor(0, 255, 255)))
        painter.drawText(50, 55, self.log_text)

        # 2. Desenha o Retângulo de Detecção (se houver alvo)
        if self.target_rect is not None:
            r, g, b = settings.HUD_BORDER_COLOR_OK
            pen = QPen(QColor(r, g, b))
            pen.setWidth(settings.HUD_BORDER_THICKNESS)
            painter.setPen(pen)
            painter.drawRect(self.target_rect)