# src\ui\hud_window.py
"""
Módulo responsável pela interface overlay (HUD) transparente.
Desenha o alvo (Verde) e as anomalias detectadas (Vermelho).
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from src.config.settings import settings

class HUDWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.target_rect = None
        self.anomalies = [] # Lista para guardar as coordenadas dos defeitos
        self.log_text = "VisionX Neural: Inicializando..."
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

    def update_anomalies(self, anomalies_list: list):
        """Recebe a lista de anomalias e converte para objetos QRect."""
        self.anomalies = [QRect(x, y, w, h) for (x, y, w, h) in anomalies_list]
        self.update()

    def clear_target(self):
        if self.target_rect is not None:
            self.target_rect = None
            self.anomalies = [] # Limpa as anomalias se perder o alvo principal
            self.update()

    def update_log(self, message: str):
        self.log_text = message
        self.update()

    def paintEvent(self, event):
        if self.target_rect is None and not self.log_text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Desenha o Log de Radar no topo esquerdo
        painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        painter.fillRect(40, 30, 600, 35, QColor(0, 0, 0, 150))
        painter.setPen(QPen(QColor(0, 255, 255)))
        painter.drawText(50, 55, self.log_text)

        # 2. Desenha o Retângulo Principal (Alvo Verde)
        if self.target_rect is not None:
            r_ok, g_ok, b_ok = settings.HUD_BORDER_COLOR_OK
            pen_ok = QPen(QColor(r_ok, g_ok, b_ok))
            pen_ok.setWidth(settings.HUD_BORDER_THICKNESS)
            painter.setPen(pen_ok)
            painter.drawRect(self.target_rect)

            # 3. Desenha as Anomalias (Retângulos Vermelhos menores)
            if self.anomalies:
                r_ng, g_ng, b_ng = settings.HUD_BORDER_COLOR_NG
                pen_ng = QPen(QColor(r_ng, g_ng, b_ng))
                pen_ng.setWidth(2) # Borda mais fina para os detalhes
                painter.setPen(pen_ng)
                for anomaly_rect in self.anomalies:
                    painter.drawRect(anomaly_rect)