# src/ui/widgets/shift_debugger.py
import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QPointF

class ShiftDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)
        
        self.is_active = False
        self.dx = 0.0
        self.dy = 0.0
        self.shift_pixels = 0.0
        self.shift_pct = 0.0
        self.tolerance = 0.08
        self.is_defect = False

    def update_data(self, detail: dict):
        """ Recebe os detalhes da análise do ShiftExpert """
        # Verifica se o motor de Shift rodou nesta peça
        if not detail or "shift_pct" not in detail:
            self.is_active = False
            self.update()
            return
            
        self.is_active = True
        self.dx = detail.get("dx", 0.0)
        self.dy = detail.get("dy", 0.0)
        self.shift_pixels = detail.get("shift_pixels", 0.0)
        self.shift_pct = detail.get("shift_pct", 0.0)
        self.tolerance = detail.get("tolerance", 0.08)
        self.is_defect = detail.get("is_defect", False)
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Fundo Escuro
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))
        
        # Título
        painter.setPen(QColor("#dddddd"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(5, 15, "Telemetria de Deslocamento (Shift)")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            font = QFont("Consolas", 10, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor Shift Inativo (MoE)")
            painter.end()
            return
            
        cx = w / 2.0
        cy = (h / 2.0) + 10

        # --- 1. Desenha a Grade / Mira Radar ---
        painter.setPen(QPen(QColor("#333333"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(0, cy), QPointF(w, cy)) # Eixo X
        painter.drawLine(QPointF(cx, 0), QPointF(cx, h)) # Eixo Y
        
        # Desenha o limite de tolerância (Círculo)
        max_visual_radius = min(cx, cy) - 20
        # O raio da tolerância é proporcional a tolerancia máxima (ex: 8% de 15%)
        tol_radius = (self.tolerance / 0.15) * max_visual_radius
        
        painter.setPen(QPen(QColor("#555555"), 1, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), tol_radius, tol_radius)

        # Identifica se é uma anomalia estrutural gigante (Passo 0 do Expert)
        is_macro_anomaly = self.is_defect and self.dx == 0.0 and self.dy == 0.0 and self.shift_pct > 0.15

        vec_color = QColor("#ff5555") if self.is_defect else QColor("#55ff55")

        if is_macro_anomaly:
            # --- 2A. Desenha o Alerta de Anomalia Estrutural (Gross Error) ---
            # Escala visual dinâmica: 50% de anomalia já preenche o radar inteiro
            escala_visual = min(1.0, self.shift_pct / 0.50)
            anomaly_radius = max_visual_radius * escala_visual
            
            painter.setPen(QPen(QColor("#ff5555"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 85, 85, 60)) # Fundo vermelho translúcido
            painter.drawEllipse(QPointF(cx, cy), anomaly_radius, anomaly_radius)
            
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            text_rect = painter.boundingRect(0, 0, w, int(h/2), Qt.AlignmentFlag.AlignCenter, "ALERTA MACRO:\nDano\nEstrutural")
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "ALERTA MACRO:\nDano\nEstrutural")
            
        else:
            # --- 2B. Desenha o Vetor de Deslocamento Padrão ---
            # Amplifica o dx e dy visualmente (15% = borda da tela)
            visual_dx = (self.dx / 15.0) * max_visual_radius if self.shift_pct > 0 else 0
            visual_dy = (self.dy / 15.0) * max_visual_radius if self.shift_pct > 0 else 0
            
            target_x = cx + visual_dx
            target_y = cy + visual_dy
            
            # Linha do vetor
            painter.setPen(QPen(vec_color, 2, Qt.PenStyle.SolidLine))
            painter.drawLine(QPointF(cx, cy), QPointF(target_x, target_y))
            
            # Ponto de impacto
            painter.setBrush(vec_color)
            painter.drawEllipse(QPointF(target_x, target_y), 4, 4)
            
            # Ponto central (Gabarito)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), 2, 2)

        # --- 3. HUD de Informações ---
        painter.setFont(QFont("Consolas", 7))
        
        # Info Tolerância
        painter.setPen(QColor("#888888"))
        painter.drawText(5, h - 20, f"Max Tol: {self.tolerance:.1%}")
        
        if is_macro_anomaly:
            # Info Status Atual (Macro)
            painter.setPen(vec_color)
            painter.drawText(5, h - 5, f"Área Alterada: {self.shift_pct:.1%}")
        else:
            # Info Status Atual (Vetor)
            painter.setPen(vec_color)
            painter.drawText(5, h - 5, f"Shift: {self.shift_pct:.1%} ({self.shift_pixels}px)")
            
            # Info Eixos
            painter.setPen(QColor("#aaaaaa"))
            axes_text = f"X:{self.dx:.1f} Y:{self.dy:.1f}"
            axes_width = painter.fontMetrics().horizontalAdvance(axes_text)
            painter.drawText(int(w - axes_width - 5), int(h - 5), axes_text)

        painter.end()