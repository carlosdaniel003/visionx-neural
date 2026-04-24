# src/ui/widgets/semantic_dna.py
"""
O Visor de DNA Semântico (Embedding Debugger).
Lê os vetores (128 dimensões) criados pelo SemanticExpert.
Gera três barras visuais: Gabarito, Anomalia e Divergência.
"""
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QRectF

class SemanticDNAWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280) # Altura padrão dos Cards Horizontais do Carrossel
        self.setMinimumWidth(550)
        
        self.is_active = False
        self.sem_loss = 0.0
        self.query_emb = None
        self.ref_emb = None

    def update_data(self, detail: dict):
        """ Recebe o dict completo do Orquestrador """
        if not detail or "semantic_loss" not in detail:
            self.is_active = False
            self.update()
            return
            
        self.is_active = True
        self.sem_loss = detail.get("semantic_loss", 0.0)
        
        # Puxa as listas de números
        q_emb = detail.get("query_emb", [])
        r_emb = detail.get("ref_emb", [])
        
        self.query_emb = np.array(q_emb) if q_emb else None
        self.ref_emb = np.array(r_emb) if r_emb else None
        
        self.update()

    def _get_color_heat(self, val):
        """ Converte um valor (0.0 a 1.0) em uma cor térmica (Azul escuro -> Verde -> Amarelo) """
        val = max(0.0, min(1.0, val))
        r = int(min(255, max(0, 255 * (val * 2 - 1))))
        b = int(min(255, max(0, 255 * (2 - val * 2))))
        g = int(min(255, max(0, 255 * (1 - abs(val * 2 - 1)))))
        return QColor(r, g, b)

    def _get_color_diff(self, val):
        """ Converte a diferença em uma cor de alerta (Preto = Igual, Vermelho Vivo = Diferente) """
        val = max(0.0, min(1.0, val))
        return QColor(int(val * 255), 0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#161b22"))

        painter.setPen(QColor("#c9d1d9"))
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.drawText(10, 20, "DNA SEMÂNTICO (EMBEDDING BARCODE)")

        if not self.is_active or self.query_emb is None or self.ref_emb is None:
            painter.setPen(QColor("#484f58"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor Semântico Inativo (MoE ignorou rota)")
            painter.end()
            return

        # =========================================================
        # MATEMÁTICA DE NORMALIZAÇÃO DAS BARRAS
        # =========================================================
        emb_q = self.query_emb # Câmera
        emb_r = self.ref_emb   # Gabarito
        
        # Acha o maior e o menor número no vetor para poder pintar corretamente
        all_vals = np.concatenate([emb_q, emb_r])
        v_min, v_max = np.min(all_vals), np.max(all_vals)
        if v_max == v_min: v_max = v_min + 1e-5

        # Calcula a diferença absoluta entre o gabarito e a câmera em cada posição
        diff = np.abs(emb_q - emb_r)
        d_max = np.max(diff) if np.max(diff) > 0 else 1.0

        num_features = len(emb_q)
        
        padding_x = 10
        available_w = w - (padding_x * 2)
        bar_w = available_w / num_features
        
        # Calcula as alturas usando o espaço real do Widget
        title_h = 15
        spacing_y = 10
        total_used_y = 35 # O topo (título) + um espaço extra
        available_h = h - total_used_y - 20 # 20 de sobra no pé
        
        bar_h = (available_h - (title_h * 3) - (spacing_y * 2)) / 3

        font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font)

        # =========================================================
        # BARRA 1: GABARITO (REFERÊNCIA)
        # =========================================================
        y_offset = total_used_y
        painter.setPen(QColor("#8b949e"))
        painter.drawText(padding_x, int(y_offset + 10), "DNA: Gabarito (Padrão)")
        y_offset += title_h
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(emb_r):
            norm_val = (val - v_min) / (v_max - v_min)
            painter.setBrush(self._get_color_heat(norm_val))
            painter.drawRect(QRectF(padding_x + (i * bar_w), y_offset, bar_w + 1, bar_h))

        # =========================================================
        # BARRA 2: CÂMERA (PEÇA TESTADA)
        # =========================================================
        y_offset += bar_h + spacing_y
        painter.setPen(QColor("#8b949e"))
        painter.drawText(padding_x, int(y_offset + 10), "DNA: Câmera (Anomalia Reportada)")
        y_offset += title_h
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(emb_q):
            norm_val = (val - v_min) / (v_max - v_min)
            painter.setBrush(self._get_color_heat(norm_val))
            painter.drawRect(QRectF(padding_x + (i * bar_w), y_offset, bar_w + 1, bar_h))

        # =========================================================
        # BARRA 3: DIVERGÊNCIA (O ALARME DE DEFEITO)
        # =========================================================
        y_offset += bar_h + spacing_y
        painter.setPen(QColor("#ff7b72"))
        painter.drawText(padding_x, int(y_offset + 10), "Divergência (Foco de Alarme da IA)")
        y_offset += title_h
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(diff):
            norm_val = val / d_max # Aumenta o contraste das diferenças!
            painter.setBrush(self._get_color_diff(norm_val))
            painter.drawRect(QRectF(padding_x + (i * bar_w), y_offset, bar_w + 1, bar_h))

        # =========================================================
        # STATUS NUMÉRICO
        # =========================================================
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        is_critical = self.sem_loss > 0.40
        status_color = QColor("#ff7b72") if is_critical else QColor("#3fb950")
        
        status_text = f"DISTÂNCIA SEMÂNTICA: {self.sem_loss:.0%}"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.setPen(status_color)
        painter.drawText(int(w - padding_x - text_width), int(total_used_y + 10), status_text)

        painter.end()