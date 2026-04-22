# src/ui/widgets/semantic_dna.py
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QRectF

class SemanticDNAWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query_emb = None
        self.ref_emb = None
        # Altura mínima para caber as 3 barras confortavelmente
        self.setMinimumHeight(100) 

    def update_dna(self, query_emb, ref_emb=None):
        """ Recebe as listas de números (embeddings) da IA e atualiza o visual """
        self.query_emb = np.array(query_emb) if query_emb else None
        self.ref_emb = np.array(ref_emb) if ref_emb else None
        self.update() # Força o PyQt a redesenhar o componente na tela

    def _get_color_heat(self, val):
        """ Converte um valor (0.0 a 1.0) em uma cor térmica (Azul -> Verde -> Vermelho) """
        val = max(0.0, min(1.0, val))
        r = int(min(255, max(0, 255 * (val * 2 - 1))))
        b = int(min(255, max(0, 255 * (2 - val * 2))))
        g = int(min(255, max(0, 255 * (1 - abs(val * 2 - 1)))))
        return QColor(r, g, b)

    def _get_color_diff(self, val):
        """ Converte a diferença em uma cor de alerta (Preto = Igual, Vermelho = Diferente) """
        val = max(0.0, min(1.0, val))
        return QColor(int(val * 255), 0, 0)

    def paintEvent(self, event):
        """ Motor de desenho gráfico de alta performance do PyQt """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()

        # Fundo escuro
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))

        if self.query_emb is None or len(self.query_emb) == 0:
            painter.setPen(QColor("#555555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Aguardando Matriz Semântica...")
            return

        # Prepara a matemática para normalizar os valores
        emb_q = self.query_emb
        emb_r = self.ref_emb if self.ref_emb is not None else np.zeros_like(emb_q)
        
        # Junta tudo para achar o maior e o menor valor e nivelar as cores
        all_vals = np.concatenate([emb_q, emb_r])
        v_min, v_max = np.min(all_vals), np.max(all_vals)
        if v_max == v_min: v_max = v_min + 1e-5 # Evita divisão por zero

        # Diferença absoluta entre a peça atual e o histórico
        diff = np.abs(emb_q - emb_r)
        d_max = np.max(diff) if np.max(diff) > 0 else 1.0

        num_features = len(emb_q)
        bar_w = w / num_features
        
        # Alturas das barras
        title_h = 12
        bar_h = (h - (title_h * 3) - 10) / 3

        font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font)

        # === DESENHA BARRA 1: PEÇA ATUAL ===
        y_offset = 5
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(5, int(y_offset + 10), "DNA: Peça Atual")
        y_offset += title_h
        
        # Pinta linha por linha do código de barras
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(emb_q):
            norm_val = (val - v_min) / (v_max - v_min)
            painter.setBrush(self._get_color_heat(norm_val))
            painter.drawRect(QRectF(i * bar_w, y_offset, bar_w + 1, bar_h))

        # === DESENHA BARRA 2: REFERÊNCIA DO BANCO ===
        y_offset += bar_h + 5
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(5, int(y_offset + 10), "DNA: Melhor Vizinho (Dataset)")
        y_offset += title_h
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(emb_r):
            norm_val = (val - v_min) / (v_max - v_min)
            painter.setBrush(self._get_color_heat(norm_val))
            painter.drawRect(QRectF(i * bar_w, y_offset, bar_w + 1, bar_h))

        # === DESENHA BARRA 3: DIVERGÊNCIA (O ALARME DE DEFEITO) ===
        y_offset += bar_h + 5
        painter.setPen(QColor("#ff5555"))
        painter.drawText(5, int(y_offset + 10), "Divergência (Foco da IA)")
        y_offset += title_h
        
        painter.setPen(Qt.PenStyle.NoPen)
        for i, val in enumerate(diff):
            norm_val = val / d_max # Destaca bem onde estão as diferenças
            painter.setBrush(self._get_color_diff(norm_val))
            painter.drawRect(QRectF(i * bar_w, y_offset, bar_w + 1, bar_h))

        painter.end()