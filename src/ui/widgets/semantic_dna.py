# src/ui/widgets/semantic_dna.py
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF

class SemanticDNAWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150) 
        self.setMinimumWidth(150)
        
        self.is_active = False
        self.sem_loss = 0.0
        self.sem_img_gab = None
        self.sem_img_test = None

    def update_dna(self, query_emb, ref_emb=None):
        """ Método legado mantido por compatibilidade """
        pass 

    def update_data(self, detail: dict):
        """ Recebe o dict completo do Orquestrador """
        if not detail or "semantic_loss" not in detail:
            self.is_active = False
            self.update()
            return
            
        self.is_active = True
        self.sem_loss = detail.get("semantic_loss", 0.0)
        self.sem_img_gab = detail.get("sem_img_gab", None)
        self.sem_img_test = detail.get("sem_img_test", None)
        self.update()

    def _draw_image_box(self, painter, img_bgr: np.ndarray, rect: QRectF, title: str):
        """ Função genérica para desenhar uma matriz colorida na tela do PyQt6 """
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(QColor("#0a0a0a"))
        painter.drawRect(rect)

        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, title)

        if img_bgr is None or img_bgr.size == 0:
            painter.setPen(QColor("#444444"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM FOTO")
            return

        h, w = img_bgr.shape[:2]
        
        if not img_bgr.flags['C_CONTIGUOUS']:
            img_bgr = np.ascontiguousarray(img_bgr)
            
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        
        scaled_img = qimg.scaled(int(rect.width() - 4), int(rect.height() - 4), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_x = rect.x() + (rect.width() - scaled_img.width()) / 2
        img_y = rect.y() + (rect.height() - scaled_img.height()) / 2
        
        painter.drawImage(int(img_x), int(img_y), scaled_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))

        painter.setPen(QColor("#dddddd"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(5, 15, "Scanner de Estrutura (Semantic ORB)")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor Semântico Inativo (MoE)")
            painter.end()
            return

        # =========================================================
        # LAYOUT DOS DOIS MONITORES (Visão Biocular ORB)
        # =========================================================
        padding = 10
        spacing = 10
        available_w = w - (padding * 2) - spacing
        box_w = available_w / 2.0
        
        y_start = 35
        y_end = h - 35 
        box_h = y_end - y_start

        rect_gab = QRectF(padding, y_start, box_w, box_h)
        rect_test = QRectF(padding + box_w + spacing, y_start, box_w, box_h)

        # As telas mostrando os pontinhos estruturais
        self._draw_image_box(painter, self.sem_img_gab, rect_gab, "PADRÃO (Quinas + Falhas Vermelhas)")
        self._draw_image_box(painter, self.sem_img_test, rect_test, "CÂMERA (Quinas Encontradas)")

        # =========================================================
        # HUD DE TELEMETRIA
        # =========================================================
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        
        is_critical = self.sem_loss > 0.40
        status_color = QColor("#ff5555") if is_critical else QColor("#55ff55")
        
        status_text = f"Dano Estrutural: {self.sem_loss:.0%}"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.setPen(status_color)
        painter.drawText(int(w - padding - text_width), h - 10, status_text)

        painter.end()