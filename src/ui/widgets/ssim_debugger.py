# src/ui/widgets/ssim_debugger.py
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF

class SSIMDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)

        self.is_active = False
        self.local_score = 0.0
        self.ctx_score = 0.0
        self.ctx_reason = ""
        
        # Matrizes brutas e imagens base do Backend (Epicentros Focados)
        self.heat_map_raw = None
        self.crop_gab = None
        self.crop_test = None

    def update_data(self, detail: dict):
        """ Recebe os detalhes da análise do SSIMExpert """
        if not detail or "heat_map_raw" not in detail:
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.local_score = detail.get("local_score", 0.0)
        self.ctx_score = detail.get("ctx_score", 0.0)
        self.ctx_reason = detail.get("ctx_reason", "")
        
        # Puxa os recortes de epicentro e a máscara da IA
        self.heat_map_raw = detail.get("heat_map_raw", None)
        self.crop_gab = detail.get("crop_gab", None)
        self.crop_test = detail.get("crop_test", None)

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
        
        # Conversão C-Contiguous obrigatória para não corromper a memória do Qt
        if not img_bgr.flags['C_CONTIGUOUS']:
            img_bgr = np.ascontiguousarray(img_bgr)
            
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        
        scaled_img = qimg.scaled(int(rect.width() - 4), int(rect.height() - 4), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_x = rect.x() + (rect.width() - scaled_img.width()) / 2
        img_y = rect.y() + (rect.height() - scaled_img.height()) / 2
        
        painter.drawImage(int(img_x), int(img_y), scaled_img)

    def _draw_overlay_heatmap(self, painter, heat_arr: np.ndarray, bg_img: np.ndarray, rect: QRectF, title: str):
        """ Renderiza o Veredito da IA (Foto Anomalia Misturada com o Mapa de Calor do SSIM) """
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(QColor("#0a0a0a"))
        painter.drawRect(rect)

        painter.setPen(QColor("#ff5555")) # Título chamativo para a IA
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, title)

        if heat_arr is None or heat_arr.size == 0 or bg_img is None:
            painter.setPen(QColor("#444444"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "FALHA NA FUSÃO")
            return

        h, w = heat_arr.shape
        
        # 1. Pinta o P&B do OpenCV com o Mapa de Calor (Azul=Normal, Vermelho=Defeito)
        heatmap_bgr = cv2.applyColorMap(heat_arr, cv2.COLORMAP_JET)
        
        # 2. Iguala a foto real pro tamanho do mapa pra não dar crash
        bg_resized = cv2.resize(bg_img, (w, h))
        
        # 3. Mistura: 50% da foto de Teste com 70% das manchas térmicas
        blended_bgr = cv2.addWeighted(bg_resized, 0.5, heatmap_bgr, 0.7, 0)
        
        if not blended_bgr.flags['C_CONTIGUOUS']:
            blended_bgr = np.ascontiguousarray(blended_bgr)
            
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(blended_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        
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
        painter.drawText(5, 15, "Laboratório de Textura (SSIM Debugger)")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor SSIM Inativo")
            painter.end()
            return

        # =========================================================
        # 1. LAYOUT DE TRÊS MONITORES (Visão Tripla de Epicentro)
        # =========================================================
        padding = 10
        spacing = 5
        available_w = w - (padding * 2) - (spacing * 2)
        box_w = available_w / 3.0
        
        y_start = 35
        y_end = h - 35 
        box_h = y_end - y_start

        rect_gab = QRectF(padding, y_start, box_w, box_h)
        rect_test = QRectF(padding + box_w + spacing, y_start, box_w, box_h)
        rect_diff = QRectF(padding + (box_w * 2) + (spacing * 2), y_start, box_w, box_h)

        # As três telinhas focadas no epicentro: Gabarito, Anomalia e o Veredito da IA.
        self._draw_image_box(painter, self.crop_gab, rect_gab, "1. PADRÃO (Epicentro)")
        self._draw_image_box(painter, self.crop_test, rect_test, "2. ANOMALIA (Epicentro)")
        self._draw_overlay_heatmap(painter, self.heat_map_raw, self.crop_test, rect_diff, "3. VEREDITO IA (Calor)")

        # =========================================================
        # 2. HUD DE TELEMETRIA INFERIOR
        # =========================================================
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        
        is_critical = self.local_score > 0.45 # O corte do Orquestrador
        status_color = QColor("#ff5555") if is_critical else QColor("#55ff55")
        
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(padding, h - 10, f"Contexto IA: {self.ctx_reason}")
        
        status_text = f"Dano Físico: {self.local_score:.0%}"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.setPen(status_color)
        painter.drawText(int(w - padding - text_width), h - 10, status_text)

        painter.end()