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
        
        # Matrizes brutas e imagens base do Backend
        self.heat_map_raw = None
        self.macro_edges = None
        self.crop_test = None
        self.full_test = None

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
        
        # Puxa os dados visuais
        self.heat_map_raw = detail.get("heat_map_raw", None)
        self.macro_edges = detail.get("macro_edges", None)
        self.crop_test = detail.get("crop_test", None)
        self.full_test = detail.get("full_test", None)

        self.update()

    def _draw_heatmap(self, painter, arr: np.ndarray, rect: QRectF, title: str):
        """ Renderiza a Lupa Micro (Mapa Termal Overlay) """
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(QColor("#0a0a0a"))
        painter.drawRect(rect)

        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, title)

        if arr is None or arr.size == 0 or self.crop_test is None:
            painter.setPen(QColor("#444444"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM DADOS (MICRO)")
            return

        # 1. Cria a máscara termal colorida
        heatmap_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        
        # 2. Pega a foto real da placa e iguala o tamanho ao mapa termal
        h, w = arr.shape
        bg_img = cv2.resize(self.crop_test, (w, h))
        
        # 3. A MÁGICA: Mistura a foto real (50%) com o Calor (70%)
        blended_bgr = cv2.addWeighted(bg_img, 0.5, heatmap_bgr, 0.7, 0)
        
        # Converte pro PyQt
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(blended_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        
        scaled_img = qimg.scaled(int(rect.width() - 4), int(rect.height() - 4), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        img_x = rect.x() + (rect.width() - scaled_img.width()) / 2
        img_y = rect.y() + (rect.height() - scaled_img.height()) / 2
        
        painter.drawImage(int(img_x), int(img_y), scaled_img)

    def _draw_xray_skeleton(self, painter, arr: np.ndarray, rect: QRectF, title: str):
        """ Renderiza o Scanner Macro (Esqueleto Verde Neon Overlay) """
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(QColor("#0a0a0a"))
        painter.drawRect(rect)

        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, title)

        if arr is None or arr.size == 0 or self.full_test is None:
            painter.setPen(QColor("#444444"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "NENHUM ELEFANTE (MACRO OK)")
            return

        h, w = arr.shape
        
        # 1. Escurece a imagem da placa em 60% para criar o clima de Raio-X
        dim_bg = cv2.addWeighted(self.full_test, 0.4, np.zeros_like(self.full_test), 0.6, 0)
        
        # 2. Pinta os "ossos" quebrados ou que sumiram de Verde Neon puro
        dim_bg[arr > 0] = [0, 255, 0] # BGR (Green)
        
        # Converte pro PyQt
        bg_rgb = cv2.cvtColor(dim_bg, cv2.COLOR_BGR2RGB)
        qimg = QImage(bg_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        
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
        painter.drawText(5, 15, "Laboratório Estrutural (SSIM Debugger)")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor SSIM Inativo (MoE)")
            painter.end()
            return

        # =========================================================
        # 1. LAYOUT DOS DOIS MONITORES (Biocular)
        # =========================================================
        padding = 10
        spacing = 10
        available_w = w - (padding * 2) - spacing
        box_w = available_w / 2.0
        
        y_start = 35
        y_end = h - 35 
        box_h = y_end - y_start

        rect_micro = QRectF(padding, y_start, box_w, box_h)
        rect_macro = QRectF(padding + box_w + spacing, y_start, box_w, box_h)

        # RENDERIZADORES COM OVERLAY 
        self._draw_heatmap(painter, self.heat_map_raw, rect_micro, "1. LUPA MICRO (Calor + Foto Real)")
        self._draw_xray_skeleton(painter, self.macro_edges, rect_macro, "2. SCANNER MACRO (Raio-X de Canny)")

        # =========================================================
        # 2. HUD DE TELEMETRIA
        # =========================================================
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        
        is_critical = self.local_score > 0.60
        status_color = QColor("#ff5555") if is_critical else QColor("#55ff55")
        
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(padding, h - 10, f"Contexto: {self.ctx_reason}")
        
        status_text = f"Ameaça SSIM: {self.local_score:.0%}"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.setPen(status_color)
        painter.drawText(int(w - padding - text_width), h - 10, status_text)

        painter.end()