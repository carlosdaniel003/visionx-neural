# src/ui/widgets/silk_debugger.py
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF

class SilkDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)

        self.is_active = False
        self.silk_error_pct = 0.0
        self.tolerance = 0.03
        self.is_defect = False
        self.reason = ""
        self.dx = 0.0
        self.dy = 0.0
        
        # Guardaremos as matrizes (imagens) geradas pelo OpenCV aqui
        self.mask_gab = None
        self.mask_test = None
        self.diff_mask = None

    def update_data(self, detail: dict):
        """ Recebe os detalhes da análise do SilkExpert """
        if not detail or "silk_error_pct" not in detail:
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.silk_error_pct = detail.get("silk_error_pct", 0.0)
        self.tolerance = detail.get("tolerance", 0.03) 
        self.is_defect = detail.get("is_defect", False)
        self.reason = detail.get("reason", "")
        self.dx = detail.get("dx", 0.0)
        self.dy = detail.get("dy", 0.0)
        
        # Puxa as imagens do Raio-X se elas existirem no dicionário
        self.mask_gab = detail.get("mask_gab", None)
        self.mask_test = detail.get("mask_test", None)
        self.diff_mask = detail.get("diff_mask", None)

        self.update()

    def _draw_np_mask(self, painter, arr: np.ndarray, rect: QRectF, title: str, colorize_red=False):
        """ Função auxiliar para desenhar matrizes do OpenCV no PyQt6 """
        # Desenha a moldura da "Telinha"
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.setBrush(QColor("#0a0a0a"))
        painter.drawRect(rect)

        # Desenha o Título da Telinha
        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, title)

        if arr is None or arr.size == 0:
            painter.setPen(QColor("#444444"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM DADOS")
            return

        h, w = arr.shape
        
        # Converte o P&B do OpenCV para uma QImage nativa
        if colorize_red:
            # Transforma os pixels brancos em Vermelho Sangue (Para a tela de Diferença)
            color_arr = np.zeros((h, w, 4), dtype=np.uint8)
            color_arr[arr > 0] = [255, 85, 85, 255] # RGBA
            qimg = QImage(color_arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        else:
            # Mantém em tons de cinza
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

        # Dimensiona a imagem para caber na moldura preservando a proporção
        scaled_img = qimg.scaled(int(rect.width() - 4), int(rect.height() - 4), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        # Centraliza a imagem dentro da moldura
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
        painter.drawText(5, 15, "Raio-X de Serigrafia (XOR Diff)")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor Silk Inativo")
            painter.end()
            return

        base_color = QColor("#ff5555") if self.is_defect else QColor("#55ff55")

        # =========================================================
        # 1. OS TRÊS MONITORES (Matrix View)
        # =========================================================
        # Calcula o espaço para as 3 telas caberem lado a lado
        padding = 10
        spacing = 5
        available_w = w - (padding * 2) - (spacing * 2)
        box_w = available_w / 3.0
        
        y_start = 35
        y_end = h - 45 # Deixa espaço para a barra e textos
        box_h = y_end - y_start

        rect_gab = QRectF(padding, y_start, box_w, box_h)
        rect_test = QRectF(padding + box_w + spacing, y_start, box_w, box_h)
        rect_diff = QRectF(padding + (box_w * 2) + (spacing * 2), y_start, box_w, box_h)

        # Desenha as matrizes!
        self._draw_np_mask(painter, self.mask_gab, rect_gab, "PADRÃO")
        self._draw_np_mask(painter, self.mask_test, rect_test, "CÂMERA")
        self._draw_np_mask(painter, self.diff_mask, rect_diff, "ANOMALIA", colorize_red=True)

        # =========================================================
        # 2. BARRA DE CONTRASTE (XOR GAUGE)
        # =========================================================
        bar_w = w * 0.8
        bar_h = 6
        bar_x = (w - bar_w) / 2.0
        bar_y = h - 30
        
        painter.setBrush(QColor("#333333"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))
        
        # A barra inteira agora representa 100% da área do chip (1.0)
        fill_max_pct = 1.0 
        fill_width = min(bar_w, (self.silk_error_pct / fill_max_pct) * bar_w)
        painter.setBrush(base_color)
        painter.drawRect(QRectF(bar_x, bar_y, fill_width, bar_h))
        
        # Posiciona a linha branca de tolerância baseada na nova escala de 100%
        tol_x = bar_x + (self.tolerance / fill_max_pct) * bar_w
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(tol_x, float(bar_y - 3)), QPointF(tol_x, float(bar_y + bar_h + 3)))

        # =========================================================
        # 3. HUD DE TELEMETRIA
        # =========================================================
        painter.setFont(QFont("Consolas", 7))
        
        painter.setPen(QColor("#888888"))
        painter.drawText(int(tol_x - 10), int(bar_y - 5), f"Tol:{self.tolerance:.1%}")
        
        painter.setPen(base_color)
        status_text = f"Erro Tinta: {self.silk_error_pct:.2%}" if not self.is_defect else "FALHA DE SERIGRAFIA"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.drawText(int(w - text_width - 5), h - 5, status_text)
        
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(5, h - 5, f"Mag-Lock: X:{self.dx:.1f} Y:{self.dy:.1f}")

        painter.end()