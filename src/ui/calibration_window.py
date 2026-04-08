"""
Módulo para calibração do alvo (Region of Interest - ROI).
Interface profissional com Lupa de Zoom, ajuste fino de bordas, Auto-Fit e visualização do alvo atual.
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QMessageBox, QHBoxLayout)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
from src.config.settings import settings

class ROILabel(QLabel):
    """Componente avançado com Lupa, redimensionamento por bordas e arraste."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMouseTracking(True) 
        self.image_pixmap = None

        self.mode = "IDLE" 
        self.current_rect = QRect()
        self.rect_before_drag = QRect()
        self.start_pos = QPoint()
        self.mouse_pos = None
        self.active_handle = None
        self.handle_size = 12 

    def set_image(self, pixmap: QPixmap):
        self.image_pixmap = pixmap
        self.setPixmap(pixmap)
        self.current_rect = QRect()
        self.mode = "IDLE"

    def get_handle(self, pos: QPoint):
        if self.current_rect.isNull(): return None
        r = self.current_rect
        hs = self.handle_size

        if QRect(r.topLeft() - QPoint(hs//2, hs//2), QSize(hs, hs)).contains(pos): return 'tl'
        if QRect(r.topRight() - QPoint(hs//2, hs//2), QSize(hs, hs)).contains(pos): return 'tr'
        if QRect(r.bottomLeft() - QPoint(hs//2, hs//2), QSize(hs, hs)).contains(pos): return 'bl'
        if QRect(r.bottomRight() - QPoint(hs//2, hs//2), QSize(hs, hs)).contains(pos): return 'br'

        if QRect(r.left() - hs//2, r.top(), hs, r.height()).contains(pos): return 'l'
        if QRect(r.right() - hs//2, r.top(), hs, r.height()).contains(pos): return 'r'
        if QRect(r.left(), r.top() - hs//2, r.width(), hs).contains(pos): return 't'
        if QRect(r.left(), r.bottom() - hs//2, r.width(), hs).contains(pos): return 'b'

        if r.contains(pos): return 'center'
        return None

    def update_cursor(self, pos: QPoint):
        handle = self.get_handle(pos)
        if handle in ['tl', 'br']: self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in ['tr', 'bl']: self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in ['l', 'r']: self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle in ['t', 'b']: self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle == 'center': self.setCursor(Qt.CursorShape.SizeAllCursor)
        else: self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.image_pixmap:
            return

        self.start_pos = event.pos()
        self.rect_before_drag = QRect(self.current_rect)
        self.active_handle = self.get_handle(self.start_pos)

        if self.active_handle == 'center':
            self.mode = "DRAGGING"
        elif self.active_handle is not None:
            self.mode = "RESIZING"
        else:
            self.mode = "CREATING"
            self.current_rect = QRect(self.start_pos, self.start_pos)

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()

        if self.mode == "IDLE" and self.image_pixmap:
            self.update_cursor(self.mouse_pos)

        elif self.mode == "CREATING":
            self.current_rect = QRect(self.start_pos, self.mouse_pos).normalized()

        elif self.mode == "DRAGGING":
            dp = self.mouse_pos - self.start_pos
            self.current_rect = self.rect_before_drag.translated(dp)

        elif self.mode == "RESIZING":
            dp = self.mouse_pos - self.start_pos
            r = QRect(self.rect_before_drag)

            if 'l' in self.active_handle: r.setLeft(r.left() + dp.x())
            if 'r' in self.active_handle: r.setRight(r.right() + dp.x())
            if 't' in self.active_handle: r.setTop(r.top() + dp.y())
            if 'b' in self.active_handle: r.setBottom(r.bottom() + dp.y())

            self.current_rect = r.normalized()

        self.update() 

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mode = "IDLE"
            self.update()

    def draw_loupe(self, painter: QPainter):
        if not self.mouse_pos or not self.image_pixmap: return

        cap_size = 40       
        zoom_factor = 4     
        loupe_size = cap_size * zoom_factor

        target_pixmap = QPixmap(cap_size, cap_size)
        target_pixmap.fill(Qt.GlobalColor.black)

        src_x = self.mouse_pos.x() - cap_size // 2
        src_y = self.mouse_pos.y() - cap_size // 2

        temp_painter = QPainter(target_pixmap)
        temp_painter.drawPixmap(0, 0, self.image_pixmap, src_x, src_y, cap_size, cap_size)
        temp_painter.end()

        scaled_crop = target_pixmap.scaled(loupe_size, loupe_size, 
                                           Qt.AspectRatioMode.KeepAspectRatio, 
                                           Qt.TransformationMode.FastTransformation)

        draw_x = self.mouse_pos.x() + 20
        draw_y = self.mouse_pos.y() + 20

        if draw_x + loupe_size > self.width(): draw_x = self.mouse_pos.x() - loupe_size - 20
        if draw_y + loupe_size > self.height(): draw_y = self.mouse_pos.y() - loupe_size - 20

        painter.fillRect(draw_x, draw_y, loupe_size, loupe_size, QColor(0, 0, 0))
        painter.drawPixmap(draw_x, draw_y, scaled_crop)

        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        center_x = draw_x + loupe_size // 2
        center_y = draw_y + loupe_size // 2
        painter.drawLine(center_x - 15, center_y, center_x + 15, center_y)
        painter.drawLine(center_x, center_y - 15, center_x, center_y + 15)

        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(draw_x, draw_y, loupe_size, loupe_size)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.image_pixmap: return
        
        painter = QPainter(self)
        
        if not self.current_rect.isNull():
            painter.fillRect(0, 0, self.width(), self.current_rect.top(), QColor(0, 0, 0, 100))
            painter.fillRect(0, self.current_rect.bottom() + 1, self.width(), self.height() - self.current_rect.bottom(), QColor(0, 0, 0, 100))
            painter.fillRect(0, self.current_rect.top(), self.current_rect.left(), self.current_rect.height(), QColor(0, 0, 0, 100))
            painter.fillRect(self.current_rect.right() + 1, self.current_rect.top(), self.width() - self.current_rect.right(), self.current_rect.height(), QColor(0, 0, 0, 100))

            pen = QPen(QColor(0, 255, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)

        self.draw_loupe(painter)


class CalibrationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.cv_image = None
        self.scale_factor = 1.0  
        self._setup_ui()
        self.load_current_template() # Carrega o alvo atual ao abrir

    def _setup_ui(self):
        self.setWindowTitle("VisionX Neural - Calibrar Zona de Interesse Avançado")
        self.resize(1200, 750) # Janela um pouco mais larga para o painel lateral
        
        main_layout = QHBoxLayout(self) # O layout principal agora é Horizontal (Lado a Lado)
        
        # ==========================================
        # LADO ESQUERDO: CONTROLES E DESENHO
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("1. Carregar Imagem Base")
        self.btn_load.setMinimumHeight(40)
        self.btn_save = QPushButton("2. Salvar Zona de Interesse")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setEnabled(False)
        
        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_roi)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_save)
        
        self.image_label = ROILabel()
        
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.image_label, stretch=1)
        
        # ==========================================
        # LADO DIREITO: PAINEL DE INFORMAÇÕES
        # ==========================================
        right_widget = QWidget()
        right_widget.setFixedWidth(300) # Fixa a largura do painel lateral
        right_layout = QVBoxLayout(right_widget)
        
        lbl_title = QLabel("🎯 Padrão Treinado Atual")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        
        self.lbl_current_template = QLabel("Nenhum padrão\nencontrado no disco.")
        self.lbl_current_template.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_current_template.setStyleSheet("background-color: #1e1e1e; color: #888; border: 2px dashed #555;")
        self.lbl_current_template.setMinimumSize(280, 280)
        
        right_layout.addWidget(lbl_title)
        right_layout.addWidget(self.lbl_current_template)
        right_layout.addStretch() # Empurra tudo para cima
        
        # Adiciona os dois lados na janela principal
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(right_widget)

    def load_current_template(self):
        """Busca o template atual no disco e exibe no painel direito."""
        path = str(settings.TEMPLATE_IMAGE_PATH)
        try:
            file_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Redimensiona para caber bonito no quadrado lateral
                scaled_pixmap = pixmap.scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.lbl_current_template.setPixmap(scaled_pixmap)
                self.lbl_current_template.setStyleSheet("background-color: #000; border: 2px solid #0ff;")
        except Exception:
            pass # Mantém o texto "Nenhum padrão encontrado"

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Imagem", "", "Imagens (*.png *.jpg *.jpeg)")
        if not file_name: return

        try:
            file_bytes = np.fromfile(file_name, dtype=np.uint8)
            self.cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            self.cv_image = None
            
        if self.cv_image is None:
            QMessageBox.critical(self, "Erro", f"Não foi possível carregar:\n{file_name}")
            return
        
        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        max_w = self.image_label.width()
        max_h = self.image_label.height()

        original_w, original_h = pixmap.width(), pixmap.height()
        scale_w = max_w / original_w if original_w > 0 else 1
        scale_h = max_h / original_h if original_h > 0 else 1
        
        self.scale_factor = min(scale_w, scale_h)
        
        if self.scale_factor < 1.0:
            new_w = int(original_w * self.scale_factor)
            new_h = int(original_h * self.scale_factor)
            display_pixmap = pixmap.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            self.scale_factor = 1.0
            display_pixmap = pixmap

        self.image_label.set_image(display_pixmap)
        self.btn_save.setEnabled(True)

    def save_roi(self):
        rect = self.image_label.current_rect
        if rect.isNull() or rect.width() < 5 or rect.height() < 5:
            QMessageBox.warning(self, "Aviso", "Desenhe ou ajuste uma zona válida primeiro.")
            return

        ui_x, ui_y, ui_w, ui_h = rect.x(), rect.y(), rect.width(), rect.height()
        
        x = int(ui_x / self.scale_factor)
        y = int(ui_y / self.scale_factor)
        w = int(ui_w / self.scale_factor)
        h = int(ui_h / self.scale_factor)
        
        roi_cropped = self.cv_image[y:y+h, x:x+w]
        
        save_path = str(settings.TEMPLATE_IMAGE_PATH)
        cv2.imwrite(save_path, roi_cropped)
        
        # Atualiza a janelinha na hora para o usuário ver o que salvou!
        self.load_current_template()
        
        QMessageBox.information(self, "Sucesso", f"Template de {w}x{h}px salvo com sucesso!")
        self.close()