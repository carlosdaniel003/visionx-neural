"""
Módulo do Painel de Controle inicial e Console de Dataset.
Permite iniciar o monitoramento e rotular imagens ao vivo (Active Learning).
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QHBoxLayout, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from src.ui.calibration_window import CalibrationWindow
from src.ui.hud_window import HUDWindow
from src.services.screen_monitor import ScreenMonitor
from src.services.dataset_manager import DatasetManager

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.hud = None
        self.monitor = None
        self.calib_window = None
        self.current_crop = None # Armazena a matriz Numpy da imagem atual
        
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("VisionX Neural - Console Central")
        self.resize(350, 400)
        # Mantém este painel sempre no topo para facilitar o clique
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        self.layout = QVBoxLayout(self)
        
        # QStackedWidget permite alternar entre telas sem fechar a janela
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        # --- Página 0: Menu Principal ---
        self.page_menu = QWidget()
        menu_layout = QVBoxLayout(self.page_menu)
        
        title = QLabel("VisionX Neural Core")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        
        btn_calib = QPushButton("⚙️ Calibrar Alvo (ROI)")
        btn_calib.setMinimumHeight(40)
        btn_calib.clicked.connect(self.open_calibration)
        
        btn_start = QPushButton("🚀 Iniciar Modo Dataset")
        btn_start.setMinimumHeight(40)
        btn_start.clicked.connect(self.start_monitoring)
        
        menu_layout.addWidget(title)
        menu_layout.addWidget(btn_calib)
        menu_layout.addWidget(btn_start)
        menu_layout.addStretch()
        self.stack.addWidget(self.page_menu)

        # --- Página 1: Console de Dataset (Active Learning) ---
        self.page_dataset = QWidget()
        ds_layout = QVBoxLayout(self.page_dataset)
        
        ds_title = QLabel("Curadoria de Dataset (Ao Vivo)")
        ds_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ds_title.setStyleSheet("font-weight: bold;")
        
        # Onde a imagem ao vivo vai aparecer
        self.lbl_preview = QLabel("Aguardando detecção do alvo...")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #000; color: #0f0; border: 2px solid #555;")
        self.lbl_preview.setMinimumSize(300, 200)
        
        # Botões de Classificação
        btn_box = QHBoxLayout()
        self.btn_ok = QPushButton("✅ SALVAR OK")
        self.btn_ok.setMinimumHeight(50)
        self.btn_ok.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_ok.clicked.connect(lambda: self.save_label("OK"))
        
        self.btn_ng = QPushButton("❌ SALVAR DEFEITO")
        self.btn_ng.setMinimumHeight(50)
        self.btn_ng.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.btn_ng.clicked.connect(lambda: self.save_label("NG"))
        
        btn_box.addWidget(self.btn_ok)
        btn_box.addWidget(self.btn_ng)
        
        ds_layout.addWidget(ds_title)
        ds_layout.addWidget(self.lbl_preview, stretch=1)
        ds_layout.addLayout(btn_box)
        self.stack.addWidget(self.page_dataset)

    def open_calibration(self):
        self.calib_window = CalibrationWindow()
        self.calib_window.show()

    def start_monitoring(self):
        # Alterna para a interface do Dataset
        self.stack.setCurrentIndex(1)
        
        # Inicia a HUD transparente
        self.hud = HUDWindow()
        self.hud.show()
        
        # Inicia o motor
        self.monitor = ScreenMonitor()
        self.monitor.pattern_found.connect(self.hud.update_target)
        self.monitor.pattern_lost.connect(self.hud.clear_target)
        self.monitor.log_updated.connect(self.hud.update_log)
        
        # NOVO: Conecta a câmera ao vivo ao painel
        self.monitor.crop_updated.connect(self.update_live_preview)
        
        self.monitor.start()

    def update_live_preview(self, crop: np.ndarray):
        """Recebe a imagem do radar em tempo real e atualiza a telinha."""
        self.current_crop = crop
        
        # Converte a matriz Numpy (BGR) para uma Imagem PyQt (RGB)
        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Mostra na tela mantendo a proporção
        self.lbl_preview.setPixmap(pixmap.scaled(self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def update_live_preview(self, crop: np.ndarray):
        """Recebe a imagem do radar em tempo real e atualiza a telinha."""
        
        # --- TRAVA DE SEGURANÇA: Se a imagem vier vazia, apenas ignora ---
        if crop is None or crop.size == 0:
            return
            
        self.current_crop = crop
        
        # Converte a matriz Numpy (BGR) para uma Imagem PyQt (RGB)
        rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Mostra na tela mantendo a proporção
        self.lbl_preview.setPixmap(pixmap.scaled(self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        """Chama o DatasetManager para salvar a imagem atual."""
        if self.current_crop is None:
            QMessageBox.warning(self, "Aviso", "Não há nenhuma imagem sendo detectada agora para salvar.")
            return
            
        filepath = DatasetManager.save_sample(self.current_crop, label)
        if filepath:
            print(f"Salvo [{label}]: {filepath}")
            # Um pequeno feedback visual de que salvou
            self.lbl_preview.setText(f"SALVO COMO {label}!")
            self.lbl_preview.setPixmap(QPixmap()) # Limpa a tela por um milissegundo para piscar