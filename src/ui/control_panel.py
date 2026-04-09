"""
Módulo do Painel de Controle e Console Duplo com Juiz Neural.
Exibe o Confidence Score (Nível de Certeza) das análises.
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from src.services.screen_monitor import ScreenMonitor
from src.services.dataset_manager import DatasetManager
from src.core.inspection import detect_anomalies
from src.core.neural_judge import NeuralJudge

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor = None
        self.current_sample = None
        self.current_ng = None
        self.neural_judge = NeuralJudge() 
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("VisionX Neural - Console Industrial IoT")
        self.resize(800, 550)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        main_layout = QVBoxLayout(self)
        
        title = QLabel("VisionX Neural: Monitoramento IoT")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        displays_layout = QHBoxLayout()
        
        sample_layout = QVBoxLayout()
        lbl_sample_title = QLabel("Padrão (Sample Image)")
        lbl_sample_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sample = QLabel("Aguardando Layout...")
        self.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sample.setStyleSheet("background-color: #000; border: 2px solid #2196F3; color: #fff;")
        self.lbl_sample.setMinimumSize(350, 300)
        sample_layout.addWidget(lbl_sample_title)
        sample_layout.addWidget(self.lbl_sample, stretch=1)
        
        ng_layout = QVBoxLayout()
        lbl_ng_title = QLabel("Anomalia (NG Image) + VisãoX")
        lbl_ng_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ng = QLabel("Aguardando Layout...")
        self.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ng.setStyleSheet("background-color: #000; border: 2px solid #F44336; color: #fff;")
        self.lbl_ng.setMinimumSize(350, 300)
        ng_layout.addWidget(lbl_ng_title)
        ng_layout.addWidget(self.lbl_ng, stretch=1)
        
        displays_layout.addLayout(sample_layout)
        displays_layout.addLayout(ng_layout)
        main_layout.addLayout(displays_layout, stretch=1)

        # Botão de Captura
        self.btn_start = QPushButton("Capturar IoT Agora")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self.start_monitoring)
        main_layout.addWidget(self.btn_start)

        # Botões de Curadoria (Active Learning)
        curation_layout = QHBoxLayout()
        
        self.btn_save_ok = QPushButton("Salvar como Falha Falsa (OK)")
        self.btn_save_ok.setMinimumHeight(45)
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ok.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_save_ok.clicked.connect(lambda: self.save_label("OK"))

        self.btn_save_ng = QPushButton("Confirmar Defeito Real (NG)")
        self.btn_save_ng.setMinimumHeight(45)
        self.btn_save_ng.setEnabled(False)
        self.btn_save_ng.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        self.btn_save_ng.clicked.connect(lambda: self.save_label("NG"))

        curation_layout.addWidget(self.btn_save_ok)
        curation_layout.addWidget(self.btn_save_ng)
        main_layout.addLayout(curation_layout)

    def start_monitoring(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Buscando na tela... (Minimizado)")
        self.lbl_sample.setText("Procurando barra Azul e Quadrado Verde...")
        self.lbl_ng.setText("Procurando barra Vermelha e Quadrado Verde...")
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.showMinimized()
        QTimer.singleShot(500, self._start_radar)

    def _start_radar(self):
        self.monitor = ScreenMonitor()
        self.monitor.layout_detected.connect(self.process_iot_images)
        self.monitor.start()

    def numpy_to_pixmap(self, img_bgr: np.ndarray) -> QPixmap:
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def process_iot_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray):
        if sample_crop.size == 0 or ng_crop.size == 0: return
        
        self.showNormal()
        self.activateWindow()
        
        self.current_sample = sample_crop
        self.current_ng = ng_crop
        
        px_sample = self.numpy_to_pixmap(sample_crop)
        self.lbl_sample.setPixmap(px_sample.scaled(self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        raw_anomalies = detect_anomalies(sample_crop, ng_crop)

        img_ng_drawn = ng_crop.copy()
        
        for (x, y, w, h) in raw_anomalies:
            suspect_gab = sample_crop[y:y+h, x:x+w]
            suspect_test = ng_crop[y:y+h, x:x+w]
            
            # Pede o veredito e a confiança ao Juiz Neural
            analysis = self.neural_judge.verify_anomaly(suspect_gab, suspect_test)
            is_real = analysis["is_defect"]
            score_txt = analysis["score_text"]
            
            if is_real:
                # Defeito Real: Quadrado Vermelho
                color = (0, 0, 255)
                label = f"DEFEITO: {score_txt}"
            else:
                # Falha Falsa (Matemática errou, IA corrigiu): Quadrado Laranja
                color = (0, 165, 255) 
                label = f"FALSO: {score_txt}"

            cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_ng_drawn, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        px_ng = self.numpy_to_pixmap(img_ng_drawn)
        self.lbl_ng.setPixmap(px_ng.scaled(self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Nova Captura IoT")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)

    def save_label(self, label: str):
        if self.current_ng is None: return
        h1, w1 = self.current_sample.shape[:2]
        h2, w2 = self.current_ng.shape[:2]
        
        h_min = min(h1, h2)
        s_resized = cv2.resize(self.current_sample, (int(w1 * h_min / h1), h_min))
        n_resized = cv2.resize(self.current_ng, (int(w2 * h_min / h2), h_min))
        
        pair_img = np.hstack((s_resized, n_resized))
        
        filepath = DatasetManager.save_sample(pair_img, label)
        if filepath:
            print(f"Dataset salvo como {label}: {filepath}")
            self.lbl_ng.setText(f"SALVO NO DATASET: {label}")
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)