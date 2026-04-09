"""
Módulo do Painel de Controle e Console Duplo com Juiz Neural v3.
Exibe painel de confiabilidade detalhado com métricas reais em padrão visual escuro (Dark Theme).
"""
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QMessageBox, QFrame, QGridLayout,
                             QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
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
        self.setWindowTitle("VisionX Neural - Console Industrial AOI")
        self.resize(900, 700)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #121212; color: #ffffff;")

        main_layout = QVBoxLayout(self)

        title = QLabel("VisionX Neural: Monitoramento AOI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 5px; color: #ffffff;")
        main_layout.addWidget(title)

        # === Displays de imagem ===
        displays_layout = QHBoxLayout()

        sample_layout = QVBoxLayout()
        lbl_sample_title = QLabel("Padrão (Sample Image)")
        lbl_sample_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sample_title.setStyleSheet("color: #aaaaaa;")
        self.lbl_sample = QLabel("Aguardando Layout...")
        self.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sample.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444444; color: #888888;")
        self.lbl_sample.setMinimumSize(350, 250)
        sample_layout.addWidget(lbl_sample_title)
        sample_layout.addWidget(self.lbl_sample, stretch=1)

        ng_layout = QVBoxLayout()
        lbl_ng_title = QLabel("Anomalia (NG Image) + VisãoX")
        lbl_ng_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ng_title.setStyleSheet("color: #aaaaaa;")
        self.lbl_ng = QLabel("Aguardando Layout...")
        self.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ng.setStyleSheet("background-color: #1a1a1a; border: 1px solid #444444; color: #888888;")
        self.lbl_ng.setMinimumSize(350, 250)
        ng_layout.addWidget(lbl_ng_title)
        ng_layout.addWidget(self.lbl_ng, stretch=1)

        displays_layout.addLayout(sample_layout)
        displays_layout.addLayout(ng_layout)
        main_layout.addLayout(displays_layout, stretch=1)

        # === Painel de Confiabilidade ===
        self.confidence_frame = QFrame()
        self.confidence_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        conf_layout = QVBoxLayout(self.confidence_frame)
        conf_layout.setSpacing(4)

        conf_title = QLabel("Painel de Confiabilidade - Análise do Juiz Neural")
        conf_title.setStyleSheet("color: #dddddd; font-weight: bold; font-size: 13px; border: none;")
        conf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(conf_title)

        # Veredito principal
        self.lbl_verdict = QLabel("Aguardando análise...")
        self.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_verdict.setStyleSheet("color: #888888; font-size: 16px; font-weight: bold; padding: 4px; border: none;")
        conf_layout.addWidget(self.lbl_verdict)

        # Grid de métricas detalhadas
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(3)

        self.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM (Similaridade)"),
            ("pct_changed", "Pixels Alterados"),
            ("edge_change", "Mudança de Bordas"),
            ("hist_corr", "Correlação Histograma"),
            ("local_score", "Score Local"),
            ("ctx_score", "Score Contexto"),
            ("db_score", "Score Dataset"),
            ("final_score", "Score Final"),
        ]

        for i, (key, label_text) in enumerate(metrics_def):
            row = i // 4
            col = (i % 4) * 2

            lbl_name = QLabel(label_text + ":")
            lbl_name.setStyleSheet("color: #8899aa; font-size: 11px; border: none; background: transparent;")
            lbl_value = QLabel("—")
            lbl_value.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold; border: none; background: transparent;")

            metrics_grid.addWidget(lbl_name, row, col)
            metrics_grid.addWidget(lbl_value, row, col + 1)
            self.metric_labels[key] = lbl_value

        conf_layout.addLayout(metrics_grid)

        # Linha de justificativa
        self.lbl_reason = QLabel("")
        self.lbl_reason.setStyleSheet("color: #aabbcc; font-size: 10px; padding-top: 4px; border: none;")
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(self.lbl_reason)

        # Informação do dataset
        self.lbl_db_info = QLabel("")
        self.lbl_db_info.setStyleSheet("color: #667788; font-size: 10px; border: none;")
        self.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(self.lbl_db_info)

        main_layout.addWidget(self.confidence_frame)

        # === Botões Customizados Padrão Escuro ===
        button_style = """
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #333333;
            }
        """

        # === Botão de Captura ===
        self.btn_start = QPushButton("Capturar AOI Agora")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.setStyleSheet(button_style)
        self.btn_start.clicked.connect(self.start_monitoring)
        main_layout.addWidget(self.btn_start)

        # === Botões de Curadoria ===
        curation_layout = QHBoxLayout()

        self.btn_save_ok = QPushButton("Salvar como Falha Falsa (OK)")
        self.btn_save_ok.setMinimumHeight(45)
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ok.setStyleSheet(button_style)
        self.btn_save_ok.clicked.connect(lambda: self.save_label("OK"))

        self.btn_save_ng = QPushButton("Confirmar Defeito Real (NG)")
        self.btn_save_ng.setMinimumHeight(45)
        self.btn_save_ng.setEnabled(False)
        self.btn_save_ng.setStyleSheet(button_style)
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
        self._reset_confidence_panel()
        self.showMinimized()
        QTimer.singleShot(500, self._start_radar)

    def _start_radar(self):
        self.monitor = ScreenMonitor()
        self.monitor.layout_detected.connect(self.process_aoi_images)
        self.monitor.start()

    def numpy_to_pixmap(self, img_bgr: np.ndarray) -> QPixmap:
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _reset_confidence_panel(self):
        self.lbl_verdict.setText("Aguardando análise...")
        self.lbl_verdict.setStyleSheet("color: #888888; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")
        for key, lbl in self.metric_labels.items():
            lbl.setText("—")
            lbl.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        self.lbl_reason.setText("")
        self.lbl_db_info.setText("")

    def _update_confidence_panel(self, analysis: dict):
        verdict = analysis.get("verdict", "?")
        conf_text = analysis.get("score_text", "?")
        is_defect = analysis.get("is_defect", False)

        if is_defect:
            self.lbl_verdict.setText(f"{verdict} - Confiança: {conf_text}")
            self.lbl_verdict.setStyleSheet("color: #ff5555; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")
        else:
            self.lbl_verdict.setText(f"{verdict} - Confiança: {conf_text}")
            self.lbl_verdict.setStyleSheet("color: #55ff55; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")

        detail = analysis.get("detail", {})

        def get_metric_color(val, invert=False):
            v = float(val) if val is not None else 0
            if invert:
                v_color = 1.0 - v
            else:
                v_color = v
            if v_color > 0.7: return "#ff5555"
            elif v_color > 0.4: return "#ffaa33"
            else: return "#55ff55"

        # SSIM
        ssim_val = detail.get("ssim", 0)
        self.metric_labels["ssim"].setText(f"{ssim_val:.3f}")
        self.metric_labels["ssim"].setStyleSheet(f"color: {get_metric_color(ssim_val, invert=True)}; font-size: 11px; font-weight: bold; border: none; background: transparent;")

        # Pixels alterados
        pct = detail.get("pct_changed", 0)
        self.metric_labels["pct_changed"].setText(f"{pct:.1%}")
        self.metric_labels["pct_changed"].setStyleSheet(f"color: {get_metric_color(pct / 0.15)}; font-size: 11px; font-weight: bold; border: none; background: transparent;")

        # Mudança de bordas
        edge = detail.get("edge_change", 0)
        self.metric_labels["edge_change"].setText(f"{edge:.1%}")
        self.metric_labels["edge_change"].setStyleSheet(f"color: {get_metric_color(edge / 0.08)}; font-size: 11px; font-weight: bold; border: none; background: transparent;")

        # Histograma
        hc = detail.get("hist_corr", 0)
        self.metric_labels["hist_corr"].setText(f"{hc:.3f}")
        self.metric_labels["hist_corr"].setStyleSheet(f"color: {get_metric_color(hc, invert=True)}; font-size: 11px; font-weight: bold; border: none; background: transparent;")

        # Scores compostos
        for key in ["local_score", "ctx_score", "db_score", "final_score"]:
            val = detail.get(key, 0)
            self.metric_labels[key].setText(f"{val:.2f}")
            color = get_metric_color(val / 0.6 if key != "final_score" else val / 0.45)
            font_size = "12px" if key == "final_score" else "11px"
            self.metric_labels[key].setStyleSheet(f"color: {color}; font-size: {font_size}; font-weight: bold; border: none; background: transparent;")

        # Justificativa
        reason = analysis.get("reason", "")
        if reason:
            self.lbl_reason.setText(f"Justificativa: {reason}")

        # Info do dataset
        db_mem = detail.get("db_has_memory", False)
        if db_mem:
            n = detail.get("db_neighbors", 0)
            vote = detail.get("db_vote", 0.5)
            sim = detail.get("db_best_sim", 0)
            self.lbl_db_info.setText(
                f"Base consultada: {n} vizinhos | Voto NG: {vote:.0%} | "
                f"Similaridade melhor match: {sim:.0%}"
            )
        else:
            self.lbl_db_info.setText("Sem dados no dataset. Salve amostras para melhorar a precisão!")

    def process_aoi_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray):
        if sample_crop.size == 0 or ng_crop.size == 0:
            return

        self.showNormal()
        self.activateWindow()

        self.current_sample = sample_crop
        self.current_ng = ng_crop

        px_sample = self.numpy_to_pixmap(sample_crop)
        self.lbl_sample.setPixmap(px_sample.scaled(
            self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        raw_anomalies = detect_anomalies(sample_crop, ng_crop)
        img_ng_drawn = ng_crop.copy()

        if not raw_anomalies:
            self._reset_confidence_panel()
            self.lbl_verdict.setText("Nenhuma anomalia detectada")
            self.lbl_verdict.setStyleSheet("color: #55ff55; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")
            self.lbl_reason.setText("A análise matemática não encontrou diferenças significativas entre as imagens.")
        else:
            biggest = max(raw_anomalies, key=lambda b: b[2] * b[3])
            last_analysis = None

            for (x, y, w, h) in raw_anomalies:
                suspect_gab = sample_crop[y:y+h, x:x+w]
                suspect_test = ng_crop[y:y+h, x:x+w]

                analysis = self.neural_judge.verify_anomaly(
                    crop_gab=suspect_gab,
                    crop_test=suspect_test,
                    full_gab=sample_crop,
                    full_test=ng_crop,
                    box_x=x, box_y=y, box_w=w, box_h=h
                )

                is_real = analysis["is_defect"]
                score_txt = analysis["score_text"]

                if is_real:
                    color = (0, 0, 255)
                    label_text = f"DEFEITO {score_txt}"
                else:
                    color = (0, 165, 255)
                    label_text = f"FALSO {score_txt}"

                cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color, 2)

                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img_ng_drawn, (x, y - th - 6), (x + tw + 4, y), color, -1)
                cv2.putText(img_ng_drawn, label_text, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                if (x, y, w, h) == biggest:
                    last_analysis = analysis

            if last_analysis:
                self._update_confidence_panel(last_analysis)

        px_ng = self.numpy_to_pixmap(img_ng_drawn)
        self.lbl_ng.setPixmap(px_ng.scaled(
            self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        self.btn_start.setEnabled(True)
        self.btn_start.setText("Nova Captura AOI")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)

    def save_label(self, label: str):
        if self.current_ng is None:
            return
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

            self.neural_judge.reload_memory()
            self.lbl_db_info.setText(
                f"Memória atualizada! "
                f"({len(self.neural_judge.memory.signatures_ok)} OK + "
                f"{len(self.neural_judge.memory.signatures_ng)} NG)"
            )