# src\ui\control_panel.py
"""
Módulo do Painel de Controle e Console Duplo com Juiz Neural v3.
Tela inteira sem cobrir a barra de tarefas (Maximized).
3ª imagem: referência mais parecida encontrada no banco de dados.
Exibe informações de Board, Parts e Value extraídas da AOI em Dark Theme.
Salva metadados JSON ao curar amostras.
Inclui Ticker Time para medir latência da operação.
Exibe % de Defeito Real e % de Falha Falsa simultaneamente.
"""
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QHBoxLayout, QMessageBox, QFrame, QGridLayout,
                             QApplication)
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
        self.current_aoi_info = {}
        self.current_analysis = None
        self.capture_start_time = 0.0
        self.neural_judge = NeuralJudge()
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("VisionX Neural - Console Industrial AOI")
        self.setStyleSheet("background-color: #121212; color: #ffffff;")

        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        self.setGeometry(available)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 6, 10, 6)
        main_layout.setSpacing(4)

        # ============================================================
        # === CABEÇALHO (Título + Ticker Time)
        # ============================================================
        top_layout = QHBoxLayout()
        
        title = QLabel("VisionX Neural: Monitoramento AOI")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; margin-bottom: 2px; color: #ffffff;")
        
        self.lbl_timer = QLabel("Latencia: 0.00s")
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_timer.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #aaaaaa;")

        # Adiciona um espaço flexível antes do título para mantê-lo centralizado,
        # e o timer fica ancorado à direita.
        spacer = QLabel("")
        spacer.setMinimumWidth(120) 
        top_layout.addWidget(spacer)
        top_layout.addWidget(title, stretch=1)
        top_layout.addWidget(self.lbl_timer)
        
        main_layout.addLayout(top_layout)

        # ============================================================
        # === BARRA DE INFORMAÇÕES DA AOI (Board / Parts / Value)
        # ============================================================
        self.aoi_info_frame = QFrame()
        self.aoi_info_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        aoi_info_layout = QHBoxLayout(self.aoi_info_frame)
        aoi_info_layout.setContentsMargins(12, 4, 12, 4)
        aoi_info_layout.setSpacing(20)

        info_label_style = "color: #888888; font-size: 11px; border: none;"
        info_value_style = ("color: #ffffff; font-size: 12px; font-weight: bold; border: none;")

        lbl_board_name = QLabel("Board:")
        lbl_board_name.setStyleSheet(info_label_style)
        self.lbl_board_value = QLabel("-")
        self.lbl_board_value.setStyleSheet(info_value_style)

        lbl_parts_name = QLabel("Parts:")
        lbl_parts_name.setStyleSheet(info_label_style)
        self.lbl_parts_value = QLabel("-")
        self.lbl_parts_value.setStyleSheet(info_value_style)

        lbl_value_name = QLabel("Value:")
        lbl_value_name.setStyleSheet(info_label_style)
        self.lbl_value_value = QLabel("-")
        self.lbl_value_value.setStyleSheet(info_value_style)

        aoi_info_layout.addWidget(lbl_board_name)
        aoi_info_layout.addWidget(self.lbl_board_value)
        aoi_info_layout.addStretch()
        aoi_info_layout.addWidget(lbl_parts_name)
        aoi_info_layout.addWidget(self.lbl_parts_value)
        aoi_info_layout.addStretch()
        aoi_info_layout.addWidget(lbl_value_name)
        aoi_info_layout.addWidget(self.lbl_value_value)

        main_layout.addWidget(self.aoi_info_frame)

        # ============================================================
        # === 3 DISPLAYS DE IMAGEM
        # ============================================================
        displays_layout = QHBoxLayout()
        displays_layout.setSpacing(8)

        sample_layout = QVBoxLayout()
        lbl_sample_title = QLabel("Padrao (Sample)")
        lbl_sample_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sample_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_sample = QLabel("Aguardando Layout...")
        self.lbl_sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sample.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        self.lbl_sample.setMinimumSize(200, 180)
        sample_layout.addWidget(lbl_sample_title)
        sample_layout.addWidget(self.lbl_sample, stretch=1)

        ng_layout = QVBoxLayout()
        lbl_ng_title = QLabel("Analise VisaoX (NG)")
        lbl_ng_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ng_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_ng = QLabel("Aguardando Layout...")
        self.lbl_ng.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ng.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        self.lbl_ng.setMinimumSize(200, 180)
        ng_layout.addWidget(lbl_ng_title)
        ng_layout.addWidget(self.lbl_ng, stretch=1)

        ref_layout = QVBoxLayout()
        self.lbl_ref_title = QLabel("Referencia (Dataset)")
        self.lbl_ref_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_ref = QLabel("Sem dados no banco")
        self.lbl_ref.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        self.lbl_ref.setMinimumSize(200, 180)
        self.lbl_ref_info = QLabel("")
        self.lbl_ref_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ref_info.setStyleSheet("color: #888888; font-size: 10px;")
        self.lbl_ref_info.setWordWrap(True)
        ref_layout.addWidget(self.lbl_ref_title)
        ref_layout.addWidget(self.lbl_ref, stretch=1)
        ref_layout.addWidget(self.lbl_ref_info)

        displays_layout.addLayout(sample_layout, stretch=1)
        displays_layout.addLayout(ng_layout, stretch=1)
        displays_layout.addLayout(ref_layout, stretch=1)
        main_layout.addLayout(displays_layout, stretch=1)

        # ============================================================
        # === PAINEL DE CONFIABILIDADE
        # ============================================================
        self.confidence_frame = QFrame()
        self.confidence_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        conf_layout = QVBoxLayout(self.confidence_frame)
        conf_layout.setSpacing(3)

        conf_title = QLabel("Painel de Confiabilidade - Analise do Juiz Neural")
        conf_title.setStyleSheet(
            "color: #dddddd; font-weight: bold; font-size: 13px; border: none;")
        conf_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(conf_title)

        self.lbl_verdict = QLabel("Aguardando analise...")
        self.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_verdict.setStyleSheet(
            "color: #888888; font-size: 16px; font-weight: bold; padding: 4px; border: none;")
        conf_layout.addWidget(self.lbl_verdict)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(3)

        self.metric_labels = {}
        metrics_def = [
            ("ssim", "SSIM (Similaridade)"),
            ("pct_changed", "Pixels Alterados"),
            ("edge_change", "Mudanca de Bordas"),
            ("hist_corr", "Correlacao Histograma"),
            ("local_score", "Score Local"),
            ("ctx_score", "Score Contexto"),
            ("db_score", "Score Dataset"),
            ("final_score", "Score Final"),
        ]

        for i, (key, label_text) in enumerate(metrics_def):
            row = i // 4
            col = (i % 4) * 2
            lbl_name = QLabel(label_text + ":")
            lbl_name.setStyleSheet(
                "color: #888888; font-size: 11px; border: none; background: transparent;")
            lbl_value = QLabel("-")
            lbl_value.setStyleSheet(
                "color: #ffffff; font-size: 11px; font-weight: bold; "
                "border: none; background: transparent;")
            metrics_grid.addWidget(lbl_name, row, col)
            metrics_grid.addWidget(lbl_value, row, col + 1)
            self.metric_labels[key] = lbl_value

        conf_layout.addLayout(metrics_grid)

        self.lbl_reason = QLabel("")
        self.lbl_reason.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; padding-top: 2px; border: none;")
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(self.lbl_reason)

        self.lbl_db_info = QLabel("")
        self.lbl_db_info.setStyleSheet(
            "color: #777777; font-size: 10px; border: none;")
        self.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conf_layout.addWidget(self.lbl_db_info)

        main_layout.addWidget(self.confidence_frame)

        # ============================================================
        # === BOTÕES
        # ============================================================
        button_style = """
            QPushButton {
                background-color: #2d2d2d; color: #ffffff; font-weight: bold;
                border: 1px solid #444444; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3d3d3d; }
            QPushButton:disabled {
                background-color: #1a1a1a; color: #555555;
                border: 1px solid #333333;
            }
        """

        self.btn_start = QPushButton("Capturar AOI Agora")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(button_style)
        self.btn_start.clicked.connect(self.start_monitoring)
        main_layout.addWidget(self.btn_start)

        curation_layout = QHBoxLayout()

        self.btn_save_ok = QPushButton("Salvar como Falha Falsa (OK)")
        self.btn_save_ok.setMinimumHeight(40)
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ok.setStyleSheet(button_style)
        self.btn_save_ok.clicked.connect(lambda: self.save_label("OK"))

        self.btn_save_ng = QPushButton("Confirmar Defeito Real (NG)")
        self.btn_save_ng.setMinimumHeight(40)
        self.btn_save_ng.setEnabled(False)
        self.btn_save_ng.setStyleSheet(button_style)
        self.btn_save_ng.clicked.connect(lambda: self.save_label("NG"))

        curation_layout.addWidget(self.btn_save_ok)
        curation_layout.addWidget(self.btn_save_ng)
        main_layout.addLayout(curation_layout)
        
        # Garante que inicie maximizado
        self.showMaximized()

    # ============================================================
    # MÉTODOS DE CONTROLE
    # ============================================================

    def start_monitoring(self):
        # Inicia a contagem de tempo
        self.capture_start_time = time.time()
        self.lbl_timer.setText("Latencia: Calculando...")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #ffaa33;")

        self.btn_start.setEnabled(False)
        self.btn_start.setText("Buscando na tela... (Minimizado)")
        self.lbl_sample.setText("Procurando barra Azul e Quadrado Verde...")
        self.lbl_ng.setText("Procurando barra Vermelha e Quadrado Verde...")
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
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

    # ============================================================
    # RESETS
    # ============================================================

    def _reset_aoi_info(self):
        self.lbl_board_value.setText("-")
        self.lbl_parts_value.setText("-")
        self.lbl_value_value.setText("-")
        self.current_aoi_info = {}
        self.current_analysis = None

    def _reset_confidence_panel(self):
        self.lbl_verdict.setText("Aguardando analise...")
        self.lbl_verdict.setStyleSheet(
            "color: #888888; font-size: 16px; font-weight: bold; padding: 4px; "
            "border: none; background: transparent;")
        for key, lbl in self.metric_labels.items():
            lbl.setText("-")
            lbl.setStyleSheet(
                "color: #ffffff; font-size: 11px; font-weight: bold; "
                "border: none; background: transparent;")
        self.lbl_reason.setText("")
        self.lbl_db_info.setText("")

    def _reset_reference_panel(self):
        self.lbl_ref.clear()
        self.lbl_ref.setText("Sem dados no banco")
        self.lbl_ref.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333333; color: #888888;")
        self.lbl_ref_title.setText("Referencia (Dataset)")
        self.lbl_ref_title.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self.lbl_ref_info.setText("")

    # ============================================================
    # UPDATES
    # ============================================================

    def _update_aoi_info(self, aoi_info: dict):
        board = aoi_info.get("board", "")
        parts = aoi_info.get("parts", "")
        value = aoi_info.get("value", "")

        self.lbl_board_value.setText(board if board else "-")
        self.lbl_parts_value.setText(parts if parts else "-")
        self.lbl_value_value.setText(value if value else "-")

        if board or parts or value:
            print(f"AOI Info - Board: {board} | Parts: {parts} | Value: {value}")

    def _update_reference_panel(self, analysis: dict):
        detail = analysis.get("detail", {})
        best_path = detail.get("db_best_path", "")
        best_label = detail.get("db_best_label", "")
        best_sim = detail.get("db_best_sim", 0)
        has_memory = detail.get("db_has_memory", False)

        if not has_memory or not best_path:
            self._reset_reference_panel()
            self.lbl_ref_info.setText(
                "Salve amostras com os botoes abaixo\n"
                "para ativar a consulta ao banco de dados.")
            return

        ref_img = cv2.imread(best_path)
        if ref_img is None:
            self._reset_reference_panel()
            self.lbl_ref_info.setText("Erro ao carregar imagem de referencia.")
            return

        px_ref = self.numpy_to_pixmap(ref_img)
        self.lbl_ref.setPixmap(px_ref.scaled(
            self.lbl_ref.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        if best_label == "NG":
            self.lbl_ref_title.setText("Referencia: DEFEITO (NG)")
            self.lbl_ref_title.setStyleSheet(
                "color: #ff5555; font-size: 12px; font-weight: bold;")
            self.lbl_ref.setStyleSheet(
                "background-color: #1a1a1a; border: 2px solid #ff5555;")
        else:
            self.lbl_ref_title.setText("Referencia: OK (Falha Falsa)")
            self.lbl_ref_title.setStyleSheet(
                "color: #55ff55; font-size: 12px; font-weight: bold;")
            self.lbl_ref.setStyleSheet(
                "background-color: #1a1a1a; border: 2px solid #55ff55;")

        sim_pct = f"{best_sim:.0%}"
        if best_label == "NG":
            explanation = (
                f"Similaridade: {sim_pct} - Classificada como DEFEITO.\n"
                f"Isso AUMENTA a confianca de defeito.")
        else:
            explanation = (
                f"Similaridade: {sim_pct} - Classificada como FALHA FALSA.\n"
                f"Isso AUMENTA a confianca de falha falsa.")

        self.lbl_ref_info.setText(explanation)

    def _update_confidence_panel(self, analysis: dict):
        verdict = analysis.get("verdict", "?")
        is_defect = analysis.get("is_defect", False)
        conf_float = analysis.get("confidence", 0.5)

        # Calcula a balança percentual
        conf_main = int(conf_float * 100)
        conf_opp = 100 - conf_main

        if is_defect:
            def_pct = conf_main
            ok_pct = conf_opp
            color_str = "#ff5555"
        else:
            ok_pct = conf_main
            def_pct = conf_opp
            color_str = "#55ff55"

        # Exibe ambas as porcentagens no painel principal
        self.lbl_verdict.setText(f"{verdict} - Defeito: {def_pct}% | Falha Falsa: {ok_pct}%")
        self.lbl_verdict.setStyleSheet(
            f"color: {color_str}; font-size: 16px; font-weight: bold; "
            "padding: 4px; border: none; background: transparent;")

        detail = analysis.get("detail", {})

        def get_metric_color(val, invert=False):
            v = float(val) if val is not None else 0
            if invert:
                v = 1.0 - v
            if v > 0.7:
                return "#ff5555"
            elif v > 0.4:
                return "#ffaa33"
            else:
                return "#55ff55"

        base_style = "font-size: 11px; font-weight: bold; border: none; background: transparent;"

        ssim_val = detail.get("ssim", 0)
        self.metric_labels["ssim"].setText(f"{ssim_val:.3f}")
        self.metric_labels["ssim"].setStyleSheet(
            f"color: {get_metric_color(ssim_val, invert=True)}; {base_style}")

        pct = detail.get("pct_changed", 0)
        self.metric_labels["pct_changed"].setText(f"{pct:.1%}")
        self.metric_labels["pct_changed"].setStyleSheet(
            f"color: {get_metric_color(pct / 0.15)}; {base_style}")

        edge = detail.get("edge_change", 0)
        self.metric_labels["edge_change"].setText(f"{edge:.1%}")
        self.metric_labels["edge_change"].setStyleSheet(
            f"color: {get_metric_color(edge / 0.08)}; {base_style}")

        hc = detail.get("hist_corr", 0)
        self.metric_labels["hist_corr"].setText(f"{hc:.3f}")
        self.metric_labels["hist_corr"].setStyleSheet(
            f"color: {get_metric_color(hc, invert=True)}; {base_style}")

        for key in ["local_score", "ctx_score", "db_score", "final_score"]:
            val = detail.get(key, 0)
            self.metric_labels[key].setText(f"{val:.2f}")
            color = get_metric_color(
                val / 0.6 if key != "final_score" else val / 0.45)
            fs = "12px" if key == "final_score" else "11px"
            self.metric_labels[key].setStyleSheet(
                f"color: {color}; font-size: {fs}; font-weight: bold; "
                f"border: none; background: transparent;")

        reason = analysis.get("reason", "")
        if reason:
            self.lbl_reason.setText(f"Justificativa: {reason}")

        db_mem = detail.get("db_has_memory", False)
        if db_mem:
            n = detail.get("db_neighbors", 0)
            vote = detail.get("db_vote", 0.5)
            sim = detail.get("db_best_sim", 0)
            self.lbl_db_info.setText(
                f"Base consultada: {n} vizinhos | Voto NG: {vote:.0%} | "
                f"Similaridade melhor match: {sim:.0%}")
        else:
            self.lbl_db_info.setText(
                "Sem dados no dataset. Salve amostras para melhorar a precisao!")

    # ============================================================
    # PROCESSAMENTO PRINCIPAL
    # ============================================================

    def process_aoi_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray,
                           aoi_info: dict):
        if sample_crop.size == 0 or ng_crop.size == 0:
            return

        # Restaura a janela de forma maximizada e a coloca em foco
        self.showMaximized()
        self.activateWindow()

        self.current_sample = sample_crop
        self.current_ng = ng_crop
        self.current_aoi_info = aoi_info
        self.current_analysis = None

        self._update_aoi_info(aoi_info)

        px_sample = self.numpy_to_pixmap(sample_crop)
        self.lbl_sample.setPixmap(px_sample.scaled(
            self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        raw_anomalies = detect_anomalies(sample_crop, ng_crop)
        img_ng_drawn = ng_crop.copy()

        if not raw_anomalies:
            self._reset_confidence_panel()
            self._reset_reference_panel()
            self.lbl_verdict.setText("Nenhuma anomalia detectada")
            self.lbl_verdict.setStyleSheet(
                "color: #55ff55; font-size: 16px; font-weight: bold; "
                "padding: 4px; border: none; background: transparent;")
            self.lbl_reason.setText(
                "A analise nao encontrou diferencas significativas entre as imagens.")
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
                conf_float = analysis.get("confidence", 0.5)
                conf_main = int(conf_float * 100)
                conf_opp = 100 - conf_main

                # Adiciona ambas as porcentagens também na marcação visual da imagem
                if is_real:
                    def_pct = conf_main
                    ok_pct = conf_opp
                    color = (0, 0, 255) # Vermelho em BGR
                else:
                    ok_pct = conf_main
                    def_pct = conf_opp
                    color = (0, 165, 255) # Laranja em BGR

                label_text = f"DEF:{def_pct}% | FALSO:{ok_pct}%"

                cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color, 2)

                font_scale = 0.45
                thickness = 1
                (tw, th), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(
                    img_ng_drawn, (x, y - th - 6), (x + tw + 4, y), color, -1)
                cv2.putText(
                    img_ng_drawn, label_text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                if (x, y, w, h) == biggest:
                    last_analysis = analysis

            if last_analysis:
                self.current_analysis = last_analysis
                self._update_confidence_panel(last_analysis)
                self._update_reference_panel(last_analysis)

        px_ng = self.numpy_to_pixmap(img_ng_drawn)
        self.lbl_ng.setPixmap(px_ng.scaled(
            self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

        self.btn_start.setEnabled(True)
        self.btn_start.setText("Nova Captura AOI")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)
        
        # Para a contagem e exibe o tempo total gasto na operação
        elapsed_time = time.time() - self.capture_start_time
        self.lbl_timer.setText(f"Latencia: {elapsed_time:.2f}s")
        self.lbl_timer.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #55ff55;")

    def save_label(self, label: str):
        if self.current_ng is None:
            return
        h1, w1 = self.current_sample.shape[:2]
        h2, w2 = self.current_ng.shape[:2]

        h_min = min(h1, h2)
        s_resized = cv2.resize(
            self.current_sample, (int(w1 * h_min / h1), h_min))
        n_resized = cv2.resize(
            self.current_ng, (int(w2 * h_min / h2), h_min))

        pair_img = np.hstack((s_resized, n_resized))

        filepath = DatasetManager.save_sample(
            pair_img, label,
            aoi_info=self.current_aoi_info,
            analysis=self.current_analysis
        )

        if filepath:
            print(f"Dataset salvo como {label}: {filepath}")
            self.lbl_ng.setText(f"SALVO NO DATASET: {label}")
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)

            self.neural_judge.reload_memory()
            self.lbl_db_info.setText(
                f"Memoria atualizada! "
                f"({len(self.neural_judge.memory.signatures_ok)} OK + "
                f"{len(self.neural_judge.memory.signatures_ng)} NG)")