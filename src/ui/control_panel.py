# src/ui/control_panel.py
"""
Módulo do Painel de Controle (Controller) e Console Duplo com Juiz Neural.
As responsabilidades visuais foram extraídas para control_panel_ui.py (SRP).
"""
import cv2
import numpy as np
import time
import socket
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

from src.services.screen_monitor import ScreenMonitor
from src.services.dataset_manager import DatasetManager
from src.core.inspection import detect_anomalies
from src.core.neural_judge import NeuralJudge
from src.services.network_receiver import NetworkReceiver

# Importando a View separada
from src.ui.control_panel_ui import ControlPanelUI

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
        
        self.is_locked = False 
        self.last_xp_ip = None 
        
        self.processor_monitor = ScreenMonitor()
        self.processor_monitor.layout_detected.connect(self.process_aoi_images)
        self.processor_monitor.log_updated.connect(self.update_status_log)

        self.network_receiver = NetworkReceiver(port=5001)
        self.network_receiver.log_updated.connect(self.update_status_log)
        self.network_receiver.image_received.connect(self.handle_network_image)
        self.network_receiver.command_received.connect(self.handle_physical_keyboard)
        self.network_receiver.start()

        self._setup_ui()

    def _setup_ui(self):
        """ Delega a montagem da interface para a classe de View """
        self.ui_builder = ControlPanelUI()
        self.ui_builder.setup_ui(self)

    # ============================================================
    # NOVOS MÉTODOS (REDE, TRAVA E FULL DUPLEX)
    # ============================================================

    def update_status_log(self, msg: str):
        print(msg)

    def handle_network_image(self, img_bgr: np.ndarray, ip: str):
        if self.is_locked:
            print(f"⚠️ Imagem da AOI ({ip}) ignorada: Aguardando decisão do operador atual.")
            return

        self.is_locked = True
        self.last_xp_ip = ip 
        self.capture_start_time = time.time()
        
        self.lbl_timer.setText("Latencia: Analisando Rede...")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #ffaa33;")
        self.btn_start.setEnabled(False)
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False) 

        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        
        self.lbl_sample.setText("Processando imagem da rede (Azul)...")
        self.lbl_ng.setText("Processando imagem da rede (Vermelha)...")

        self.processor_monitor.process_external_image(img_bgr)

    def handle_physical_keyboard(self, comando_xp: str):
        print(f"🔄 Replicando comando do teclado do XP: {comando_xp}")
        if comando_xp == "OK":
            self.save_label("OK", source="xp_keyboard")
        elif comando_xp == "NG":
            self.save_label("NG", source="xp_keyboard")

    def send_command_to_xp(self, tecla: str):
        if not self.last_xp_ip:
            print("⚠️ Impossível enviar comando: IP do XP desconhecido.")
            return
            
        try:
            print(f"👉 Mandando XP apertar '{tecla}'...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.last_xp_ip, 5000)) 
            
            mensagem = f"PRESS_{tecla}"
            s.send(mensagem.encode('utf-8'))
            s.close()
        except Exception as e:
            print(f"❌ Erro ao enviar comando para o XP ({self.last_xp_ip}): {e}")

    def closeEvent(self, event):
        self.network_receiver.stop()
        event.accept()

    # ============================================================
    # MÉTODOS DE CONTROLE (MSS MANUAL MANTIDO)
    # ============================================================

    def start_monitoring(self):
        if self.is_locked:
            return
            
        self.is_locked = True
        self.last_xp_ip = None 
        self.capture_start_time = time.time()
        self.lbl_timer.setText("Latencia: Calculando...")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #ffaa33;")

        self.btn_start.setEnabled(False)
        self.btn_start.setText("Buscando na tela... (Minimizado)")
        self.lbl_sample.setText("Procurando barra Azul e Quadrado Verde...")
        self.lbl_ng.setText("Procurando barra Vermelha e Quadrado Verde...")
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
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
    # RESETS E UPDATES
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

    def _update_aoi_info(self, aoi_info: dict):
        board = aoi_info.get("board", "")
        parts = aoi_info.get("parts", "")
        value = aoi_info.get("value", "")

        self.lbl_board_value.setText(board if board else "-")
        self.lbl_parts_value.setText(parts if parts else "-")
        self.lbl_value_value.setText(value if value else "-")

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

        raw_anomalies, aoi_epicenters = detect_anomalies(sample_crop, ng_crop)
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
                    part_metadata=aoi_info.get("parts", ""),
                    full_gab=sample_crop,
                    full_test=ng_crop,
                    box_x=x, box_y=y, box_w=w, box_h=h,
                    aoi_epicenters=aoi_epicenters
                )

                is_real = analysis["is_defect"]
                conf_float = analysis.get("confidence", 0.5)
                conf_main = int(conf_float * 100)
                conf_opp = 100 - conf_main

                if is_real:
                    def_pct = conf_main
                    ok_pct = conf_opp
                    color = (0, 0, 255) 
                else:
                    ok_pct = conf_main
                    def_pct = conf_opp
                    color = (0, 165, 255) 

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
        self.btn_start.setText("Capturar Local Manualmente (MSS)")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)
        self.btn_skip.setEnabled(True) 
        
        elapsed_time = time.time() - self.capture_start_time
        self.lbl_timer.setText(f"Latencia: {elapsed_time:.2f}s")
        self.lbl_timer.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #55ff55;")

    # ============================================================
    # SALVAMENTO E SINCRONIA FULL-DUPLEX (COM ACTIVE LEARNING)
    # ============================================================
    
    def skip_image(self):
        print("⏭️ Imagem descartada internamente. Aguardando próxima...")
        
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        
        self.is_locked = False
        self.lbl_sample.setText("Aguardando capturas da Rede...")
        self.lbl_ng.setText("Aguardando capturas da Rede...")

    def save_label(self, user_decision: str, source="button"):
        if self.current_ng is None:
            return

        if source == "button":
            if user_decision == "OK":
                self.send_command_to_xp("0")
            elif user_decision == "NG":
                self.send_command_to_xp("1")

        # --- AVALIAÇÃO DE ACTIVE LEARNING ---
        save_heavy_image = False
        
        if self.current_analysis:
            ia_decision = "NG" if self.current_analysis.get("is_defect") else "OK"
            ia_confidence = self.current_analysis.get("confidence", 0.0)
            
            # Se o humano discordou da IA...
            if user_decision != ia_decision:
                print(f"🧠 Active Learning: IA errou (Disse {ia_decision}, Humano disse {user_decision}). Salvando PNG para retreino.")
                save_heavy_image = True
            
            # Ou se a IA acertou, mas estava insegura (< 60% de confiança)
            elif ia_confidence < 0.60:
                print(f"🧠 Active Learning: IA insegura (Confiança {ia_confidence:.0%}). Salvando PNG para reforço.")
                save_heavy_image = True
            else:
                print(f"🧠 Active Learning: IA correta e segura (Confiança {ia_confidence:.0%}). Salvando apenas o JSON.")
        else:
            # Fallback de segurança: salva PNG se por algum motivo não houver análise
            save_heavy_image = True

        # ====================================

        filepath = DatasetManager.save_sample(
            ng_image=self.current_ng,
            label=user_decision,
            sample_image=self.current_sample,  
            aoi_info=self.current_aoi_info,
            analysis=self.current_analysis,
            save_images=save_heavy_image 
        )

        if filepath:
            tipo_salvo = "PNG + JSON" if save_heavy_image else "Apenas JSON"
            print(f"Dataset salvo como {user_decision} ({tipo_salvo}) (Origem: {source}): {filepath}")
            
            self.lbl_ng.setText(f"SALVO: {user_decision} ({tipo_salvo})")
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)
            self.btn_skip.setEnabled(False)

            self.neural_judge.reload_memory()
            self.lbl_db_info.setText(
                f"Memoria atualizada! "
                f"({len(self.neural_judge.memory.signatures_ok)} OK + "
                f"{len(self.neural_judge.memory.signatures_ng)} NG)")

        self.is_locked = False
        self.lbl_sample.setText("Aguardando capturas da Rede...")
        self.lbl_ng.setText("Aguardando capturas da Rede...")