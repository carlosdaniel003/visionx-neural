# src/ui/control_panel.py
"""
Módulo do Painel de Controle (Controller) e Console Duplo.
Ajuste: Correção do Active Learning. Peças perfeitas agora recebem confidence 1.0 e 
        o current_analysis é sempre preenchido, evitando o salvamento desnecessário de imagens no HD.
Integração MoE: Passando a categoria OCR para o Juiz Neural isolar a busca no K-NN.
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
from src.core.shift_gatekeeper import ShiftGatekeeper
from src.core.silkscreen_gatekeeper import SilkscreenGatekeeper
from src.services.network_receiver import NetworkReceiver

from src.ui.control_panel_ui import ControlPanelUI

# --- NOVO: Importando a Bússola Lexical ---
from src.utils.text_normalizer import normalize_aoi_text

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.current_analysis = None
        self.capture_start_time = 0.0
        
        # INICIANDO O ESQUADRÃO DE IA
        self.neural_judge = NeuralJudge()
        self.shift_gatekeeper = ShiftGatekeeper()
        self.silkscreen_gatekeeper = SilkscreenGatekeeper()
        
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
        self.ui_builder = ControlPanelUI()
        self.ui_builder.setup_ui(self)

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
        if comando_xp == "OK": self.save_label("OK", source="xp_keyboard")
        elif comando_xp == "NG": self.save_label("NG", source="xp_keyboard")

    def send_command_to_xp(self, tecla: str):
        if not self.last_xp_ip:
            print("⚠️ Impossível enviar comando: IP do XP desconhecido.")
            return
        try:
            print(f"👉 Mandando XP apertar '{tecla}'...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.last_xp_ip, 5000)) 
            s.send(f"PRESS_{tecla}".encode('utf-8'))
            s.close()
        except Exception as e:
            print(f"❌ Erro ao enviar comando para o XP ({self.last_xp_ip}): {e}")

    def closeEvent(self, event):
        self.network_receiver.stop()
        event.accept()

    def start_monitoring(self):
        if self.is_locked: return
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

    def _reset_aoi_info(self):
        self.lbl_board_value.setText("-")
        self.lbl_parts_value.setText("-")
        self.lbl_value_value.setText("-")
        if hasattr(self, 'lbl_category_value'):
            self.lbl_category_value.setText("-")
        self.current_aoi_info = {}
        self.current_analysis = None

    def _reset_confidence_panel(self):
        self.lbl_verdict.setText("Aguardando analise...")
        self.lbl_verdict.setStyleSheet("color: #888888; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")
        for key, lbl in self.metric_labels.items():
            lbl.setText("-")
            lbl.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        self.lbl_reason.setText("")
        self.lbl_db_info.setText("")

    def _reset_reference_panel(self):
        self.frame_dna.update_dna([], [])
        self.frame_radar.update_data({})
        self.frame_knn.update_data({})

    def _update_aoi_info(self, aoi_info: dict):
        self.lbl_board_value.setText(aoi_info.get("board", "-"))
        self.lbl_parts_value.setText(aoi_info.get("parts", "-"))
        
        # --- Lógica Visual Inteligente para Categoria ---
        if hasattr(self, 'lbl_category_value'):
            self.lbl_category_value.setText(aoi_info.get("category", "Unknown"))
            self.lbl_value_value.setText(aoi_info.get("value", "-"))
        else:
            cat = aoi_info.get("category", "Unknown")
            val = aoi_info.get("value", "-")
            self.lbl_value_value.setText(f"[{cat}] {val}")

    def _update_reference_panel(self, analysis: dict):
        detail = analysis.get("detail", {})
        self.frame_dna.update_dna(detail.get("embedding", []))
        self.frame_radar.update_data(detail)
        self.frame_knn.update_data(detail)

    def _update_confidence_panel(self, analysis: dict):
        verdict = analysis.get("verdict", "?")
        is_defect = analysis.get("is_defect", False)
        conf_float = analysis.get("confidence", 0.5)

        conf_main = int(conf_float * 100)
        conf_opp = 100 - conf_main
        color_str = "#ff5555" if is_defect else "#55ff55"
        def_pct, ok_pct = (conf_main, conf_opp) if is_defect else (conf_opp, conf_main)

        self.lbl_verdict.setText(f"{verdict} - Defeito: {def_pct}% | Falha Falsa: {ok_pct}%")
        self.lbl_verdict.setStyleSheet(f"color: {color_str}; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")

        detail = analysis.get("detail", {})

        def get_metric_color(val, invert=False):
            v = float(val) if val is not None else 0
            if invert: v = 1.0 - v
            if v > 0.7: return "#ff5555"
            elif v > 0.4: return "#ffaa33"
            else: return "#55ff55"

        base_style = "font-size: 11px; font-weight: bold; border: none; background: transparent;"

        ssim_val = detail.get("ssim", 0)
        self.metric_labels["ssim"].setText(f"{ssim_val:.3f}")
        self.metric_labels["ssim"].setStyleSheet(f"color: {get_metric_color(ssim_val, invert=True)}; {base_style}")

        pct = detail.get("pct_changed", 0)
        self.metric_labels["pct_changed"].setText(f"{pct:.1%}")
        self.metric_labels["pct_changed"].setStyleSheet(f"color: {get_metric_color(pct / 0.15)}; {base_style}")

        edge = detail.get("edge_change", 0)
        self.metric_labels["edge_change"].setText(f"{edge:.1%}")
        self.metric_labels["edge_change"].setStyleSheet(f"color: {get_metric_color(edge / 0.08)}; {base_style}")

        hc = detail.get("hist_corr", 0)
        self.metric_labels["hist_corr"].setText(f"{hc:.3f}")
        self.metric_labels["hist_corr"].setStyleSheet(f"color: {get_metric_color(hc, invert=True)}; {base_style}")

        for key in ["local_score", "ctx_score", "db_score", "final_score"]:
            val = detail.get(key, 0)
            self.metric_labels[key].setText(f"{val:.2f}")
            color = get_metric_color(val / 0.6 if key != "final_score" else val / 0.45)
            fs = "12px" if key == "final_score" else "11px"
            self.metric_labels[key].setStyleSheet(f"color: {color}; font-size: {fs}; font-weight: bold; border: none; background: transparent;")

        if analysis.get("reason", ""):
            self.lbl_reason.setText(f"Justificativa: {analysis.get('reason', '')}")

        if detail.get("db_has_memory", False):
            self.lbl_db_info.setText(f"Base consultada: {detail.get('db_neighbors', 0)} vizinhos | Voto NG: {detail.get('db_vote', 0.5):.0%} | Similaridade melhor match: {detail.get('db_best_sim', 0):.0%}")
        else:
            self.lbl_db_info.setText("Sem dados no dataset. Salve amostras para melhorar a precisao!")

    def process_aoi_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray, aoi_info: dict):
        if sample_crop.size == 0 or ng_crop.size == 0: return

        # --- NORMALIZAÇÃO DE OCR VIA FUZZY MATCHING ---
        raw_val = aoi_info.get("value", "")
        cat_name, norm_val = normalize_aoi_text(raw_val)
        aoi_info["category"] = cat_name
        aoi_info["value"] = norm_val
        # ----------------------------------------------

        # --- CORREÇÃO DO ESCORREGAMENTO FANTASMA DA JANELA ---
        if self.isMinimized():
            self.showNormal()  # Restaura se estiver minimizada
            self.showMaximized() # Maximiza
        self.raise_()         # Traz para a frente (z-index)
        self.activateWindow() # Dá o foco do teclado
        # -----------------------------------------------------

        self.current_sample = sample_crop
        self.current_ng = ng_crop
        self.current_aoi_info = aoi_info
        self.current_analysis = None
        self._update_aoi_info(aoi_info)

        px_sample = self.numpy_to_pixmap(sample_crop)
        self.lbl_sample.setPixmap(px_sample.scaled(self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        raw_anomalies, aoi_epicenters, global_box_info, gab_focus, test_focus = detect_anomalies(sample_crop, ng_crop)
        img_ng_drawn = ng_crop.copy()

        shift_data = self.shift_gatekeeper.check_global_shift(gab_focus, test_focus, global_box_info, aoi_info, aoi_epicenters)
        silk_data = self.silkscreen_gatekeeper.check_silkscreen_anomaly(gab_focus, test_focus, aoi_info, aoi_epicenters)

        is_gatekeeper_hit = False
        gatekeeper_reason = ""
        gatekeeper_type = ""

        if shift_data["is_critical_shift"]:
            is_gatekeeper_hit = True
            gatekeeper_type = "SHIFT"
            gatekeeper_reason = f"SHIFT CRITICO: {shift_data['shift_pixels']}px ({shift_data['shift_pct']:.1%})"
        elif silk_data.get("is_critical_silk"):
            is_gatekeeper_hit = True
            gatekeeper_type = "SILK"
            gatekeeper_reason = f"COMPONENTE INVERTIDO/ERRADO ({silk_data['silk_error_pct']:.1%})"

        biggest = None
        last_analysis = None

        if raw_anomalies:
            biggest = max(raw_anomalies, key=lambda b: b[2] * b[3])
            for (x, y, w, h) in raw_anomalies:
                suspect_gab = sample_crop[y:y+h, x:x+w]
                suspect_test = ng_crop[y:y+h, x:x+w]
                # AJUSTE MoE: Passando category_metadata para o Juiz
                analysis = self.neural_judge.verify_anomaly(
                    crop_gab=suspect_gab, crop_test=suspect_test, 
                    part_metadata=aoi_info.get("parts", ""),
                    category_metadata=aoi_info.get("category", ""),
                    full_gab=sample_crop, full_test=ng_crop, box_x=x, box_y=y, box_w=w, box_h=h, aoi_epicenters=aoi_epicenters
                )
                is_real = analysis["is_defect"]
                color = (0, 0, 255) if is_real else (0, 165, 255) 
                label_text = f"DEF:{int(analysis['confidence'] * 100)}% | FALSO:{100 - int(analysis['confidence'] * 100)}%"
                cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color, 2)
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_ng_drawn, (x, y - th - 6), (x + tw + 4, y), color, -1)
                cv2.putText(img_ng_drawn, label_text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                if (x, y, w, h) == biggest: last_analysis = analysis
        else:
            query_img = test_focus if test_focus is not None and test_focus.size > 0 else ng_crop
            part_name = aoi_info.get("parts", "")
            cat_name = aoi_info.get("category", "")
            # AJUSTE MoE: Passando category_name para a query direta
            db_result = self.neural_judge.memory.query_similar(query_img, part_name=part_name, category_name=cat_name)
            
            # AJUSTE ACTIVE LEARNING: Confidence 1.0 (100% de certeza que é Falha Falsa pois não tem manchas)
            last_analysis = {
                "is_defect": False, "confidence": 1.0, "score_text": "0%", "verdict": "FALHA FALSA", "reason": "Sem anomalias de textura (SSIM)",
                "detail": {
                    "local_score": 0.0, "ctx_score": 0.0, "db_score": db_result["vote_defect"], "final_score": 0.0,
                    "ssim": 1.0, "pct_changed": 0.0, "edge_change": 0.0, "hist_corr": 1.0,
                    "db_best_sim": db_result["best_similarity"], "db_best_path": db_result["best_match_path"],
                    "db_best_label": db_result["best_match_label"], "db_has_memory": db_result["has_memory"],
                    "db_neighbors": db_result["n_neighbors"], "db_vote": db_result["vote_defect"], "embedding": db_result["query_embedding"]
                }
            }

        if is_gatekeeper_hit:
            last_analysis["is_defect"] = True
            last_analysis["verdict"] = "DEFEITO REAL"
            last_analysis["reason"] = f"{gatekeeper_reason} | {last_analysis['reason']}"
            
            color_red = (0, 0, 255)
            valid_epicenter = aoi_epicenters[0] if aoi_epicenters and len(aoi_epicenters) > 0 else None
                
            if valid_epicenter and gatekeeper_type == "SHIFT":
                x, y, w, h = valid_epicenter
                cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color_red, 2)
                (tw, th), _ = cv2.getTextSize(gatekeeper_reason, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_ng_drawn, (x, y - th - 6), (x + tw + 4, y), color_red, -1)
                cv2.putText(img_ng_drawn, gatekeeper_reason, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            elif gatekeeper_type == "SILK" and "silk_box" in silk_data:
                x, y, w, h = silk_data["silk_box"]
                cv2.rectangle(img_ng_drawn, (x, y), (x+w, y+h), color_red, 2)
                (tw, th), _ = cv2.getTextSize(gatekeeper_reason, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_ng_drawn, (x, y - th - 6), (x + tw + 4, y), color_red, -1)
                cv2.putText(img_ng_drawn, gatekeeper_reason, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                h_img, w_img = img_ng_drawn.shape[:2]
                cv2.rectangle(img_ng_drawn, (5, 5), (w_img-5, h_img-5), color_red, 4)
                (tw, th), _ = cv2.getTextSize(gatekeeper_reason, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_ng_drawn, (5, 5), (5 + tw + 4, 5 + th + 6), color_red, -1)
                cv2.putText(img_ng_drawn, gatekeeper_reason, (7, 5 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # AJUSTE ACTIVE LEARNING: Sempre preenche o current_analysis para a hora de salvar
        self.current_analysis = last_analysis
        
        if not is_gatekeeper_hit and not raw_anomalies:
            self._reset_confidence_panel()
            self._reset_reference_panel()
            self.lbl_verdict.setText("Nenhuma anomalia detectada")
            self.lbl_verdict.setStyleSheet("color: #55ff55; font-size: 16px; font-weight: bold; padding: 4px; border: none; background: transparent;")
            self.lbl_reason.setText("A analise nao encontrou diferencas significativas ou deslocamentos.")
        else:
            self._update_confidence_panel(last_analysis)
            self._update_reference_panel(last_analysis)

        px_ng = self.numpy_to_pixmap(img_ng_drawn)
        self.lbl_ng.setPixmap(px_ng.scaled(self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self.btn_start.setText("Capturar Local Manualmente (MSS)")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)
        self.btn_skip.setEnabled(True) 
        
        elapsed_time = time.time() - self.capture_start_time
        self.lbl_timer.setText(f"Latencia: {elapsed_time:.2f}s")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #55ff55;")

    def skip_image(self):
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        self.is_locked = False
        self.lbl_sample.setText("Aguardando capturas da Rede...")
        self.lbl_ng.setText("Aguardando capturas da Rede...")
        self.btn_start.setEnabled(True)

    def save_label(self, user_decision: str, source="button"):
        if self.current_ng is None: return
        if source == "button":
            if user_decision == "OK": self.send_command_to_xp("0")
            elif user_decision == "NG": self.send_command_to_xp("1")

        save_heavy_image = False
        if self.current_analysis:
            ia_decision = "NG" if self.current_analysis.get("is_defect") else "OK"
            ia_confidence = self.current_analysis.get("confidence", 0.0)
            if user_decision != ia_decision or ia_confidence < 0.60:
                save_heavy_image = True
        else: save_heavy_image = True

        filepath = DatasetManager.save_sample(
            ng_image=self.current_ng, label=user_decision, sample_image=self.current_sample,  
            aoi_info=self.current_aoi_info, analysis=self.current_analysis, save_images=save_heavy_image 
        )

        if filepath:
            self.lbl_ng.setText(f"SALVO: {user_decision}")
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)
            self.btn_skip.setEnabled(False)
            self.neural_judge.reload_memory()
            self.lbl_db_info.setText(f"Memoria atualizada! ({len(self.neural_judge.memory.signatures_ok)} OK + {len(self.neural_judge.memory.signatures_ng)} NG)")

        self.is_locked = False
        self.lbl_sample.setText("Aguardando capturas da Rede...")
        self.lbl_ng.setText("Aguardando capturas da Rede...")
        self.btn_start.setEnabled(True)