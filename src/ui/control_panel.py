# src/ui/control_panel.py
"""
Módulo do Painel de Controle (Controller) e Console Duplo.
Ajuste SRP Seguro: A lógica de matemática visual e recortes do OpenCV foi 
isolada no EpicenterExtractor para limpar o peso desta classe.
Ajuste de Carrossel: Garante que os painéis Semantic e SSIM apareçam lado a lado
em vez de se sobrescreverem.
"""
import cv2
import numpy as np
import time
import socket
import os
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

try:
    import pytesseract
    possible_tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\cdaniel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Users\cdaniel\AppData\Local\Tesseract-OCR\tesseract.exe",
        r".\tesseract\tesseract.exe" 
    ]
    tesseract_found = False
    for path in possible_tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"👁️ OCR Ativado com sucesso em: {path}")
            break
    if not tesseract_found: print("⚠️ Executável do Tesseract não encontrado nos caminhos padrões.")
except ImportError:
    print("⚠️ Biblioteca 'pytesseract' não instalada no VENV. Rode: pip install pytesseract")

from src.services.screen_monitor import ScreenMonitor
from src.services.dataset_manager import DatasetManager
from src.core.inspection import detect_anomalies
from src.services.network_receiver import NetworkReceiver
from src.ui.control_panel_ui import ControlPanelUI
from src.utils.text_normalizer import normalize_aoi_text
from src.core.moe_orchestrator import MoEOrchestrator

# Importa o nosso novo Módulo de Extração de Matemática Pesada
from src.core.epicenter_extractor import EpicenterExtractor

class ImageRenderer:
    @staticmethod
    def draw_multilayer_boxes(img_bgr: np.ndarray, analysis: dict) -> np.ndarray:
        img_drawn = img_bgr.copy()
        all_boxes = analysis.get("all_boxes", {})
        color_map = {
            "shift":       {"color": (204, 50, 153), "label": "SHIFT"},
            "silk":        {"color": (0, 0, 255), "label": "SILK"},
            "ssim_local":  {"color": (255, 170, 0), "label": "SSIM-MICRO"},
            "ssim_global": {"color": (0, 255, 255), "label": "SSIM-MACRO"},
            "semantic":    {"color": (147, 20, 255), "label": "SEMANTICA"}
        }
        for engine_name, box in all_boxes.items():
            if box and engine_name in color_map:
                x, y, w, h = box
                color = color_map[engine_name]["color"]
                lbl_text = f"[{color_map[engine_name]['label']}]"
                if not analysis.get("is_defect", False):
                    color = (0, 165, 255) 
                    lbl_text += " FALSO"
                cv2.rectangle(img_drawn, (x, y), (x+w, y+h), color, 2)
                (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_drawn, (x, y - th - 6), (x + tw + 4, y), color, -1)
                cv2.putText(img_drawn, lbl_text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img_drawn

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.current_analysis = None
        self.capture_start_time = 0.0
        self.orchestrator = MoEOrchestrator()
        self.is_locked = False 
        self.last_xp_ip = None 
        
        self.processor_monitor = ScreenMonitor()
        self.processor_monitor.layout_detected.connect(self.process_aoi_images)

        self.network_receiver = NetworkReceiver(port=5001)
        self.network_receiver.image_received.connect(self.handle_network_image)
        self.network_receiver.command_received.connect(self.handle_physical_keyboard)
        self.network_receiver.start()

        self._setup_ui()

    def _setup_ui(self):
        self.ui_builder = ControlPanelUI()
        self.ui_builder.setup_ui(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'current_sample') and self.current_sample is not None and self.current_sample.size > 0:
            px_sample = self.numpy_to_pixmap(self.current_sample)
            if self.lbl_sample.width() > 0 and self.lbl_sample.height() > 0:
                self.lbl_sample.setPixmap(px_sample.scaled(self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if hasattr(self, 'current_ng') and self.current_ng is not None and self.current_ng.size > 0 and hasattr(self, 'current_analysis') and self.current_analysis:
            img_drawn = ImageRenderer.draw_multilayer_boxes(self.current_ng, self.current_analysis)
            px_ng = self.numpy_to_pixmap(img_drawn)
            if self.lbl_ng.width() > 0 and self.lbl_ng.height() > 0:
                self.lbl_ng.setPixmap(px_ng.scaled(self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def handle_network_image(self, img_bgr: np.ndarray, ip: str):
        if self.is_locked: return
        self.is_locked = True
        self.last_xp_ip = ip 
        self.capture_start_time = time.time()
        if self.isMinimized(): self.showNormal() 
        self.showMaximized() 
        self.raise_()         
        self.activateWindow()
        self.lbl_timer.setText("Latencia: Analisando Rede...")
        self.btn_start.setEnabled(False)
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False) 
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        self.processor_monitor.process_external_image(img_bgr)

    def handle_physical_keyboard(self, comando_xp: str):
        if comando_xp == "OK": self.save_label("OK", source="xp_keyboard")
        elif comando_xp == "NG": self.save_label("NG", source="xp_keyboard")

    def send_command_to_xp(self, tecla: str):
        if not self.last_xp_ip: return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.last_xp_ip, 5000)) 
            s.send(f"PRESS_{tecla}".encode('utf-8'))
            s.close()
        except: pass

    def closeEvent(self, event):
        self.network_receiver.stop()
        event.accept()

    def start_monitoring(self):
        if self.is_locked: return
        self.is_locked = True
        self.last_xp_ip = None 
        self.capture_start_time = time.time()
        self.lbl_timer.setText("Latencia: Calculando...")
        self.btn_start.setEnabled(False)
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
        self.lbl_category_value.setText("-")
        self.lbl_value_value.setText("-")
        self.current_aoi_info = {}
        self.current_analysis = None

    def _reset_confidence_panel(self):
        self.lbl_verdict.setText("AGUARDANDO PEÇA")
        self.lbl_verdict.setStyleSheet("color: #8b949e; font-size: 16px; font-weight: bold; border: none;")
        self.lbl_reason.setText("---")
        for key, lbl in self.metric_labels.items():
            lbl.setText("-")
        self.lbl_db_info.setText("Sem dados no momento.")

    def _reset_reference_panel(self):
        # Desliga todos os cards do scroll (vai religar apenas os ativos)
        for frame in ['frame_ssim_debug', 'frame_silk', 'frame_dna', 'frame_shift', 'frame_radar']:
            if hasattr(self, frame):
                getattr(self, frame).setVisible(False)
        if hasattr(self, 'frame_knn'): self.frame_knn.update_data({})

    def _update_aoi_info(self, aoi_info: dict):
        self.lbl_board_value.setText(aoi_info.get("board", "-"))
        self.lbl_parts_value.setText(aoi_info.get("parts", "-"))
        self.lbl_category_value.setText(aoi_info.get("category", "Unknown"))
        self.lbl_value_value.setText(aoi_info.get("value", "-"))

    def _update_reference_panel(self, analysis: dict):
        detail = analysis.get("detail", {})
        active_engines = analysis.get("active_engines", [])
        
        # O Layout é Horizontal (Carrossel). Nós apenas ativamos as janelas que o MoE usou.
        # Assim eles vão se alinhar lado a lado!
        
        # 1. Rota de Textura (Obrigatória em Missing e Solder)
        if "ssim_expert.py" in active_engines and hasattr(self, 'frame_ssim_debug'):
            self.frame_ssim_debug.update_data(detail)
            self.frame_ssim_debug.setVisible(True)
            
        # 2. Rota de Tinta
        if "silk_expert.py" in active_engines and hasattr(self, 'frame_silk'):
            self.frame_silk.update_data(detail)
            self.frame_silk.setVisible(True)
            
        # 3. Rota Estrutural do ORB (Esta era a que não estava entrando no empilhamento horizontal)
        if "semantic_expert.py" in active_engines and hasattr(self, 'frame_dna'):
            self.frame_dna.update_data(detail)
            self.frame_dna.setVisible(True)
            
        # 4. Rota Geométrica
        if "shift_expert.py" in active_engines and hasattr(self, 'frame_shift'):
            self.frame_shift.update_data(detail)
            self.frame_shift.setVisible(True)
            
        # 5. Radar Legado
        if not active_engines and hasattr(self, 'frame_radar'):
            self.frame_radar.update_data(detail)
            self.frame_radar.setVisible(True)
            
        # 6. Histórico (KNN) no rodapé
        if hasattr(self, 'frame_knn'):
            self.frame_knn.update_data(detail)

    def _update_confidence_panel(self, analysis: dict):
        verdict = analysis.get("verdict", "?")
        is_defect = analysis.get("is_defect", False)
        conf_float = analysis.get("confidence", 0.5)

        conf_main = int(conf_float * 100)
        conf_opp = 100 - conf_main
        color_str = "#ff7b72" if is_defect else "#3fb950"
        def_pct, ok_pct = (conf_main, conf_opp) if is_defect else (conf_opp, conf_main)

        self.lbl_verdict.setText(f"{verdict.upper()} • (Defeito: {def_pct}% | Falso: {ok_pct}%)")
        self.lbl_verdict.setStyleSheet(f"color: {color_str}; font-size: 16px; font-weight: bold; border: none;")

        if analysis.get("reason", ""): 
            self.lbl_reason.setText(f"Justificativa IA: {analysis.get('reason', '')}")

        detail = analysis.get("detail", {})
        
        metrics_mapping = {
            "ssim": f"{detail.get('ssim', 0):.3f}",
            "pct_changed": f"{detail.get('pct_changed', 0):.1%}",
            "hist_corr": f"{detail.get('hist_corr', 0):.3f}",
            "semantic_loss": f"{detail.get('semantic_loss', 0):.1%}",
            "local_score": f"{detail.get('local_score', 0):.2f}",
            "ctx_score": f"{detail.get('ctx_score', 0):.2f}",
            "final_score": f"{detail.get('final_score', 0):.2f}"
        }

        for key, text_value in metrics_mapping.items():
            if key in self.metric_labels:
                val_float = detail.get(key, 0.0)
                if key in ["ssim", "hist_corr"]:
                    color = "#3fb950" if val_float > 0.6 else ("#ffd33d" if val_float > 0.4 else "#ff7b72")
                else:
                    color = "#ff7b72" if val_float > 0.6 else ("#ffd33d" if val_float > 0.3 else "#3fb950")
                
                font_sz = "14px" if key == "final_score" else "12px"
                self.metric_labels[key].setStyleSheet(f"color: {color}; font-size: {font_sz}; font-weight: bold; border: none; background: transparent;")
                self.metric_labels[key].setText(text_value)

        if detail.get("has_memory", False) or detail.get("db_has_memory", False):
            vote = detail.get('vote_defect', detail.get('db_vote', 0.5))
            sim = detail.get('best_similarity', detail.get('db_best_sim', 0.0))
            self.lbl_db_info.setText(f"Voto Dataset: {vote:.0%} NG | Match Visual: {sim:.0%}")
        else:
            self.lbl_db_info.setText("Sem dados anteriores. Salve amostras para treinar.")

    def process_aoi_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray, aoi_info: dict):
        if sample_crop.size == 0 or ng_crop.size == 0: return

        raw_val = aoi_info.get("value", "")
        cat_name, norm_val = normalize_aoi_text(raw_val)
        aoi_info["category"] = cat_name
        aoi_info["value"] = norm_val

        if self.isMinimized(): self.showNormal() 
        self.showMaximized() 
        self.raise_()         
        self.activateWindow() 

        self.current_sample = sample_crop
        self.current_ng = ng_crop
        self.current_aoi_info = aoi_info
        self.current_analysis = None
        self._update_aoi_info(aoi_info)

        px_sample = self.numpy_to_pixmap(sample_crop)
        if self.lbl_sample.width() > 0 and self.lbl_sample.height() > 0:
            self.lbl_sample.setPixmap(px_sample.scaled(self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        raw_anomalies, old_epicenters, global_box_info, gab_focus, test_focus = detect_anomalies(sample_crop, ng_crop)
        
        # =====================================================================
        # DELEGAÇÃO SRP: O Módulo Externo faz o trabalho pesado matemático!
        # =====================================================================
        real_epicenters, focus_gab, focus_ng = EpicenterExtractor.extract_focus(
            sample_crop, ng_crop, old_epicenters, global_box_info
        )

        # Atualiza a UI Visual das Lupa Foco com o resultado retornado
        if focus_gab.size > 0 and hasattr(self, 'lbl_sample_focus') and self.lbl_sample_focus.width() > 0:
            px_focus_gab = self.numpy_to_pixmap(focus_gab)
            self.lbl_sample_focus.setPixmap(px_focus_gab.scaled(self.lbl_sample_focus.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            if hasattr(self, 'lbl_sample_focus'): self.lbl_sample_focus.setText("Inválido/Sem Foco")

        if focus_ng.size > 0 and hasattr(self, 'lbl_ng_focus') and self.lbl_ng_focus.width() > 0:
            px_focus_ng = self.numpy_to_pixmap(focus_ng)
            self.lbl_ng_focus.setPixmap(px_focus_ng.scaled(self.lbl_ng_focus.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            if hasattr(self, 'lbl_ng_focus'): self.lbl_ng_focus.setText("Inválido/Sem Foco")

        # O ORQUESTRADOR recebe a caixa finalizada e limpa!
        analysis = self.orchestrator.inspect(sample_crop, ng_crop, raw_anomalies, aoi_info, global_box_info, real_epicenters)
        self.current_analysis = analysis

        img_ng_drawn = ImageRenderer.draw_multilayer_boxes(ng_crop, analysis)

        if not analysis.get("all_boxes") and not analysis.get("is_defect"):
             self.lbl_verdict.setText("NENHUMA ANOMALIA DETECTADA")
             self.lbl_verdict.setStyleSheet("color: #3fb950; font-size: 16px; font-weight: bold; border: none;")
             self.lbl_reason.setText("A análise matemática não encontrou diferenças críticas.")
             self._update_reference_panel(analysis)
        else:
             self._update_confidence_panel(analysis)
             self._update_reference_panel(analysis)

        px_ng = self.numpy_to_pixmap(img_ng_drawn)
        if self.lbl_ng.width() > 0 and self.lbl_ng.height() > 0:
            self.lbl_ng.setPixmap(px_ng.scaled(self.lbl_ng.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self.btn_start.setText("Capturar Local Manualmente")
        self.btn_save_ok.setEnabled(True)
        self.btn_save_ng.setEnabled(True)
        self.btn_skip.setEnabled(True) 
        
        elapsed_time = time.time() - self.capture_start_time
        self.lbl_timer.setText(f"Latência: {elapsed_time:.2f}s")
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #3fb950;")

    def skip_image(self):
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        self.is_locked = False
        self.btn_start.setEnabled(True)

    def save_label(self, user_decision: str, source="button"):
        if self.current_ng is None: return
        if source == "button":
            if user_decision == "OK": self.send_command_to_xp("0")
            elif user_decision == "NG": self.send_command_to_xp("1")

        save_heavy_image = True
        filepath = DatasetManager.save_sample(
            ng_image=self.current_ng, label=user_decision, sample_image=self.current_sample,  
            aoi_info=self.current_aoi_info, analysis=self.current_analysis, save_images=save_heavy_image 
        )

        if filepath:
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)
            self.btn_skip.setEnabled(False)
            self.orchestrator.reload_memory() 

        self.is_locked = False
        self.btn_start.setEnabled(True)