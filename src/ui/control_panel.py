# src/ui/control_panel.py
"""
Módulo do Painel de Controle (Controller) e Console Duplo.
Ajuste de Machine Learning: Implementado "Hard Negative Mining" (Filtro de Discordância).
O sistema agora só salva imagens no HD para treinamento caso o Operador Humano
discorde do Veredito da IA. Se ambos concordarem, a imagem é descartada para poupar disco.
"""
import cv2
import numpy as np
import time
import socket
import os
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

try:
    import pytesseract
    possible_tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\cdaniel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Users\cdaniel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
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
        self._inspection_images_visible = False
        self.capture_start_time = 0.0
        self.capture_start_source = "idle"
        self.last_analysis_time_seconds = 0.0
        self.orchestrator = MoEOrchestrator()
        self.is_locked = False 
        self.last_xp_ip = None 
        
        self.processor_monitor = ScreenMonitor()
        self.processor_monitor.layout_detected.connect(self.process_aoi_images)

        self.network_receiver = NetworkReceiver(port=5001)
        self.network_receiver.image_received.connect(self.handle_network_image)
        self.network_receiver.command_received.connect(self.handle_physical_keyboard)
        self.network_receiver.log_updated.connect(self.update_network_status)
        
        self.network_receiver.start()

        self._setup_ui()

    def _setup_ui(self):
        self.ui_builder = ControlPanelUI()
        self.ui_builder.setup_ui(self)
        
        if hasattr(self, 'btn_light_mid'):
            self.btn_light_mid.clicked.connect(
                lambda: self.change_lighting("MID", "odin_control")
            )
            self.btn_light_side.clicked.connect(
                lambda: self.change_lighting("SIDE", "odin_control")
            )
            self.btn_light_top.clicked.connect(
                lambda: self.change_lighting("TOP", "odin_control")
            )
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        # As setas de iluminação usam QShortcut em nível de janela.
        # keyPressEvent permanece somente como fallback padrão do QWidget.
        super().keyPressEvent(event)

    def change_lighting(self, light_mode: str, source: str):
        light_mode = str(light_mode or "").strip().upper()
        command_by_light = {
            "TOP": "LEFT",
            "SIDE": "DOWN",
            "MID": "RIGHT",
        }
        command = command_by_light.get(light_mode)
        if command is None:
            self.update_network_status(
                f"Falha ao ajustar iluminação: modo inválido '{light_mode}'."
            )
            return False

        normalized_source = str(source or "").strip().lower()

        # Durante a sequência automática de adesivo, os controles manuais do
        # próprio ODIN não podem trocar a iluminação no meio de TOP/MID.
        if (
            bool(
                getattr(
                    self,
                    "adhesive_multilight_automation_active",
                    False,
                )
            )
            and normalized_source in {
                "local",
                "button",
                "odin_control",
                "odin_keyboard",
            }
        ):
            self.update_network_status(
                "Captura automática SIDE/TOP/MID em andamento; "
                "controle manual de iluminação temporariamente bloqueado."
            )
            return False

        if normalized_source in {
            "local",
            "button",
            "odin_control",
            "odin_keyboard",
        }:
            if not self.send_command_to_xp(command):
                return False

        if hasattr(self, 'lbl_light_value'):
            self.lbl_light_value.setText(light_mode)

        self.update_brain_status(
            f"Iluminação ajustada: {light_mode}",
            False,
        )
        return True

    def update_network_status(self, message: str):
        if hasattr(self.ui_builder, 'lbl_status_network'):
            if "Erro" in message or "Falha" in message or "❌" in message:
                self.ui_builder.lbl_status_network.setStyleSheet("color: #ff7b72; font-size: 11px; font-weight: bold; border: none;")
            elif "ALERTA" in message or "⚡" in message:
                self.ui_builder.lbl_status_network.setStyleSheet("color: #ffd33d; font-size: 11px; font-weight: bold; border: none;")
            else:
                self.ui_builder.lbl_status_network.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold; border: none;")
            self.ui_builder.lbl_status_network.setText(message)

    def update_brain_status(self, message: str, is_active: bool = False):
        if hasattr(self.ui_builder, 'lbl_status_brain'):
            color = "#58a6ff" if is_active else "#8b949e"
            self.ui_builder.lbl_status_brain.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; border: none;")
            self.ui_builder.lbl_status_brain.setText(message)

    def update_history_status(self, label: str, source: str):
        if hasattr(self.ui_builder, 'lbl_status_history'):
            color = "#ff7b72" if label == "NG" else "#3fb950"
            normalized_source = str(source or "").strip().lower()
            src_text = (
                "Autônomo"
                if normalized_source in {"auto", "production_auto"}
                else (
                    "Operador IA"
                    if normalized_source == "button"
                    else "Operador AOI"
                )
            )
            msg = f"💾 Última Peça: {label} ({src_text})"
            self.ui_builder.lbl_status_history.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; border: none;")
            self.ui_builder.lbl_status_history.setText(msg)

    def _safe_maximize(self):
        if self.isMinimized():
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents() 

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not bool(getattr(self, "_inspection_images_visible", False)):
            return

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
        self.is_locked = True
        self.last_xp_ip = ip
        received_at = float(
            getattr(
                self.network_receiver,
                "last_delivered_image_received_at",
                0.0,
            )
            or 0.0
        )
        self.capture_start_time = (
            received_at if received_at > 0.0 else time.perf_counter()
        )
        self.capture_start_source = "network_payload_received"
        
        self._safe_maximize()
        
        self.update_brain_status("🧠 Recebendo da Rede...", True)
        
        self.lbl_timer.setText("Analisando...")
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
        elif comando_xp in ["MID", "SIDE", "TOP"]:
            self.change_lighting(comando_xp, source="network")

    def send_command_to_xp(self, tecla: str):
        command = f"PRESS_{str(tecla).strip().upper()}"
        if not self.last_xp_ip:
            self.update_network_status(
                f"Falha ao enviar {command}: AOI Windows XP ainda não identificada."
            )
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.last_xp_ip, 5000))
                s.sendall(command.encode("utf-8"))
            self.update_network_status(
                f"{command} enviado para AOI XP ({self.last_xp_ip}:5000)."
            )
            return True
        except Exception as exc:
            self.update_network_status(
                f"Falha ao enviar {command} para AOI XP: {exc}"
            )
            print(
                f"Falha ao enviar {command} para "
                f"{self.last_xp_ip}:5000: {exc}"
            )
            return False

    def closeEvent(self, event):
        self.network_receiver.stop()
        event.accept()

    def start_monitoring(self):
        self.is_locked = True
        self.last_xp_ip = None
        # O clique do operador não faz parte do tempo de análise. O cronômetro
        # começa somente quando um frame MSS válido foi realmente capturado.
        self.capture_start_time = 0.0
        self.capture_start_source = "local_capture_pending"
        
        self.update_brain_status("🧠 Capturando Tela (Local)...", True)
        
        self.lbl_timer.setText("Aguardando captura...")
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

    def _clear_inspection_images(self):
        """Remove somente os visuais da peça anterior, preservando a evidência técnica."""
        self._inspection_images_visible = False

        for name, placeholder in (
            ("lbl_sample", "Aguardando peça"),
            ("lbl_sample_focus", "Sem foco"),
            ("lbl_ng", "Aguardando peça"),
            ("lbl_ng_focus", "Sem foco"),
        ):
            label = getattr(self, name, None)
            if label is None:
                continue
            try:
                label.clear()
                label.setText(placeholder)
            except Exception:
                pass

        preview = getattr(self, "lbl_capture_evidence_preview", None)
        if preview is not None:
            try:
                if hasattr(preview, "clear_source_image"):
                    preview.clear_source_image("Aguardando captura")
                else:
                    preview.clear()
                    preview.setText("Aguardando captura")
            except Exception:
                pass

    def _reset_confidence_panel(self):
        self._clear_inspection_images()
        self.lbl_verdict.setText("AGUARDANDO PEÇA")
        self.lbl_verdict.setStyleSheet("color: #8b949e; font-size: 16px; font-weight: bold; border: none;")
        self.lbl_reason.setText("---")
        for key, lbl in self.metric_labels.items():
            lbl.setText("-")
        self.lbl_db_info.setText("Sem dados no momento.")

    def _reset_reference_panel(self):
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
        
        if "ssim_expert.py" in active_engines and hasattr(self, 'frame_ssim_debug'):
            self.frame_ssim_debug.update_data(detail)
            self.frame_ssim_debug.setVisible(True)
            
        if "silk_expert.py" in active_engines and hasattr(self, 'frame_silk'):
            self.frame_silk.update_data(detail)
            self.frame_silk.setVisible(True)
            
        if "semantic_expert.py" in active_engines and hasattr(self, 'frame_dna'):
            self.frame_dna.update_data(detail)
            self.frame_dna.setVisible(True)
            
        if "shift_expert.py" in active_engines and hasattr(self, 'frame_shift'):
            self.frame_shift.update_data(detail)
            self.frame_shift.setVisible(True)
            
        if not active_engines and hasattr(self, 'frame_radar'):
            self.frame_radar.update_data(detail)
            self.frame_radar.setVisible(True)
            
        if hasattr(self, 'frame_knn'):
            self.frame_knn.update_data(detail)

    def _update_confidence_panel(self, analysis: dict):
        verdict = analysis.get("verdict", "?")
        is_defect = analysis.get("is_defect", False)
        conf_float = analysis.get("confidence", 0.5)

        conf_main = int(conf_float * 100)
        conf_opp = 100 - conf_main
        review_required = str(verdict or "").strip().upper() == "REVISÃO OBRIGATÓRIA"
        color_str = "#ff7b72" if (is_defect or review_required) else "#3fb950"
        if review_required:
            def_pct, ok_pct = 50, 50
        else:
            def_pct, ok_pct = (
                (conf_main, conf_opp)
                if is_defect
                else (conf_opp, conf_main)
            )

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
            self.lbl_db_info.setText("Sem dados no momento.")

    def process_aoi_images(self, sample_crop: np.ndarray, ng_crop: np.ndarray, aoi_info: dict):
        if sample_crop.size == 0 or ng_crop.size == 0: return

        # Para MSS, o início real é o frame que acabou de ser capturado, não o
        # clique em "Capturar local". Para rede, o receptor já registrou o
        # término do recebimento do payload completo antes de emitir o frame.
        if str(getattr(self, "capture_cycle_source", "") or "") == "local":
            monitor = getattr(self, "monitor", None)
            received_at = float(
                getattr(monitor, "last_capture_received_at", 0.0) or 0.0
            )
            self.capture_start_time = (
                received_at if received_at > 0.0 else time.perf_counter()
            )
            self.capture_start_source = "local_mss_frame_received"
        elif float(getattr(self, "capture_start_time", 0.0) or 0.0) <= 0.0:
            self.capture_start_time = time.perf_counter()
            self.capture_start_source = "process_entry_fallback"

        notify_cycle = getattr(
            self,
            "notify_production_cycle_started",
            None,
        )
        if callable(notify_cycle):
            notify_cycle()

        self._inspection_images_visible = True
        self.update_brain_status("🧠 Processando Tensores Matemáticos...", True)

        raw_val = aoi_info.get("value", "")
        cat_name, norm_val = normalize_aoi_text(raw_val)
        aoi_info["category"] = cat_name
        aoi_info["value"] = norm_val

        self._safe_maximize()

        self.current_sample = sample_crop
        self.current_ng = ng_crop
        self.current_aoi_info = aoi_info
        self.current_analysis = None
        self._update_aoi_info(aoi_info)

        px_sample = self.numpy_to_pixmap(sample_crop)
        if self.lbl_sample.width() > 0 and self.lbl_sample.height() > 0:
            self.lbl_sample.setPixmap(px_sample.scaled(self.lbl_sample.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        raw_anomalies, old_epicenters, global_box_info, gab_focus, test_focus = detect_anomalies(sample_crop, ng_crop)
        
        real_epicenters, focus_gab, focus_ng = EpicenterExtractor.extract_focus(
            sample_crop, ng_crop, old_epicenters, global_box_info
        )

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

        analysis = self.orchestrator.inspect(sample_crop, ng_crop, raw_anomalies, aoi_info, global_box_info, real_epicenters)
        self.current_analysis = analysis

        img_ng_drawn = ImageRenderer.draw_multilayer_boxes(ng_crop, analysis)

        multilight_pending = bool(
            getattr(self, "adhesive_multilight_pending_start", False)
        )
        if multilight_pending:
            # Em adesivo, SIDE é somente a primeira observação. O julgamento
            # final só existe depois que TOP e MID também forem analisadas.
            self.lbl_verdict.setText("ADESIVO • AGUARDANDO TOP/MID")
            self.lbl_verdict.setStyleSheet(
                "color: #ffd33d; font-size: 16px; font-weight: bold; "
                "border: none;"
            )
            self.lbl_reason.setText(
                "SIDE concluída. O resultado final será calculado após "
                "as três iluminações."
            )
        elif not analysis.get("all_boxes") and not analysis.get("is_defect"):
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

        # "Tempo de análise" termina quando o resultado já foi entregue aos
        # widgets e o Qt processou a pintura pendente. Excluímos input do
        # operador para não introduzir ações humanas dentro da medição.
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        analysis_displayed_at = time.perf_counter()
        elapsed_time = max(
            0.0,
            analysis_displayed_at
            - float(getattr(self, "capture_start_time", 0.0) or 0.0),
        )
        self.last_analysis_time_seconds = elapsed_time

        detail = analysis.setdefault("detail", {})
        detail["analysis_time_seconds"] = float(elapsed_time)
        detail["analysis_time_start_source"] = str(
            getattr(self, "capture_start_source", "") or ""
        )
        detail["analysis_time_contract"] = (
            "imagem recebida/capturada -> resultado pintado na interface"
        )

        self.lbl_timer.setText(f"{elapsed_time:.2f} s")
        self.lbl_timer.setToolTip(
            "Tempo real desde o recebimento/captura do frame até o resultado "
            "estar atualizado na interface."
        )
        self.lbl_timer.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; font-weight: bold; color: #3fb950;")

        current_mode = self.combo_mode.currentText()

        if current_mode == "Modo Produção":
            # O Modo Produção v1 não decide no mesmo instante em que a análise
            # termina. Primeiro toda a interface é renderizada; depois o
            # ProductionAutonomyController percorre a tela e só então aplica a
            # política: FALHA FALSA -> 0 automático; NG/revisão -> operador.
            self.update_brain_status(
                "Análise concluída • preparando apresentação do Modo Produção.",
                True,
            )
            if not bool(
                getattr(self, "adhesive_multilight_pending_start", False)
            ):
                notify = getattr(
                    self,
                    "notify_production_analysis_ready",
                    None,
                )
                if callable(notify):
                    notify(analysis)
            
        elif current_mode == "Modo Sombra":
            self.update_brain_status("⏳ Aguardando Decisão Humana no Teclado XP...", True)
            self.btn_start.setText("Nova Captura (Forçar)")
            self.btn_start.setEnabled(True) 
            self.btn_skip.setEnabled(True)
            
        else:
            self.update_brain_status("⏳ Aguardando Operador na Tela da IA...", True)
            self.btn_start.setText("Nova Captura (Descartar Atual)")
            self.btn_start.setEnabled(True) 
            self.btn_save_ok.setEnabled(True)
            self.btn_save_ng.setEnabled(True)
            self.btn_skip.setEnabled(True) 

    def skip_image(self):
        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()
        self.is_locked = False
        
        self.btn_start.setText("Capturar Local (MSS)")
        self.btn_start.setEnabled(True)
        
        self.update_brain_status("⏳ Sistema Ocioso", False)

    def prepare_for_next_network_image(self):
        """Limpa a peça julgada imediatamente e deixa o painel aguardando a AOI."""
        for label, placeholder in (
            (self.lbl_sample, "Aguardando próxima imagem"),
            (self.lbl_sample_focus, "Sem Foco"),
            (self.lbl_ng, "Aguardando próxima imagem"),
            (self.lbl_ng_focus, "Sem Foco"),
        ):
            label.clear()
            label.setText(placeholder)

        self._reset_confidence_panel()
        self._reset_reference_panel()
        self._reset_aoi_info()

        self.current_sample = None
        self.current_ng = None
        self.current_analysis = None

        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self.btn_start.setText("Capturar Local (MSS)")
        self.btn_start.setEnabled(True)

        self.lbl_timer.setText("Aguardando imagem...")
        self.update_brain_status("Aguardando próxima imagem da AOI...", True)

    def save_label(self, user_decision: str, source="button"):
        if self.current_ng is None: return
        
        # Envia ordem para a máquina avançar, independentemente de salvar a foto.
        # O Modo Produção precisa saber se o comando realmente saiu antes de
        # contabilizar a placa como julgamento autônomo.
        self.last_decision_command_success = None
        if source in {"button", "auto", "production_auto"}:
            command = "0" if user_decision == "OK" else "1" if user_decision == "NG" else ""
            if command:
                sent = bool(self.send_command_to_xp(command))
                self.last_decision_command_success = sent
                if source == "production_auto" and not sent:
                    self.is_locked = True
                    self.update_brain_status(
                        "Falha ao enviar decisão automática ao XP. "
                        "Aguardando operador.",
                        True,
                    )
                    return False

        # =========================================================
        # FILTRO DE DISCORDÂNCIA (HARD NEGATIVE MINING)
        # =========================================================
        ia_decision = "OK"
        if self.current_analysis and self.current_analysis.get("is_defect", False):
            ia_decision = "NG"

        if ia_decision == user_decision:
            # IA e Operador concordaram perfeitamente!
            # Não salvamos nada no HD para manter o disco limpo e focar só no que é difícil.
            self.is_locked = False
            self.btn_start.setText("Capturar Local (MSS)")
            self.btn_start.setEnabled(True)
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)
            self.btn_skip.setEnabled(False)
            
            self.update_brain_status("⏳ Sistema Ocioso", False)
            self.update_history_status(f"{user_decision} (Concordou / Descartado)", source)
            return

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
        self.btn_start.setText("Capturar Local (MSS)")
        self.btn_start.setEnabled(True)
        
        self.update_brain_status("⏳ Sistema Ocioso", False)
        self.update_history_status(user_decision, source)