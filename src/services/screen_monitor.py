"""
Módulo responsável pela captura de tela e detecção avançada (ORB + Homografia).
Imune a variações de escala (zoom) e rotação, com correção de DPI para a Interface.
"""
import cv2
import numpy as np
import mss
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication # NOVO: Importação crucial para corrigir o DPI
from src.config.settings import settings

class ScreenMonitor(QThread):
    pattern_found = pyqtSignal(int, int, int, int)
    pattern_lost = pyqtSignal()
    log_updated = pyqtSignal(str) 
    crop_updated = pyqtSignal(np.ndarray) 

    def __init__(self):
        super().__init__()
        self.running = True
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.template = None
        self.kp_template = None
        self.des_template = None
        self.template_w = 0
        self.template_h = 0
        self._load_template()

    def _load_template(self):
        path = str(settings.TEMPLATE_IMAGE_PATH)
        try:
            file_bytes = np.fromfile(path, dtype=np.uint8)
            self.template = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if self.template is not None:
                self.template_h, self.template_w = self.template.shape
                self.kp_template, self.des_template = self.orb.detectAndCompute(self.template, None)
            else:
                print(f"AVISO: Falha ao decodificar template_padrao.png")
        except Exception as e:
            print(f"AVISO: Imagem padrão não encontrada ou erro de leitura: {e}")

    def run(self):
        if self.des_template is None:
            self.log_updated.emit("ERRO: Template sem features ORB detectadas!")
            return

        MIN_MATCH_COUNT = 15 

        with mss.mss() as sct:
            monitor = sct.monitors[1] 
            frame_count = 0 

            while self.running:
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # Usado para o recorte
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) # Usado para a matemática

                kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)
                
                max_matches = 0
                found_rect = None

                if des_frame is not None and len(des_frame) > 0:
                    matches = self.matcher.match(self.des_template, des_frame)
                    good_matches = [m for m in matches if m.distance < 50]
                    max_matches = len(good_matches)

                    if max_matches >= MIN_MATCH_COUNT:
                        src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if matrix is not None:
                            pts = np.float32([[0, 0], [0, self.template_h - 1], [self.template_w - 1, self.template_h - 1], [self.template_w - 1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, matrix)
                            x, y, w, h = cv2.boundingRect(np.int32(dst))
                            
                            if 20 < w < monitor["width"] and 20 < h < monitor["height"]:
                                found_rect = (x, y, w, h)

                # --- PROCESSAMENTO DO ALVO ENCONTRADO ---
                if found_rect:
                    x, y, w, h = found_rect

                    # 1. Trava de Segurança para o Recorte (Clamping)
                    # Usa os pixels reais (físicos) para garantir que o OpenCV nunca crashe
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(frame_bgr.shape[1], x + w)
                    y2 = min(frame_bgr.shape[0], y + h)

                    if x2 > x1 and y2 > y1:
                        crop = frame_bgr[y1:y2, x1:x2].copy()
                        self.crop_updated.emit(crop)

                    # 2. Correção de Escala do Windows (DPI)
                    # Lê em tempo real a escala que seu Windows está usando
                    screen = QGuiApplication.primaryScreen()
                    dpi_scale = screen.devicePixelRatio() if screen else 1.0

                    # Desfaz a matemática do Windows para a HUD desenhar no local perfeito
                    ui_x = int(x / dpi_scale)
                    ui_y = int(y / dpi_scale)
                    ui_w = int(w / dpi_scale)
                    ui_h = int(h / dpi_scale)

                    self.pattern_found.emit(ui_x, ui_y, ui_w, ui_h)
                else:
                    self.pattern_lost.emit()

                # --- ATUALIZAÇÃO DO LOG NA TELA ---
                frame_count += 1
                if frame_count % 15 == 0:
                    if found_rect:
                        msg = f"Radar ORB: ALVO DETECTADO! 🎯 ({max_matches} conexoes)"
                    else:
                        msg = f"Radar ORB: Buscando... (Max: {max_matches}/{MIN_MATCH_COUNT} pts)"
                    self.log_updated.emit(msg)

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()