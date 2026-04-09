"""
Módulo responsável pela captura de tela e detecção do Layout da IoT.
Modo One-Shot: Tira um único print ao encontrar o layout e encerra a thread para poupar CPU.
"""
import cv2
import numpy as np
import mss
from PyQt6.QtCore import QThread, pyqtSignal
from src.config.settings import settings

class ScreenMonitor(QThread):
    log_updated = pyqtSignal(str) 
    layout_detected = pyqtSignal(np.ndarray, np.ndarray) 

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1] 
            frame_count = 0 

            while self.running:
                screenshot = sct.grab(monitor)
                frame_bgra = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                # 1. Cria as máscaras de cor
                mask_blue = cv2.inRange(hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
                mask_red1 = cv2.inRange(hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
                mask_red2 = cv2.inRange(hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                mask_green = cv2.inRange(hsv, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)

                # 2. Verifica se a Assinatura (Barra Azul e Vermelha) existe
                blue_cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                red_cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                has_blue_header = any(cv2.contourArea(c) > 2000 for c in blue_cnts)
                has_red_header = any(cv2.contourArea(c) > 2000 for c in red_cnts)

                if has_blue_header and has_red_header:
                    # 3. Busca os quadrados verdes
                    green_cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_green_boxes = [c for c in green_cnts if cv2.contourArea(c) > 1000]

                    if len(valid_green_boxes) >= 2:
                        valid_green_boxes = sorted(valid_green_boxes, key=cv2.contourArea, reverse=True)[:2]
                        
                        boxes = [cv2.boundingRect(c) for c in valid_green_boxes]
                        boxes.sort(key=lambda b: b[0]) 

                        sample_box = boxes[0] 
                        ng_box = boxes[1]     

                        m = 4 # Margem para remover a linha verde da foto
                        sx, sy, sw, sh = sample_box
                        nx, ny, nw, nh = ng_box

                        if sw > m*2 and sh > m*2 and nw > m*2 and nh > m*2:
                            crop_sample = frame_bgr[sy+m : sy+sh-m, sx+m : sx+sw-m].copy()
                            crop_ng = frame_bgr[ny+m : ny+nh-m, nx+m : nx+nw-m].copy()
                            
                            # Emite o sinal e DESLIGA A VARREDURA (One-Shot)
                            self.layout_detected.emit(crop_sample, crop_ng)
                            self.log_updated.emit("Monitor IoT: SNAPSHOT CAPTURADO! Parando varredura.")
                            self.running = False
                            break # Sai do loop infinito instantaneamente

                frame_count += 1
                if frame_count % 15 == 0:
                    if has_blue_header and has_red_header:
                        self.log_updated.emit("Monitor IoT: LAYOUT DETECTADO! 📡 Analisando imagens...")
                    else:
                        self.log_updated.emit("Monitor IoT: Aguardando interface da máquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()