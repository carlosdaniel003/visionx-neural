"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot: Tira um único print ao encontrar o layout e encerra a thread.

Captura a imagem COMPLETA entregue pela AOI (entre a barra colorida e o painel de info),
e extrai as informações de texto (Board, Parts, Value) via OCR.
"""
import cv2
import numpy as np
import mss
import re
from PyQt6.QtCore import QThread, pyqtSignal
from src.config.settings import settings

# Tenta importar pytesseract (opcional — funciona sem ele)
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class ScreenMonitor(QThread):
    log_updated = pyqtSignal(str)
    # Sinal: (crop_sample, crop_ng, aoi_info_dict)
    layout_detected = pyqtSignal(np.ndarray, np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def _find_color_bar(self, mask, min_area=2000):
        """Encontra a maior barra colorida e retorna seu bounding rect."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        largest = max(valid, key=cv2.contourArea)
        return cv2.boundingRect(largest)

    def _extract_aoi_region(self, frame_bgr, bar_rect, green_boxes):
        """
        Extrai a região da imagem que a AOI realmente entrega.
        
        Layout da AOI (de cima para baixo):
        ┌─────────────────────────────────┐
        │  Barra colorida (Azul/Vermelha) │  ← bar_rect
        ├─────────────────────────────────┤
        │  Zona de texto (Board/Parts/Val)│  ← entre barra e início da imagem
        ├─────────────────────────────────┤
        │                                 │
        │  IMAGEM REAL DA AOI             │  ← o que queremos capturar
        │  (contém quadrado verde menor)  │
        │                                 │
        ├─────────────────────────────────┤
        │  Painel de info (NÃO capturar)  │  ← abaixo da imagem
        └─────────────────────────────────┘
        
        A imagem real fica entre a barra colorida e o painel de info.
        Os quadrados verdes nos ajudam a delimitar a área da imagem.
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        # A coluna X da região é definida pela barra colorida
        region_x1 = bx
        region_x2 = bx + bw

        # O topo da região de imagem começa logo abaixo da barra colorida
        region_y_top = by + bh

        # Usa os quadrados verdes para encontrar o limite inferior da imagem
        # (o retângulo verde maior indica até onde a imagem vai)
        if green_boxes:
            # Pega o Y mais baixo entre todos os quadrados verdes
            green_bottoms = [gy + gh for (gx, gy, gw, gh) in green_boxes]
            green_tops = [gy for (gx, gy, gw, gh) in green_boxes]
            
            # A imagem vai do topo da barra até um pouco além do verde mais baixo
            region_y_bottom = max(green_bottoms) + 5  # pequena margem
        else:
            # Fallback: pega 70% da altura abaixo da barra
            region_y_bottom = min(frame_h, region_y_top + int((frame_h - region_y_top) * 0.7))

        # Garante limites válidos
        region_y_top = max(0, region_y_top)
        region_y_bottom = min(frame_h, region_y_bottom)
        region_x1 = max(0, region_x1)
        region_x2 = min(frame_w, region_x2)

        # Margem interna para não pegar a borda da interface
        margin = 2
        crop = frame_bgr[
            region_y_top + margin : region_y_bottom - margin,
            region_x1 + margin : region_x2 - margin
        ].copy()

        # Zona de texto: entre a barra colorida e o início dos quadrados verdes
        text_region = None
        if green_boxes:
            green_top = min(green_tops)
            text_y1 = region_y_top
            text_y2 = green_top
            if text_y2 > text_y1 + 10:  # Tem espaço para texto
                text_region = frame_bgr[text_y1:text_y2, region_x1:region_x2].copy()

        return crop, text_region

    def _extract_text_info(self, text_region: np.ndarray) -> dict:
        """
        Extrai Board, Parts e Value da zona de texto usando OCR.
        Se Tesseract não estiver disponível, tenta uma abordagem visual.
        """
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}

        if text_region is None or text_region.size == 0:
            return info

        if not HAS_TESSERACT:
            # Sem OCR: salva a região de texto como indicador visual
            info["raw_text"] = "[OCR não disponível - instale pytesseract]"
            return info

        try:
            # Pré-processamento para OCR
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            
            # Aumenta o tamanho para melhorar o OCR
            scale = 2
            gray_big = cv2.resize(gray, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
            
            # Binarização adaptativa (funciona bem com fundos variados)
            binary = cv2.adaptiveThreshold(
                gray_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # OCR
            raw = pytesseract.image_to_string(binary, config='--psm 6')
            info["raw_text"] = raw.strip()

            # Extrai campos específicos via regex
            # Padrões flexíveis para "Board:", "Parts:", "Value:"
            for line in raw.split('\n'):
                line_clean = line.strip()
                
                board_match = re.search(r'[Bb]oard\s*[:\-]?\s*(.*)', line_clean)
                if board_match:
                    info["board"] = board_match.group(1).strip()

                parts_match = re.search(r'[Pp]arts?\s*[:\-]?\s*(.*)', line_clean)
                if parts_match:
                    info["parts"] = parts_match.group(1).strip()

                value_match = re.search(r'[Vv]alue\s*[:\-]?\s*(.*)', line_clean)
                if value_match:
                    info["value"] = value_match.group(1).strip()

        except Exception as e:
            info["raw_text"] = f"[Erro OCR: {e}]"

        return info

    def run(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            frame_count = 0

            while self.running:
                screenshot = sct.grab(monitor)
                frame_bgra = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                # 1. Máscaras de cor
                mask_blue = cv2.inRange(hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
                mask_red1 = cv2.inRange(hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
                mask_red2 = cv2.inRange(hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                mask_green = cv2.inRange(hsv, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)

                # 2. Encontra barras azul e vermelha
                blue_bar = self._find_color_bar(mask_blue)
                red_bar = self._find_color_bar(mask_red)

                has_blue = blue_bar is not None
                has_red = red_bar is not None

                if has_blue and has_red:
                    # 3. Encontra TODOS os quadrados verdes
                    green_cnts, _ = cv2.findContours(
                        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_boxes = [cv2.boundingRect(c) for c in green_cnts 
                                   if cv2.contourArea(c) > 500]

                    if len(green_boxes) >= 2:
                        # Separa os quadrados verdes em dois grupos (sample e ng)
                        # baseado na proximidade com a barra azul ou vermelha
                        blue_cx = blue_bar[0] + blue_bar[2] // 2  # Centro X da barra azul
                        red_cx = red_bar[0] + red_bar[2] // 2    # Centro X da barra vermelha

                        sample_greens = []
                        ng_greens = []

                        for gb in green_boxes:
                            gx_center = gb[0] + gb[2] // 2
                            dist_blue = abs(gx_center - blue_cx)
                            dist_red = abs(gx_center - red_cx)

                            if dist_blue < dist_red:
                                sample_greens.append(gb)
                            else:
                                ng_greens.append(gb)

                        if sample_greens and ng_greens:
                            # Extrai a região completa de cada lado
                            crop_sample, text_sample = self._extract_aoi_region(
                                frame_bgr, blue_bar, sample_greens)
                            crop_ng, text_ng = self._extract_aoi_region(
                                frame_bgr, red_bar, ng_greens)

                            if crop_sample.size > 0 and crop_ng.size > 0:
                                # Extrai informações de texto (usa a região do sample)
                                aoi_info = self._extract_text_info(text_sample)

                                # Se não encontrou texto no sample, tenta no NG
                                if not aoi_info["board"] and text_ng is not None:
                                    aoi_info_ng = self._extract_text_info(text_ng)
                                    if aoi_info_ng["board"]:
                                        aoi_info = aoi_info_ng

                                # Emite e para
                                self.layout_detected.emit(crop_sample, crop_ng, aoi_info)
                                self.log_updated.emit(
                                    "Monitor AOI: SNAPSHOT CAPTURADO! Parando varredura.")
                                self.running = False
                                break

                frame_count += 1
                if frame_count % 15 == 0:
                    if has_blue and has_red:
                        self.log_updated.emit(
                            "Monitor AOI: LAYOUT DETECTADO! 📡 Analisando imagens...")
                    else:
                        self.log_updated.emit(
                            "Monitor AOI: Aguardando interface da máquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()