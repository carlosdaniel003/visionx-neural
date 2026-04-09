# src\services\screen_monitor.py
"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot: Tira um único print ao encontrar o layout e encerra a thread.

Captura a imagem COMPLETA entregue pela AOI, REMOVE o fundo cinza da interface,
e extrai as informações de texto (Board, Parts, Value) via OCR.
"""
import cv2
import numpy as np
import mss
import re
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from src.config.settings import settings

# Configura e importa pytesseract (opcional)
HAS_TESSERACT = False
try:
    import pytesseract

    # Configura o caminho do executável
    tesseract_path = Path(settings.TESSERACT_CMD)
    if tesseract_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
        HAS_TESSERACT = True
        print(f"✅ Tesseract encontrado: {tesseract_path}")
    else:
        # Tenta encontrar automaticamente no PATH do sistema
        import shutil
        auto_path = shutil.which("tesseract")
        if auto_path:
            pytesseract.pytesseract.tesseract_cmd = auto_path
            HAS_TESSERACT = True
            print(f"✅ Tesseract encontrado no PATH: {auto_path}")
        else:
            print(f"⚠️ Tesseract NÃO encontrado em: {tesseract_path}")
            print("   OCR desativado. Ajuste TESSERACT_CMD em settings.py")
except ImportError:
    print("⚠️ pytesseract não instalado. OCR desativado.")


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

    def _remove_gray_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove o fundo cinza da interface da AOI, mantendo apenas a foto real da PCB.
        
        A AOI coloca a foto dentro de um painel cinza. A foto pode ser menor que
        o painel, sobrando bordas cinzas. Aqui detectamos onde o cinza termina e
        a imagem real começa.
        
        Estratégia:
        1. Cria uma máscara de "pixels cinza" (R≈G≈B, brilho médio)
        2. Inverte para pegar os "pixels NÃO cinza" (a foto real)
        3. Encontra o bounding rect da área não-cinza
        4. Recorta
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return image

        # Identifica pixels cinza da interface:
        # - Os 3 canais (B,G,R) são próximos entre si (baixa saturação)
        # - Brilho entre AOI_GRAY_MIN e AOI_GRAY_MAX
        b, g, r = cv2.split(image)
        b_f = b.astype(np.float32)
        g_f = g.astype(np.float32)
        r_f = r.astype(np.float32)

        # Diferença máxima entre os canais (cinza = todos próximos)
        max_channel = np.maximum(np.maximum(b_f, g_f), r_f)
        min_channel = np.minimum(np.minimum(b_f, g_f), r_f)
        channel_diff = max_channel - min_channel

        # Brilho médio
        brightness = (b_f + g_f + r_f) / 3.0

        # Máscara de cinza: diferença baixa entre canais + brilho no range do cinza da AOI
        gray_mask = (
            (channel_diff < settings.AOI_GRAY_THRESHOLD) &
            (brightness >= settings.AOI_GRAY_MIN) &
            (brightness <= settings.AOI_GRAY_MAX)
        ).astype(np.uint8) * 255

        # Dilata a máscara cinza para conectar regiões próximas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray_mask = cv2.dilate(gray_mask, kernel, iterations=2)

        # Inverte: pixels NÃO cinza = a foto real
        not_gray = cv2.bitwise_not(gray_mask)

        # Encontra os contornos da área não-cinza
        contours, _ = cv2.findContours(not_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image  # Nada encontrado, retorna original

        # Pega a maior região não-cinza (a foto da PCB)
        largest = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest)

        # Proteção: a região precisa ser significativa (pelo menos 20% da imagem)
        if rw * rh < (w * h * 0.2):
            return image

        # Margem de segurança de 2px para dentro
        margin = 2
        rx = min(rx + margin, w - 1)
        ry = min(ry + margin, h - 1)
        rw = max(rw - margin * 2, 10)
        rh = max(rh - margin * 2, 10)

        cropped = image[ry:ry+rh, rx:rx+rw].copy()

        return cropped

    def _extract_aoi_region(self, frame_bgr, bar_rect, green_boxes, hsv_frame):
        """
        Extrai a região da imagem que a AOI realmente entrega.

        Layout da AOI (de cima para baixo):
        ┌─────────────────────────────────┐
        │  Barra colorida (Azul/Vermelha) │  ← bar_rect (topo)
        ├─────────────────────────────────┤
        │  Texto (Board/Parts/Value)      │  ← entre barra e imagem
        ├─────────────────────────────────┤
        │  ┌───────────────────────┐      │
        │  │ FOTO REAL DA PCB      │      │  ← cercada por cinza
        │  │ (quadrado verde menor)│      │
        │  └───────────────────────┘      │
        │         fundo cinza             │
        └─────────────────────────────────┘

        Passos:
        1. Recorta a região entre a barra colorida e o fundo dos quadrados verdes
        2. Remove o fundo cinza da interface
        3. Extrai o texto da zona acima da foto
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        # Limites X definidos pela barra colorida
        region_x1 = max(0, bx)
        region_x2 = min(frame_w, bx + bw)

        # Topo: logo abaixo da barra colorida
        region_y_top = by + bh

        # Fundo: usa o retângulo verde maior para delimitar
        if green_boxes:
            green_bottoms = [gy + gh for (gx, gy, gw, gh) in green_boxes]
            green_tops = [gy for (gx, gy, gw, gh) in green_boxes]
            region_y_bottom = max(green_bottoms) + 5
        else:
            region_y_bottom = min(frame_h, region_y_top + int((frame_h - region_y_top) * 0.7))

        region_y_top = max(0, region_y_top)
        region_y_bottom = min(frame_h, region_y_bottom)

        # Zona de texto: entre a barra e o topo dos quadrados verdes
        text_region = None
        if green_boxes:
            green_top = min(green_tops)
            text_y1 = region_y_top
            text_y2 = green_top
            if text_y2 > text_y1 + 10:
                text_region = frame_bgr[text_y1:text_y2, region_x1:region_x2].copy()

        # Recorta a região bruta (contém cinza + foto)
        raw_crop = frame_bgr[region_y_top:region_y_bottom, region_x1:region_x2].copy()

        # Remove o fundo cinza, ficando só com a foto real da PCB
        clean_crop = self._remove_gray_background(raw_crop)

        return clean_crop, text_region

    def _extract_text_info(self, text_region: np.ndarray) -> dict:
        """
        Extrai Board, Parts e Value da zona de texto usando OCR.
        """
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}

        if text_region is None or text_region.size == 0:
            return info

        if not HAS_TESSERACT:
            info["raw_text"] = "[OCR não disponível]"
            return info

        try:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

            # Aumenta 2x para melhorar OCR
            scale = 2
            gray_big = cv2.resize(gray, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

            # Binarização adaptativa
            binary = cv2.adaptiveThreshold(
                gray_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # OCR
            raw = pytesseract.image_to_string(binary, config='--psm 6')
            info["raw_text"] = raw.strip()

            # Extrai campos via regex
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
                    # 3. Encontra quadrados verdes
                    green_cnts, _ = cv2.findContours(
                        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_boxes = [cv2.boundingRect(c) for c in green_cnts
                                   if cv2.contourArea(c) > 500]

                    if len(green_boxes) >= 2:
                        # Separa verdes por proximidade com barra azul ou vermelha
                        blue_cx = blue_bar[0] + blue_bar[2] // 2
                        red_cx = red_bar[0] + red_bar[2] // 2

                        sample_greens = []
                        ng_greens = []

                        for gb in green_boxes:
                            gx_center = gb[0] + gb[2] // 2
                            if abs(gx_center - blue_cx) < abs(gx_center - red_cx):
                                sample_greens.append(gb)
                            else:
                                ng_greens.append(gb)

                        if sample_greens and ng_greens:
                            # Extrai região completa + remove cinza
                            crop_sample, text_sample = self._extract_aoi_region(
                                frame_bgr, blue_bar, sample_greens, hsv)
                            crop_ng, text_ng = self._extract_aoi_region(
                                frame_bgr, red_bar, ng_greens, hsv)

                            if crop_sample.size > 0 and crop_ng.size > 0:
                                # OCR
                                aoi_info = self._extract_text_info(text_sample)
                                if not aoi_info["board"] and text_ng is not None:
                                    aoi_info_ng = self._extract_text_info(text_ng)
                                    if aoi_info_ng["board"]:
                                        aoi_info = aoi_info_ng

                                self.layout_detected.emit(
                                    crop_sample, crop_ng, aoi_info)
                                self.log_updated.emit(
                                    "Monitor AOI: SNAPSHOT CAPTURADO!")
                                self.running = False
                                break

                frame_count += 1
                if frame_count % 15 == 0:
                    if has_blue and has_red:
                        self.log_updated.emit(
                            "Monitor AOI: LAYOUT DETECTADO! 📡 Analisando...")
                    else:
                        self.log_updated.emit(
                            "Monitor AOI: Aguardando interface da máquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()