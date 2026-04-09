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

    possible_paths = [
        Path(settings.TESSERACT_CMD),
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]

    for p in possible_paths:
        if p.exists():
            pytesseract.pytesseract.tesseract_cmd = str(p)
            HAS_TESSERACT = True
            print(f"✅ Tesseract encontrado: {p}")
            break

    if not HAS_TESSERACT:
        import shutil
        auto_path = shutil.which("tesseract")
        if auto_path:
            pytesseract.pytesseract.tesseract_cmd = auto_path
            HAS_TESSERACT = True
            print(f"✅ Tesseract encontrado no PATH: {auto_path}")
        else:
            print("⚠️ Tesseract NÃO encontrado. Ajuste TESSERACT_CMD em settings.py")

except ImportError:
    print("⚠️ pytesseract não instalado. OCR desativado.")


class ScreenMonitor(QThread):
    log_updated = pyqtSignal(str)
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
        """Remove o fundo cinza da interface da AOI."""
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return image

        b, g, r = cv2.split(image)
        b_f = b.astype(np.float32)
        g_f = g.astype(np.float32)
        r_f = r.astype(np.float32)

        max_channel = np.maximum(np.maximum(b_f, g_f), r_f)
        min_channel = np.minimum(np.minimum(b_f, g_f), r_f)
        channel_diff = max_channel - min_channel
        brightness = (b_f + g_f + r_f) / 3.0

        gray_mask = (
            (channel_diff < settings.AOI_GRAY_THRESHOLD) &
            (brightness >= settings.AOI_GRAY_MIN) &
            (brightness <= settings.AOI_GRAY_MAX)
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray_mask = cv2.dilate(gray_mask, kernel, iterations=2)
        not_gray = cv2.bitwise_not(gray_mask)

        contours, _ = cv2.findContours(not_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest)

        if rw * rh < (w * h * 0.2):
            return image

        margin = 2
        rx = min(rx + margin, w - 1)
        ry = min(ry + margin, h - 1)
        rw = max(rw - margin * 2, 10)
        rh = max(rh - margin * 2, 10)

        return image[ry:ry+rh, rx:rx+rw].copy()

    def _extract_aoi_region(self, frame_bgr, bar_rect, green_boxes):
        """Extrai a foto real da AOI (sem cinza)."""
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        region_x1 = max(0, bx)
        region_x2 = min(frame_w, bx + bw)
        region_y_top = by + bh

        if green_boxes:
            green_bottoms = [gy + gh for (gx, gy, gw, gh) in green_boxes]
            region_y_bottom = max(green_bottoms) + 5
        else:
            region_y_bottom = min(frame_h, region_y_top + int((frame_h - region_y_top) * 0.7))

        region_y_top = max(0, region_y_top)
        region_y_bottom = min(frame_h, region_y_bottom)

        raw_crop = frame_bgr[region_y_top:region_y_bottom, region_x1:region_x2].copy()
        clean_crop = self._remove_gray_background(raw_crop)

        return clean_crop

    def _ocr_fast(self, image: np.ndarray) -> str:
        """
        OCR rápido: usa apenas a melhor estratégia (Otsu + PSM 6).
        Otimizado para fontes Windows XP.
        """
        if not HAS_TESSERACT or image is None or image.size == 0:
            return ""

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Escala para fontes XP pequenas
            scale = max(3, min(5, 200 // max(h, 1)))
            gray_big = cv2.resize(gray, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

            # Otsu (melhor para texto XP em fundo cinza)
            _, binary = cv2.threshold(gray_big, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            raw = pytesseract.image_to_string(binary, config='--psm 6 --oem 3')
            return raw.strip()

        except Exception:
            return ""

    def _extract_text_info(self, frame_bgr, blue_bar, red_bar) -> dict:
        """
        Extrai Board, Parts e Value do texto ACIMA das barras coloridas.
        Estratégia rápida: pega a zona acima da barra azul (que normalmente
        tem a mesma informação) e faz OCR uma única vez.
        """
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}

        if not HAS_TESSERACT:
            info["raw_text"] = "[OCR não disponível]"
            return info

        # Zona de texto: 400px acima da barra até o topo da barra
        # Usa a barra azul como referência principal
        bx, by, bw, bh = blue_bar
        frame_h, frame_w = frame_bgr.shape[:2]

        above_y1 = max(0, by - 400)
        above_y2 = by

        if above_y2 <= above_y1 + 5:
            info["raw_text"] = "[Zona de texto muito pequena]"
            return info

        # Recorta a zona de texto (mesma largura da barra, expandida 50px)
        expand_x = 50
        x1 = max(0, bx - expand_x)
        x2 = min(frame_w, bx + bw + expand_x)
        text_zone = frame_bgr[above_y1:above_y2, x1:x2].copy()

        raw = self._ocr_fast(text_zone)

        if not raw:
            # Fallback: tenta a barra vermelha
            bx2, by2, bw2, bh2 = red_bar
            above_y1_r = max(0, by2 - 400)
            above_y2_r = by2
            x1r = max(0, bx2 - expand_x)
            x2r = min(frame_w, bx2 + bw2 + expand_x)
            if above_y2_r > above_y1_r + 5:
                text_zone_r = frame_bgr[above_y1_r:above_y2_r, x1r:x2r].copy()
                raw = self._ocr_fast(text_zone_r)

        if not raw:
            info["raw_text"] = "[Nenhum texto encontrado]"
            return info

        info["raw_text"] = raw

        # ========================================================
        # Parse dos campos com regex.
        # O texto OCR vem em linhas tipo:
        #   "Board : ABC123"
        #   "Parts : R101  Block : 2"
        #   "Value : 10K"
        #
        # Parts precisa parar ANTES de "Block" ou fim da linha.
        # ========================================================
        for line in raw.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue

            # Board — pega até o fim da linha
            if not info["board"]:
                m = re.search(
                    r'[BbRr8Hh][Oo0][Aa][Rr][Dd]\s*[:\-\.\s]\s*(\S.*)',
                    line_clean
                )
                if m:
                    info["board"] = m.group(1).strip()

            # Parts — pega até "Block", "Value", ou fim da linha
            # Isso evita pegar "Block" como parte do Parts
            if not info["parts"]:
                m = re.search(
                    r'[PpFf][Aa][Rr][Tt][Ss]?\s*[:\-\.\s]\s*(.+?)(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Vv][Aa][Ll]|\s*$)',
                    line_clean
                )
                if m:
                    val = m.group(1).strip()
                    # Limpa caracteres residuais de OCR no final
                    val = re.sub(r'\s+$', '', val)
                    if val:
                        info["parts"] = val

            # Value — pega até "Block", "Parts", ou fim da linha
            if not info["value"]:
                m = re.search(
                    r'[Vv][Aa][Ll1iI][Uu][Ee]\s*[:\-\.\s]\s*(.+?)(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Pp][Aa][Rr]|\s*$)',
                    line_clean
                )
                if m:
                    val = m.group(1).strip()
                    val = re.sub(r'\s+$', '', val)
                    if val:
                        info["value"] = val

        print(f"📋 OCR — Board: '{info['board']}' | "
              f"Parts: '{info['parts']}' | Value: '{info['value']}'")

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

                mask_blue = cv2.inRange(hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
                mask_red1 = cv2.inRange(hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
                mask_red2 = cv2.inRange(hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                mask_green = cv2.inRange(hsv, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)

                blue_bar = self._find_color_bar(mask_blue)
                red_bar = self._find_color_bar(mask_red)

                has_blue = blue_bar is not None
                has_red = red_bar is not None

                if has_blue and has_red:
                    green_cnts, _ = cv2.findContours(
                        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_boxes = [cv2.boundingRect(c) for c in green_cnts
                                   if cv2.contourArea(c) > 500]

                    if len(green_boxes) >= 2:
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
                            crop_sample = self._extract_aoi_region(
                                frame_bgr, blue_bar, sample_greens)
                            crop_ng = self._extract_aoi_region(
                                frame_bgr, red_bar, ng_greens)

                            if crop_sample.size > 0 and crop_ng.size > 0:
                                # OCR rápido
                                aoi_info = self._extract_text_info(
                                    frame_bgr, blue_bar, red_bar)

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