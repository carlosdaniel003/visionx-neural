# src\services\screen_monitor.py
"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.

ESTRATÉGIA "RADAR DE DENSIDADE COM ÂNCORA GEOMÉTRICA":
1. O Topo da foto é SEMPRE a base exata da barra colorida.
2. Escaneia para baixo e para os lados até bater no cinza da interface.
3. REGRA DE OURO DA ALTURA: As duas fotos devem ter a mesma altura. 
   Se uma vazar, a que cortou menor impõe seu limite e apara a outra.
"""
import cv2
import numpy as np
import mss
import re
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from src.config.settings import settings

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

    def _extract_aoi_region(self, frame_bgr, bar_rect, all_greens):
        """
        Faz o recorte cirúrgico a partir da barra até a parede cinza da interface.
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        # 1. Isola a coluna exatamente na mesma largura e posição da barra
        col_x1 = max(0, bx)
        col_x2 = min(frame_w, bx + bw)
        actual_bw = col_x2 - col_x1
        
        # Puxa uma altura limite segura para baixo
        col_y1 = by + bh
        col_y2 = min(frame_h, col_y1 + 650)
        actual_bh = col_y2 - col_y1

        if actual_bw == 0 or actual_bh == 0:
            return np.array([])

        col_img = frame_bgr[col_y1:col_y2, col_x1:col_x2].copy()

        # 2. Criar a máscara do fundo Cinza (#c0c0c0 e #d4d0c7)
        b, g, r = cv2.split(col_img)
        b_f, g_f, r_f = b.astype(int), g.astype(int), r.astype(int)

        max_c = np.maximum(np.maximum(b_f, g_f), r_f)
        min_c = np.minimum(np.minimum(b_f, g_f), r_f)
        diff = max_c - min_c

        # Regra implacável do Cinza
        bg_mask = ((diff < 25) & (max_c > 150) & (max_c < 240)).astype(np.uint8) * 255
        fg_mask = cv2.bitwise_not(bg_mask)

        # 3. Encontrar os limites das Caixas Verdes DESTA coluna
        g_bottom = 0
        g_center_x = actual_bw // 2
        valid_greens = 0

        for gx, gy, gw, gh in all_greens:
            cx_global = gx + gw // 2 
            
            if col_x1 <= cx_global <= col_x2:
                rel_y1 = gy - col_y1
                rel_y2 = rel_y1 + gh
                
                if rel_y2 > 0 and rel_y1 < actual_bh:
                    g_bottom = max(g_bottom, min(actual_bh, rel_y2))
                    valid_greens += 1
                    g_center_x = cx_global - col_x1

        if valid_greens == 0:
            g_bottom = actual_bh // 2

        # ==========================================
        # RADAR VERTICAL
        # ==========================================
        # O topo da foto SEMPRE começa encostado na barra colorida.
        top_y = 0 

        row_density = np.sum(fg_mask > 0, axis=1) / actual_bw

        bottom_y = g_bottom
        while bottom_y < actual_bh - 1:
            if row_density[bottom_y] < 0.05:
                break
            bottom_y += 1

        foto_h = bottom_y - top_y
        if foto_h <= 0:
            return col_img 

        # ==========================================
        # RADAR HORIZONTAL
        # ==========================================
        col_density = np.sum(fg_mask[top_y:bottom_y, :] > 0, axis=0) / foto_h

        left_x = g_center_x
        while left_x > 0:
            if col_density[left_x] < 0.05:
                break
            left_x -= 1

        right_x = g_center_x
        while right_x < actual_bw - 1:
            if col_density[right_x] < 0.05:
                break
            right_x += 1

        # 6. Recorte Definitivo
        crop_y1 = top_y
        crop_y2 = min(actual_bh, bottom_y)
        crop_x1 = max(0, left_x + 1)
        crop_x2 = min(actual_bw, right_x - 1)

        return col_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    def _ocr_fast(self, image: np.ndarray) -> str:
        if not HAS_TESSERACT or image is None or image.size == 0:
            return ""

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            scale = max(3, min(5, 200 // max(h, 1)))
            gray_big = cv2.resize(gray, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

            _, binary = cv2.threshold(gray_big, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            raw = pytesseract.image_to_string(binary, config='--psm 6 --oem 3')
            return raw.strip()

        except Exception:
            return ""

    def _extract_text_info(self, frame_bgr, blue_bar, red_bar) -> dict:
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}
        if not HAS_TESSERACT:
            info["raw_text"] = "[OCR não disponível]"
            return info

        bx, by, bw, bh = blue_bar
        frame_h, frame_w = frame_bgr.shape[:2]

        above_y1 = max(0, by - 400)
        above_y2 = by

        if above_y2 <= above_y1 + 5:
            return info

        expand_x = 50
        x1 = max(0, bx - expand_x)
        x2 = min(frame_w, bx + bw + expand_x)
        text_zone = frame_bgr[above_y1:above_y2, x1:x2].copy()

        raw = self._ocr_fast(text_zone)

        if not raw:
            bx2, by2, bw2, bh2 = red_bar
            above_y1_r = max(0, by2 - 400)
            above_y2_r = by2
            x1r = max(0, bx2 - expand_x)
            x2r = min(frame_w, bx2 + bw2 + expand_x)
            if above_y2_r > above_y1_r + 5:
                text_zone_r = frame_bgr[above_y1_r:above_y2_r, x1r:x2r].copy()
                raw = self._ocr_fast(text_zone_r)

        info["raw_text"] = raw

        for line in raw.split('\n'):
            line_clean = line.strip()
            if not line_clean:
                continue

            if not info["board"]:
                m = re.search(r'[BbRr8Hh][Oo0][Aa][Rr][Dd]\s*[:\-\.\s]\s*(\S.*)', line_clean)
                if m: info["board"] = m.group(1).strip()

            if not info["parts"]:
                m = re.search(r'[PpFf][Aa][Rr][Tt][Ss]?\s*[:\-\.\s]\s*(.+?)(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Vv][Aa][Ll]|\s*$)', line_clean)
                if m: info["parts"] = re.sub(r'\s+$', '', m.group(1).strip())

            if not info["value"]:
                m = re.search(r'[Vv][Aa][Ll1iI][Uu][Ee]\s*[:\-\.\s]\s*(.+?)(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Pp][Aa][Rr]|\s*$)', line_clean)
                if m: info["value"] = re.sub(r'\s+$', '', m.group(1).strip())

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
                    all_greens = [cv2.boundingRect(c) for c in green_cnts if cv2.contourArea(c) > 500]

                    if len(all_greens) >= 1:
                        
                        crop_sample = self._extract_aoi_region(frame_bgr, blue_bar, all_greens)
                        crop_ng = self._extract_aoi_region(frame_bgr, red_bar, all_greens)

                        if crop_sample.size > 0 and crop_ng.size > 0:
                            
                            # =========================================================
                            # SINCRONIZAÇÃO DE ALTURA (A REGRA DE OURO)
                            # Pega a menor altura entre as duas imagens e apara o fundo da maior
                            # =========================================================
                            h_sample = crop_sample.shape[0]
                            h_ng = crop_ng.shape[0]
                            min_h = min(h_sample, h_ng)

                            crop_sample = crop_sample[0:min_h, :]
                            crop_ng = crop_ng[0:min_h, :]

                            # Sincronização de Largura (Mantendo as matrizes simétricas)
                            sh, sw = crop_sample.shape[:2]
                            if crop_ng.shape[:2] != (sh, sw):
                                crop_ng = cv2.resize(crop_ng, (sw, sh))

                            aoi_info = self._extract_text_info(frame_bgr, blue_bar, red_bar)

                            self.layout_detected.emit(crop_sample, crop_ng, aoi_info)
                            self.log_updated.emit("Monitor AOI: SNAPSHOT CAPTURADO! 📸")
                            self.running = False
                            break

                frame_count += 1
                if frame_count % 15 == 0:
                    if has_blue and has_red:
                        self.log_updated.emit("Monitor AOI: LAYOUT DETECTADO! 📡 Analisando...")
                    else:
                        self.log_updated.emit("Monitor AOI: Aguardando interface da máquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()