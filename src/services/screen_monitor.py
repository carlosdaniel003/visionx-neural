# src\services\screen_monitor.py
"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot.

ESTRATÉGIA v14 "CINZA SUSTENTADO PURO":
═══════════════════════════════════════
Removida a detecção de botões — causava falsos positivos
com os retângulos verdes grandes da AOI.

Usa APENAS:
1. Máscara restrita de cinza (spread ≤ 15, bright 178-218, sat ≤ 25)
2. "Cinza sustentado" para BOTTOM: para onde ≥90% do restante é cinza
3. Colunas com gap_limit=6
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
            print("⚠️ Tesseract NÃO encontrado.")
except ImportError:
    print("⚠️ pytesseract não instalado. OCR desativado.")


DEBUG_DIR = Path("public/debug_crop")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


class ScreenMonitor(QThread):
    log_updated = pyqtSignal(str)
    layout_detected = pyqtSignal(np.ndarray, np.ndarray, dict)

    def __init__(self):
        super().__init__()
        self.running = True

    # =================================================================
    # DETECÇÃO DE BARRAS
    # =================================================================

    def _find_color_bar(self, mask, min_area=2000):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        return cv2.boundingRect(max(valid, key=cv2.contourArea))

    def _find_sibling_bar(self, mask, reference_bar, min_area=1000):
        ref_x, ref_y, ref_w, ref_h = reference_bar
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < h * 2:
                continue
            if abs(y - ref_y) > 30:
                continue
            if ref_h > 0 and (h > ref_h * 2 or h < ref_h * 0.3):
                continue
            if ref_w > 0 and w < ref_w * 0.3:
                continue
            if abs(x - ref_x) < ref_w * 0.5:
                continue
            candidates.append((cv2.contourArea(c), (x, y, w, h)))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # =================================================================
    # MÁSCARA DE CINZA DE INTERFACE
    # =================================================================

    def _build_interface_gray_mask(self, strip_bgr):
        """
        Máscara booleana (h, w): True = pixel é cinza de INTERFACE.
        spread ≤ 15, brightness 178-218, saturation ≤ 25
        """
        b = strip_bgr[:, :, 0].astype(np.int16)
        g = strip_bgr[:, :, 1].astype(np.int16)
        r = strip_bgr[:, :, 2].astype(np.int16)

        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        spread = max_c - min_c
        brightness = (b + g + r) // 3

        hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.int16)

        return (
            (spread <= 15) &
            (brightness >= 178) &
            (brightness <= 218) &
            (saturation <= 25)
        )

    # =================================================================
    # ENCONTRAR BORDA DA FOTO — "CINZA SUSTENTADO"
    # =================================================================

    def _find_photo_bounds(self, is_gray_arr, min_remaining=15,
                            sustain_pct=0.90):
        """
        TOP: primeira linha NÃO-cinza
        BOTTOM: primeira linha cinza onde ≥ sustain_pct do restante é cinza
                E restam ≥ min_remaining linhas.
        Se não encontrar borda sustentada, usa última linha não-cinza.
        """
        n = len(is_gray_arr)

        # TOP
        top = None
        for i in range(n):
            if not is_gray_arr[i]:
                top = i
                break

        if top is None:
            return None, None

        # Soma acumulada reversa
        gray_count = np.asarray(is_gray_arr, dtype=np.int32)
        reverse_cumsum = np.cumsum(gray_count[::-1])[::-1]
        remaining = np.arange(n, 0, -1)
        gray_pct_to_end = reverse_cumsum / remaining

        # BOTTOM
        bottom = n

        for i in range(top + 1, n):
            if is_gray_arr[i]:
                lines_left = n - i
                if (lines_left >= min_remaining and
                        gray_pct_to_end[i] >= sustain_pct):
                    bottom = i
                    break

        # Se não encontrou borda sustentada, última não-cinza
        if bottom == n:
            for i in range(n - 1, top, -1):
                if not is_gray_arr[i]:
                    bottom = i + 1
                    break

        if (bottom - top) < 5:
            return None, None

        return top, bottom

    # =================================================================
    # COLUNAS — GAP SIMPLES
    # =================================================================

    def _find_col_span(self, is_gray_arr, gap_limit=6):
        n = len(is_gray_arr)

        left = None
        for i in range(n):
            if not is_gray_arr[i]:
                left = i
                break

        if left is None:
            return None, None

        right = left + 1
        consecutive_gray = 0

        for i in range(left + 1, n):
            if not is_gray_arr[i]:
                right = i + 1
                consecutive_gray = 0
            else:
                consecutive_gray += 1
                if consecutive_gray >= gap_limit:
                    break

        if (right - left) < 5:
            return None, None

        return left, right

    # =================================================================
    # MOTOR DE RECORTE v14
    # =================================================================

    def _extract_photo(self, frame_bgr, bar_rect, max_height, label=""):
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        x1 = max(0, bx)
        x2 = min(frame_w, bx + bw)
        y1 = by + bh
        y2 = min(frame_h, y1 + max_height)

        if y2 - y1 < 20 or x2 - x1 < 20:
            return np.array([])

        strip = frame_bgr[y1:y2, x1:x2].copy()
        strip_h, strip_w = strip.shape[:2]

        cv2.imwrite(str(DEBUG_DIR / f"{label}_01_strip.png"), strip)

        # Máscara de cinza
        gray_mask = self._build_interface_gray_mask(strip)

        # =============================================
        # DEBUG
        # =============================================
        total_gray_pct = np.mean(gray_mask)
        print(f"   [{label}] Máscara cinza: {total_gray_pct:.1%} "
              f"(strip {strip_w}x{strip_h})")

        # =============================================
        # LINHAS — cinza sustentado
        # =============================================
        row_gray_pct = np.mean(gray_mask, axis=1)
        is_gray_row = row_gray_pct >= 0.75

        gray_row_count = np.sum(is_gray_row)
        photo_row_count = strip_h - gray_row_count
        print(f"   [{label}] Linhas: {photo_row_count} foto, "
              f"{gray_row_count} cinza (de {strip_h} total)")

        top, bottom = self._find_photo_bounds(is_gray_row)

        if top is None:
            if gray_row_count == 0:
                # Nenhum cinza → foto preenche tudo
                top = 0
                bottom = strip_h
                print(f"   [{label}] Nenhum cinza → foto = strip inteiro")
            else:
                print(f"⚠️ [{label}] Sem foto detectada — fallback")
                cv2.imwrite(str(DEBUG_DIR / f"{label}_FALLBACK.png"), strip)

                gm_rows = np.zeros((strip_h, 40, 3), dtype=np.uint8)
                for row in range(strip_h):
                    bar_w = int(row_gray_pct[row] * 35)
                    color = (0, 0, 200) if is_gray_row[row] else (0, 255, 0)
                    cv2.line(gm_rows, (0, row), (max(1, bar_w), row),
                              color, 1)
                thresh_x = int(0.75 * 35)
                cv2.line(gm_rows, (thresh_x, 0), (thresh_x, strip_h),
                          (0, 255, 255), 1)
                cv2.imwrite(str(DEBUG_DIR / f"{label}_graymap_rows.png"),
                              gm_rows)
                return strip

        top = max(0, top)
        bottom = min(strip_h, bottom)
        print(f"   [{label}] Resultado: top={top} bottom={bottom} "
              f"(h={bottom-top})")

        # =============================================
        # COLUNAS (gap=6)
        # =============================================
        band_mask = gray_mask[top:bottom, :]
        col_gray_pct = np.mean(band_mask, axis=0)
        is_gray_col = col_gray_pct >= 0.75

        left, right = self._find_col_span(is_gray_col, gap_limit=6)

        if left is None:
            left = 0
            right = strip_w
            print(f"   [{label}] Colunas: sem bordas → largura total")
        else:
            left = max(0, left)
            right = min(strip_w, right)
            print(f"   [{label}] Colunas: left={left} right={right} "
                  f"(w={right-left})")

        photo = strip[top:bottom, left:right].copy()

        print(f"📐 [{label}] Strip {strip_w}x{strip_h} → "
              f"Foto {right-left}x{bottom-top}")

        # === Debug ===
        debug_vis = strip.copy()
        cv2.rectangle(debug_vis, (left, top), (right, bottom),
                       (0, 255, 0), 2)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_03_detected.png"), debug_vis)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_04_photo.png"), photo)

        # Graymap linhas
        gm_rows = np.zeros((strip_h, 40, 3), dtype=np.uint8)
        for row in range(strip_h):
            bar_w = int(row_gray_pct[row] * 35)
            color = (0, 0, 200) if is_gray_row[row] else (0, 255, 0)
            cv2.line(gm_rows, (0, row), (max(1, bar_w), row), color, 1)
        cv2.line(gm_rows, (0, top), (40, top), (255, 0, 255), 2)
        cv2.line(gm_rows, (0, min(bottom, strip_h - 1)),
                  (40, min(bottom, strip_h - 1)), (255, 0, 255), 2)
        thresh_x = int(0.75 * 35)
        cv2.line(gm_rows, (thresh_x, 0), (thresh_x, strip_h),
                  (0, 255, 255), 1)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_graymap_rows.png"), gm_rows)

        # Graymap colunas
        gm_cols = np.zeros((40, strip_w, 3), dtype=np.uint8)
        for col in range(strip_w):
            bar_h = int(col_gray_pct[col] * 35)
            color = (0, 0, 200) if is_gray_col[col] else (0, 255, 0)
            cv2.line(gm_cols, (col, 40), (col, 40 - max(1, bar_h)),
                      color, 1)
        if left > 0:
            cv2.line(gm_cols, (left, 0), (left, 40), (255, 0, 255), 2)
        if right < strip_w:
            cv2.line(gm_cols, (min(right, strip_w - 1), 0),
                      (min(right, strip_w - 1), 40), (255, 0, 255), 2)
        thresh_y = 40 - int(0.75 * 35)
        cv2.line(gm_cols, (0, thresh_y), (strip_w, thresh_y),
                  (0, 255, 255), 1)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_graymap_cols.png"), gm_cols)

        return photo

    # =================================================================
    # OCR
    # =================================================================

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
            return pytesseract.image_to_string(
                binary, config='--psm 6 --oem 3').strip()
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
        x1t = max(0, bx - expand_x)
        x2t = min(frame_w, bx + bw + expand_x)
        text_zone = frame_bgr[above_y1:above_y2, x1t:x2t].copy()
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

        if not raw:
            info["raw_text"] = "[Nenhum texto encontrado]"
            return info

        info["raw_text"] = raw
        for line in raw.split('\n'):
            lc = line.strip()
            if not lc:
                continue
            if not info["board"]:
                m = re.search(
                    r'[BbRr8Hh][Oo0][Aa][Rr][Dd]\s*[:\-\.\s]\s*(\S.*)', lc)
                if m:
                    info["board"] = m.group(1).strip()
            if not info["parts"]:
                m = re.search(
                    r'[PpFf][Aa][Rr][Tt][Ss]?\s*[:\-\.\s]\s*(.+?)'
                    r'(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Vv][Aa][Ll]|\s*$)', lc)
                if m and m.group(1).strip():
                    info["parts"] = m.group(1).strip()
            if not info["value"]:
                m = re.search(
                    r'[Vv][Aa][Ll1iI][Uu][Ee]\s*[:\-\.\s]\s*(.+?)'
                    r'(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Pp][Aa][Rr]|\s*$)', lc)
                if m and m.group(1).strip():
                    info["value"] = m.group(1).strip()

        print(f"📋 OCR — Board: '{info['board']}' | "
              f"Parts: '{info['parts']}' | Value: '{info['value']}'")
        return info

    # =================================================================
    # LOOP PRINCIPAL
    # =================================================================

    def run(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            frame_count = 0

            while self.running:
                screenshot = sct.grab(monitor)
                frame_bgra = np.array(screenshot)
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                frame_h, frame_w = frame_bgr.shape[:2]

                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                mask_blue = cv2.inRange(
                    hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
                mask_red1 = cv2.inRange(
                    hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
                mask_red2 = cv2.inRange(
                    hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                blue_bar = self._find_color_bar(mask_blue)
                if blue_bar is None:
                    frame_count += 1
                    if frame_count % 15 == 0:
                        self.log_updated.emit(
                            "Monitor AOI: Aguardando interface da máquina...")
                    self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))
                    continue

                red_bar = self._find_sibling_bar(mask_red, blue_bar)
                if red_bar is None:
                    red_bar = self._find_color_bar(mask_red)

                if red_bar is None:
                    frame_count += 1
                    if frame_count % 15 == 0:
                        self.log_updated.emit(
                            "Monitor AOI: Barra azul OK, vermelha não encontrada...")
                    self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))
                    continue

                bar_bottom = max(blue_bar[1] + blue_bar[3],
                                 red_bar[1] + red_bar[3])
                max_photo_height = min(600, frame_h - bar_bottom - 50)

                print(f"🔍 Barras: Azul({blue_bar[0]},{blue_bar[1]} "
                      f"w={blue_bar[2]} h={blue_bar[3]}) | "
                      f"Verm({red_bar[0]},{red_bar[1]} "
                      f"w={red_bar[2]} h={red_bar[3]})")

                crop_sample = self._extract_photo(
                    frame_bgr, blue_bar, max_photo_height, "AZUL")
                crop_ng = self._extract_photo(
                    frame_bgr, red_bar, max_photo_height, "VERM")

                if crop_sample.size > 0 and crop_ng.size > 0:
                    aoi_info = self._extract_text_info(
                        frame_bgr, blue_bar, red_bar)

                    self.layout_detected.emit(
                        crop_sample, crop_ng, aoi_info)
                    self.log_updated.emit(
                        f"Monitor AOI: CAPTURADO! "
                        f"Azul {crop_sample.shape[1]}x{crop_sample.shape[0]} | "
                        f"Verm {crop_ng.shape[1]}x{crop_ng.shape[0]}")
                    self.running = False
                    break

                frame_count += 1
                if frame_count % 15 == 0:
                    self.log_updated.emit(
                        "Monitor AOI: LAYOUT DETECTADO! 📡 Analisando...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()