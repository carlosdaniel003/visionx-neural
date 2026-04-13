# src\services\screen_monitor.py
"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot.

ESTRATÉGIA v11.1:
Igual a v11 mas com duas melhorias:
1. gray_gap reduzido de 12 → 6 (gap entre foto e botões é fino)
2. Detecta botões coloridos (verde "0.OK", vermelho "1.NG") no strip
   e usa como limite HARD — nunca vai além deles
3. Também detecta a interface XP (#ece9d8 / #d4d0c8) como cinza
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
    # MÁSCARA DE "NÃO-FOTO" (cinza + interface)
    # =================================================================

    def _build_not_photo_mask(self, strip_bgr):
        """
        Máscara booleana (h, w): True = pixel NÃO é foto.

        Detecta:
        - Cinza puro #c0c0c0 (192,192,192): fundo onde a foto fica
        - Cinza interface XP #d4d0c8 (212,208,200): bordas da janela
        - Cinza/bege #ece9d8 (236,233,216): fundo de botões XP
        - Branco/quase-branco (#f0f0f0+): fundo de campos de input

        Critério: spread ≤ 25 (R≈G≈B) E brilho ≥ 170
        Isso pega TODOS os tons de cinza/bege da interface
        mas NÃO pega fotos de PCB (que têm cores/contraste).
        """
        b = strip_bgr[:, :, 0].astype(np.int16)
        g = strip_bgr[:, :, 1].astype(np.int16)
        r = strip_bgr[:, :, 2].astype(np.int16)

        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        spread = max_c - min_c
        brightness = (b + g + r) // 3

        return (spread <= 25) & (brightness >= 170) & (brightness <= 250)

    # =================================================================
    # LIMITE HARD: DETECTA ONDE COMEÇA A INTERFACE ABAIXO DA FOTO
    # =================================================================

    def _find_interface_limit(self, strip_bgr, gray_mask):
        """
        Varre o strip de BAIXO PRA CIMA procurando onde a INTERFACE
        começa (botões, texto, campos).

        A interface abaixo da foto tem:
        - Botões coloridos (verde "0. OK", vermelho "1. NG")
        - Texto preto sobre fundo cinza ("Answer", "Quit Inspection")
        - Campos de input brancos

        Retorna a linha Y onde a interface começa (limite hard).
        Se não encontrar, retorna strip_h (sem limite).
        """
        strip_h, strip_w = strip_bgr.shape[:2]

        # Converte pra HSV para detectar botões verde/vermelho
        hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)

        # Máscara de verde (botão "0. OK")
        green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
        # Máscara de vermelho (botão "1. NG")
        red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combina: qualquer botão colorido
        button_mask = cv2.bitwise_or(green_mask, red_mask)

        # Conta pixels de botão por linha
        button_per_row = np.sum(button_mask > 0, axis=1)

        # Se uma linha tem >= 20 pixels de botão, é interface
        min_button_pixels = 20

        # Varre de BAIXO PRA CIMA procurando a primeira linha de botão
        interface_start = strip_h
        for row in range(strip_h - 1, -1, -1):
            if button_per_row[row] >= min_button_pixels:
                interface_start = row
            elif row < interface_start - 5:
                # Já passou do bloco de botões pra cima
                break

        # Agora varre de interface_start PRA CIMA procurando onde
        # começa a zona de cinza/interface ACIMA dos botões
        # (pode ter texto "Answer", margem, etc)
        not_photo_pct = np.mean(gray_mask, axis=1)
        for row in range(interface_start - 1, -1, -1):
            if not_photo_pct[row] < 0.70:
                # Esta linha tem foto/conteúdo, para aqui
                interface_start = row + 1
                break

        return interface_start

    # =================================================================
    # ENCONTRAR BORDAS — DIREÇÃO ÚNICA
    # =================================================================

    def _find_photo_span(self, is_not_photo_arr, gap_limit=6):
        """
        Encontra o span da foto.

        is_not_photo_arr: True = NÃO é foto (cinza/interface)

        Varre de CIMA PRA BAIXO:
        1. Primeiro False (= foto) → TOP
        2. A partir de TOP, conta consecutivos True (não-foto).
           Quando >= gap_limit → foto acabou.
        """
        n = len(is_not_photo_arr)

        top = None
        for i in range(n):
            if not is_not_photo_arr[i]:
                top = i
                break

        if top is None:
            return None, None

        bottom = top + 1
        consecutive_not_photo = 0

        for i in range(top + 1, n):
            if not is_not_photo_arr[i]:
                bottom = i + 1
                consecutive_not_photo = 0
            else:
                consecutive_not_photo += 1
                if consecutive_not_photo >= gap_limit:
                    break

        if (bottom - top) < 5:
            return None, None

        return top, bottom

    # =================================================================
    # MOTOR DE RECORTE v11.1
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

        # Máscara de "não-foto"
        not_photo_mask = self._build_not_photo_mask(strip)

        # Limite hard: onde a interface começa
        interface_y = self._find_interface_limit(strip, not_photo_mask)
        if interface_y < strip_h:
            print(f"   [{label}] Interface detectada em Y={interface_y} "
                  f"(corta {strip_h - interface_y}px de botões)")
            # Marca tudo abaixo do limite como "não-foto"
            not_photo_mask[interface_y:, :] = True

        # =============================================
        # LINHAS
        # =============================================
        row_not_photo_pct = np.mean(not_photo_mask, axis=1)
        is_not_photo_row = row_not_photo_pct >= 0.75

        top, bottom = self._find_photo_span(is_not_photo_row, gap_limit=6)

        if top is None:
            print(f"⚠️ [{label}] Sem foto detectada")
            cv2.imwrite(str(DEBUG_DIR / f"{label}_FALLBACK.png"), strip)
            return strip

        top = max(0, top)
        bottom = min(strip_h, bottom)
        print(f"   [{label}] Linhas: top={top} bottom={bottom} (h={bottom-top})")

        # =============================================
        # COLUNAS (só na faixa TOP:BOTTOM)
        # =============================================
        band_mask = not_photo_mask[top:bottom, :]
        col_not_photo_pct = np.mean(band_mask, axis=0)
        is_not_photo_col = col_not_photo_pct >= 0.75

        left, right = self._find_photo_span(is_not_photo_col, gap_limit=6)

        if left is None:
            left = 0
            right = strip_w
            print(f"   [{label}] Colunas: sem bordas → largura total")
        else:
            left = max(0, left)
            right = min(strip_w, right)
            print(f"   [{label}] Colunas: left={left} right={right} (w={right-left})")

        photo = strip[top:bottom, left:right].copy()

        print(f"📐 [{label}] Strip {strip_w}x{strip_h} → "
              f"Foto {right-left}x{bottom-top}")

        # === Debug ===
        debug_vis = strip.copy()
        cv2.rectangle(debug_vis, (left, top), (right, bottom), (0, 255, 0), 2)
        if interface_y < strip_h:
            cv2.line(debug_vis, (0, interface_y), (strip_w, interface_y),
                      (0, 0, 255), 2)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_03_detected.png"), debug_vis)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_04_photo.png"), photo)

        # Graymap linhas
        gm_rows = np.zeros((strip_h, 40, 3), dtype=np.uint8)
        for row in range(strip_h):
            bar_w = int(row_not_photo_pct[row] * 35)
            color = (0, 0, 200) if is_not_photo_row[row] else (0, 255, 0)
            cv2.line(gm_rows, (0, row), (max(1, bar_w), row), color, 1)
        cv2.line(gm_rows, (0, top), (40, top), (255, 0, 255), 2)
        cv2.line(gm_rows, (0, min(bottom, strip_h - 1)),
                  (40, min(bottom, strip_h - 1)), (255, 0, 255), 2)
        if interface_y < strip_h:
            cv2.line(gm_rows, (0, interface_y), (40, interface_y),
                      (0, 0, 255), 2)
        thresh_x = int(0.75 * 35)
        cv2.line(gm_rows, (thresh_x, 0), (thresh_x, strip_h), (0, 255, 255), 1)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_graymap_rows.png"), gm_rows)

        # Graymap colunas
        gm_cols = np.zeros((40, strip_w, 3), dtype=np.uint8)
        for col in range(strip_w):
            bar_h = int(col_not_photo_pct[col] * 35)
            color = (0, 0, 200) if is_not_photo_col[col] else (0, 255, 0)
            cv2.line(gm_cols, (col, 40), (col, 40 - max(1, bar_h)), color, 1)
        if left > 0:
            cv2.line(gm_cols, (left, 0), (left, 40), (255, 0, 255), 2)
        if right < strip_w:
            cv2.line(gm_cols, (min(right, strip_w - 1), 0),
                      (min(right, strip_w - 1), 40), (255, 0, 255), 2)
        thresh_y = 40 - int(0.75 * 35)
        cv2.line(gm_cols, (0, thresh_y), (strip_w, thresh_y), (0, 255, 255), 1)
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