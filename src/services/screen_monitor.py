"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot.

PROBLEMAS IDENTIFICADOS NAS VERSÕES ANTERIORES:
1. _find_color_bar pegava o MAIOR contorno vermelho, que podia ser
   o botão "1. NG" ou a borda da janela em vez da barra "NG Image"
2. O strip ia até 700px abaixo da barra, incluindo botões e interface
3. A foto não era separada do fundo #c0c0c0 corretamente

ESTRATÉGIA v6 "BARRAS IRMÃS + UNIFORMIDADE":
1. Encontra a barra AZUL (confiável — só tem uma)
2. Encontra a barra VERMELHA que está NA MESMA ALTURA e tem TAMANHO SIMILAR
   (são barras irmãs — "Sample Image" e "NG Image")
3. Strip: largura da barra, da barra pra baixo
4. Limite inferior do strip: onde a barra OPOSTA estaria se existisse
   (as duas barras marcam uma faixa horizontal — abaixo de uma barra
   e antes da próxima seção de interface)
5. Dentro do strip: uniformidade de linha para achar a foto
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
    # DETECÇÃO DE BARRAS — versão melhorada
    # =================================================================

    def _find_color_bar(self, mask, min_area=2000):
        """Encontra a maior barra colorida."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        return cv2.boundingRect(max(valid, key=cv2.contourArea))

    def _find_sibling_bar(self, mask, reference_bar, min_area=1000):
        """
        Encontra a barra vermelha que é "irmã" da barra azul.
        Critérios: mesma altura Y (±30px), largura similar (±50%),
        formato de barra (largura >> altura).

        Isso evita pegar o botão "1. NG" ou bordas vermelhas da janela.
        """
        ref_x, ref_y, ref_w, ref_h = reference_bar

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # Deve ser uma barra (largura >> altura)
            if w < h * 2:
                continue

            # Deve estar na mesma faixa de Y que a referência (±30px)
            if abs(y - ref_y) > 30:
                continue

            # Deve ter altura similar (±100%)
            if ref_h > 0 and (h > ref_h * 2 or h < ref_h * 0.3):
                continue

            # Deve ter largura razoável (pelo menos 30% da referência)
            if ref_w > 0 and w < ref_w * 0.3:
                continue

            # Não pode ser a mesma barra (deve estar em X diferente)
            if abs(x - ref_x) < ref_w * 0.5:
                continue

            # Score: quanto mais parecida com a referência, melhor
            score = cv2.contourArea(c)
            candidates.append((score, (x, y, w, h)))

        if not candidates:
            return None

        # Retorna a candidata com maior área (mais provável de ser a barra)
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # =================================================================
    # MOTOR DE RECORTE
    # =================================================================

    def _extract_photo(self, frame_bgr, bar_rect, max_height, label=""):
        """
        Extrai a foto abaixo de UMA barra colorida.

        bar_rect: (x, y, w, h) da barra colorida
        max_height: altura máxima do strip (distância entre barras e
                    próxima seção da interface)
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        # Strip: mesma largura da barra, da barra até max_height
        x1 = max(0, bx)
        x2 = min(frame_w, bx + bw)
        y1 = by + bh
        y2 = min(frame_h, y1 + max_height)

        if y2 - y1 < 20 or x2 - x1 < 20:
            return np.array([])

        strip = frame_bgr[y1:y2, x1:x2].copy()
        strip_h, strip_w = strip.shape[:2]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        # Debug
        cv2.imwrite(str(DEBUG_DIR / f"{label}_01_strip.png"), strip)

        # === Variação por linha (desvio padrão horizontal) ===
        gray_f = gray.astype(np.float32)
        row_std = np.std(gray_f, axis=1)
        col_std = np.std(gray_f, axis=0)

        # Debug: gráfico de variação
        self._debug_variation(row_std, col_std, label, strip_w, strip_h)

        # Threshold: interface/fundo tem std ≈ 0, foto tem std > threshold
        var_threshold = 1.5

        # === Encontra TOP e BOTTOM da foto ===
        # Foto = linhas com variação acima do threshold
        is_photo_row = row_std >= var_threshold

        # Para BOTTOM: varre de CIMA pra baixo.
        # A foto é o PRIMEIRO bloco contínuo de linhas com variação.
        # Quando encontra um gap grande de linhas uniformes DEPOIS da foto,
        # é o fundo #c0c0c0 → para ali.
        top, bottom = self._find_first_block(is_photo_row, min_block=3,
                                              max_gap=8)

        # === Encontra LEFT e RIGHT da foto ===
        is_photo_col = col_std >= var_threshold
        left, right = self._find_first_block(is_photo_col, min_block=3,
                                              max_gap=8)

        if top is None or left is None:
            print(f"⚠️ [{label}] Foto não detectada → fallback")
            print(f"   row_std: min={row_std.min():.2f} max={row_std.max():.2f}")
            print(f"   col_std: min={col_std.min():.2f} max={col_std.max():.2f}")
            cv2.imwrite(str(DEBUG_DIR / f"{label}_02_FALLBACK.png"), strip)
            return strip

        # Proteção de limites
        top = max(0, top)
        bottom = min(strip_h, bottom)
        left = max(0, left)
        right = min(strip_w, right)

        photo = strip[top:bottom, left:right].copy()

        print(f"📐 [{label}] Strip {strip_w}x{strip_h} → "
              f"Foto {right-left}x{bottom-top} em ({left},{top})-({right},{bottom})")

        # Debug
        debug_vis = strip.copy()
        cv2.rectangle(debug_vis, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_03_detected.png"), debug_vis)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_04_photo.png"), photo)

        return photo

    def _find_first_block(self, is_active, min_block=3, max_gap=8):
        """
        Encontra o PRIMEIRO bloco contínuo de True no array,
        permitindo gaps pequenos (≤ max_gap linhas uniformes no meio da foto).

        Retorna (start, end) do bloco, ou (None, None).

        A ideia é:
        - Varre do início ao fim
        - Quando encontra uma sequência de True (foto), marca o início
        - Permite gaps pequenos (partes uniformes DENTRO da foto)
        - Quando encontra um gap GRANDE (> max_gap), a foto acabou
        """
        n = len(is_active)
        if not np.any(is_active):
            return None, None

        # Encontra o primeiro True
        first_true = np.argmax(is_active)

        # A partir do primeiro True, varre pra frente
        block_start = int(first_true)
        block_end = block_start
        gap_count = 0

        for i in range(block_start, n):
            if is_active[i]:
                block_end = i + 1
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap:
                    # Gap grande demais → foto acabou
                    break

        # Valida tamanho mínimo
        if (block_end - block_start) < min_block:
            return None, None

        return block_start, block_end

    def _debug_variation(self, row_std, col_std, label, strip_w, strip_h):
        """Salva gráficos de variação para debug."""
        threshold = 1.5

        # Gráfico de variação por LINHA
        h = len(row_std)
        gw = 300
        graph = np.zeros((h, gw, 3), dtype=np.uint8)
        max_val = max(row_std.max(), 1)
        for i, val in enumerate(row_std):
            bar_len = int((val / max_val) * (gw - 10))
            color = (0, 255, 0) if val >= threshold else (0, 0, 255)
            cv2.line(graph, (0, i), (bar_len, i), color, 1)
        tx = int((threshold / max_val) * (gw - 10))
        cv2.line(graph, (tx, 0), (tx, h), (0, 255, 255), 1)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_var_rows.png"), graph)

        # Gráfico de variação por COLUNA
        w = len(col_std)
        gh = 200
        graph2 = np.zeros((gh, w, 3), dtype=np.uint8)
        max_val2 = max(col_std.max(), 1)
        for j, val in enumerate(col_std):
            bar_len = int((val / max_val2) * (gh - 10))
            color = (0, 255, 0) if val >= threshold else (0, 0, 255)
            cv2.line(graph2, (j, gh), (j, gh - bar_len), color, 1)
        ty = gh - int((threshold / max_val2) * (gh - 10))
        cv2.line(graph2, (0, ty), (w, ty), (0, 255, 255), 1)
        cv2.imwrite(str(DEBUG_DIR / f"{label}_var_cols.png"), graph2)

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

                # === Passo 1: Encontra barra AZUL (confiável) ===
                blue_bar = self._find_color_bar(mask_blue)
                if blue_bar is None:
                    frame_count += 1
                    if frame_count % 15 == 0:
                        self.log_updated.emit(
                            "Monitor AOI: Aguardando interface da máquina...")
                    self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))
                    continue

                # === Passo 2: Encontra barra VERMELHA irmã ===
                # Usa a barra azul como referência para encontrar
                # a barra vermelha correta (mesma altura Y, tamanho similar)
                red_bar = self._find_sibling_bar(mask_red, blue_bar)

                if red_bar is None:
                    # Fallback: tenta a maior barra vermelha
                    red_bar = self._find_color_bar(mask_red)

                if red_bar is None:
                    frame_count += 1
                    if frame_count % 15 == 0:
                        self.log_updated.emit(
                            "Monitor AOI: Barra azul OK, vermelha não encontrada...")
                    self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))
                    continue

                # === Passo 3: Calcula a altura máxima da zona de fotos ===
                # A zona de fotos vai das barras até os botões/interface
                # abaixo. Estimamos olhando a distância entre as barras
                # e a borda inferior da tela, limitando a ~60% da tela.
                bar_bottom = max(blue_bar[1] + blue_bar[3],
                                 red_bar[1] + red_bar[3])
                # Procura onde tem interface abaixo das fotos
                # (botões "0. OK", "1. NG", "Answer", etc)
                # Limitamos a 600px ou até encontrar elementos de interface
                max_photo_height = min(600, frame_h - bar_bottom - 50)

                # Tenta encontrar o limite inferior real:
                # Varre as linhas abaixo das barras no frame inteiro
                # procurando onde a interface recomeça (botões, texto)
                # Usa a coluna central entre as duas barras
                mid_x = (blue_bar[0] + blue_bar[0] + blue_bar[2]) // 2
                scan_col = frame_bgr[bar_bottom:bar_bottom + max_photo_height,
                                      max(0, mid_x - 5):min(frame_w, mid_x + 5)]
                if scan_col.size > 0:
                    scan_gray = cv2.cvtColor(scan_col, cv2.COLOR_BGR2GRAY)
                    scan_std = np.std(scan_gray.astype(np.float32), axis=1)
                    # Variação alta na coluna central = ainda é foto/fundo
                    # Quando cai pra 0 e depois sobe (botão) = limite
                    # Por segurança, usamos max_photo_height

                print(f"🔍 Barras: Azul({blue_bar[0]},{blue_bar[1]} "
                      f"w={blue_bar[2]} h={blue_bar[3]}) | "
                      f"Verm({red_bar[0]},{red_bar[1]} "
                      f"w={red_bar[2]} h={red_bar[3]}) | "
                      f"Max altura: {max_photo_height}px")

                # === Passo 4: Extrai fotos ===
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