"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot.

ESTRATÉGIA "ENCONTRAR O RETÂNGULO #c0c0c0":
══════════════════════════════════════════════
O layout da AOI no Windows XP para CADA foto é:

  ████████████████████████████  ← Barra colorida (azul/vermelha)
  ┌──────────────────────────┐
  │ #d4d0c7 (interface)      │  ← pode não existir se foto colar na barra
  │  ┌────────────────────┐  │
  │  │ #c0c0c0 (zona foto)│  │  ← retângulo sólido, sempre existe
  │  │  ┌──────────┐      │  │
  │  │  │  FOTO    │      │  │  ← foto dentro do #c0c0c0
  │  │  │  REAL    │      │  │
  │  │  └──────────┘      │  │
  │  │ #c0c0c0            │  │
  │  └────────────────────┘  │
  │ #d4d0c7                  │
  └──────────────────────────┘

O retângulo #c0c0c0 pode:
- Começar logo abaixo da barra (sem #d4d0c7 acima)
- Ou ter uma faixa de #d4d0c7 acima

A foto pode:
- Preencher todo o #c0c0c0 (sem margem)
- Preencher só largura (margem em cima/baixo)
- Preencher só altura (margem nos lados)
- Ser menor que o #c0c0c0 em ambas as direções

COMO ENCONTRAR:
1. Recorta strip (largura da barra, barra→baixo)
2. Classifica CADA pixel: é #c0c0c0? é #d4d0c7? é outro (foto)?
3. O retângulo #c0c0c0 é encontrado por scan das bordas:
   - De CIMA pra baixo: primeira linha com #c0c0c0 dominante
   - De BAIXO pra cima: primeira linha com #c0c0c0 dominante
   - Da ESQUERDA pra direita: primeira coluna com #c0c0c0 dominante
   - Da DIREITA pra esquerda: primeira coluna com #c0c0c0 dominante
4. Tudo dentro desse retângulo = foto (pegamos INTEIRO, incluindo o #c0c0c0)
5. Depois, DENTRO do retângulo, removemos o #c0c0c0 das bordas
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
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        return cv2.boundingRect(max(valid, key=cv2.contourArea))

    # =================================================================
    # CLASSIFICAÇÃO DE PIXELS POR COR
    # =================================================================

    def _classify_pixels(self, region_bgr):
        """
        Classifica cada pixel do recorte em 3 categorias:
          0 = FOTO (nem #c0c0c0 nem #d4d0c7)
          1 = ZONA_FOTO (#c0c0c0 — fundo do retângulo onde a foto fica)
          2 = INTERFACE (#d4d0c7 — borda da interface XP)

        Ambas as cores são cinza (R≈G≈B), a diferença é o brilho:
          #c0c0c0 = 192,192,192 (mais escuro)
          #d4d0c7 = 212,208,199 (mais claro, levemente amarelado)

        Como é um print de tela, usamos tolerância generosa.
        """
        b, g, r = cv2.split(region_bgr)
        b_i = b.astype(np.int16)
        g_i = g.astype(np.int16)
        r_i = r.astype(np.int16)

        # Diferença máxima entre canais (cinza = baixa diferença)
        max_c = np.maximum(np.maximum(b_i, g_i), r_i)
        min_c = np.minimum(np.minimum(b_i, g_i), r_i)
        spread = max_c - min_c
        brightness = ((b_i + g_i + r_i) / 3).astype(np.int16)

        # É cinza? (baixa variação entre canais)
        is_gray = spread <= 25

        # #c0c0c0 = brilho ~192 (range: 175-205)
        is_c0 = is_gray & (brightness >= 175) & (brightness <= 205)

        # #d4d0c7 = brilho ~206 (range: 195-225), levemente mais claro
        # Também tem leve tom amarelo (R>B), mas com print pode não ser visível
        is_d4 = is_gray & (brightness >= 195) & (brightness <= 225)

        # Se um pixel se encaixa em ambos (zona de sobreposição 195-205),
        # usa a saturação como desempate: #d4d0c7 é levemente amarelado
        overlap = is_c0 & is_d4
        if np.any(overlap):
            # #d4d0c7 tem R > B tipicamente
            r_minus_b = r_i - b_i
            # Se R-B > 5 na zona de sobreposição → é interface
            is_d4[overlap] = r_minus_b[overlap] > 5
            is_c0[overlap] = ~is_d4[overlap]

        # Mapa: 0=foto, 1=zona_foto, 2=interface
        result = np.zeros(region_bgr.shape[:2], dtype=np.uint8)
        result[is_c0] = 1  # zona de foto
        result[is_d4] = 2  # interface

        return result

    # =================================================================
    # ENCONTRAR O RETÂNGULO DA ZONA DE FOTO (#c0c0c0)
    # =================================================================

    def _find_photo_zone_rect(self, pixel_map):
        """
        Encontra o retângulo #c0c0c0 no mapa de pixels.
        
        Varre de fora pra dentro em cada direção.
        O retângulo #c0c0c0 é a região onde NÃO é interface (#d4d0c7).
        
        Dentro do retângulo pode ter:
        - Pixels tipo 1 (#c0c0c0 = margem ao redor da foto)
        - Pixels tipo 0 (foto real)
        
        Retorna (x1, y1, x2, y2) ou None
        """
        h, w = pixel_map.shape

        # Máscara: 1 onde é zona_foto(1) ou foto(0), 0 onde é interface(2)
        not_interface = (pixel_map != 2).astype(np.uint8)

        # Densidade por linha: fração de pixels que NÃO são interface
        row_not_iface = np.sum(not_interface, axis=1).astype(np.float32) / w
        col_not_iface = np.sum(not_interface, axis=0).astype(np.float32) / h

        # A zona de foto começa onde > 50% da linha não é interface
        # (conservador — a zona de foto ocupa a maioria da largura)
        threshold = 0.3

        active_rows = np.where(row_not_iface >= threshold)[0]
        active_cols = np.where(col_not_iface >= threshold)[0]

        if len(active_rows) < 5 or len(active_cols) < 5:
            return None

        y1 = int(active_rows[0])
        y2 = int(active_rows[-1]) + 1
        x1 = int(active_cols[0])
        x2 = int(active_cols[-1]) + 1

        if (y2 - y1) < 15 or (x2 - x1) < 15:
            return None

        return x1, y1, x2, y2

    # =================================================================
    # DENTRO DO RETÂNGULO: REMOVER MARGENS #c0c0c0
    # =================================================================

    def _trim_c0_margins(self, zone_bgr, pixel_map_zone):
        """
        Dentro do retângulo #c0c0c0, remove as margens cinza ao redor da foto.
        
        Se a foto não preenche todo o retângulo, sobra #c0c0c0 nas bordas.
        Varre de fora pra dentro em cada eixo para encontrar onde a foto começa.
        
        Se a foto preenche tudo → retorna tudo (nada a remover).
        """
        h, w = pixel_map_zone.shape

        # Máscara: 1 onde é foto (tipo 0), 0 onde é #c0c0c0 (tipo 1)
        is_photo = (pixel_map_zone == 0).astype(np.uint8)

        # Conta pixels de foto por linha e coluna
        row_photo = np.sum(is_photo, axis=1).astype(np.float32) / w
        col_photo = np.sum(is_photo, axis=0).astype(np.float32) / h

        # A foto pode ter MUITO pouca variação de cor em algumas áreas
        # (áreas escuras uniformes). Então usamos um limiar muito baixo.
        threshold = 0.02

        active_rows = np.where(row_photo >= threshold)[0]
        active_cols = np.where(col_photo >= threshold)[0]

        # Se quase nenhum pixel é "foto" → a foto pode ser toda cinza
        # (ex: placa com muita solda/cobre que parece cinza)
        # Nesse caso, retorna a zona inteira
        if len(active_rows) < 3 or len(active_cols) < 3:
            return zone_bgr

        y1 = int(active_rows[0])
        y2 = int(active_rows[-1]) + 1
        x1 = int(active_cols[0])
        x2 = int(active_cols[-1]) + 1

        # Se o trim removeu mais de 80% → provavelmente errou, retorna tudo
        trimmed_area = (y2 - y1) * (x2 - x1)
        total_area = h * w
        if trimmed_area < total_area * 0.2:
            return zone_bgr

        return zone_bgr[y1:y2, x1:x2].copy()

    # =================================================================
    # PIPELINE COMPLETO: barra → foto
    # =================================================================

    def _extract_photo(self, frame_bgr, bar_rect, label=""):
        """
        Extrai a foto abaixo de UMA barra colorida.

        1. Strip: largura da barra, da barra pra baixo 650px
        2. Classifica pixels (foto / #c0c0c0 / #d4d0c7)
        3. Encontra retângulo #c0c0c0 (zona de foto)
        4. Dentro do retângulo, remove margens #c0c0c0
        5. Resultado = foto real
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]

        # Strip generoso abaixo da barra
        x1 = max(0, bx)
        x2 = min(frame_w, bx + bw)
        y1 = by + bh
        y2 = min(frame_h, y1 + 650)

        if y2 - y1 < 20 or x2 - x1 < 20:
            return np.array([])

        strip = frame_bgr[y1:y2, x1:x2].copy()
        strip_h, strip_w = strip.shape[:2]

        # Classifica pixels
        pmap = self._classify_pixels(strip)

        # Encontra o retângulo da zona de foto
        zone_rect = self._find_photo_zone_rect(pmap)

        if zone_rect is None:
            print(f"⚠️ [{label}] Zona #c0c0c0 não encontrada → "
                  f"fallback strip inteiro {strip_w}x{strip_h}")
            return strip

        zx1, zy1, zx2, zy2 = zone_rect
        zone_bgr = strip[zy1:zy2, zx1:zx2].copy()
        zone_pmap = pmap[zy1:zy2, zx1:zx2]

        print(f"📐 [{label}] Strip {strip_w}x{strip_h} → "
              f"Zona #c0c0c0 {zx2-zx1}x{zy2-zy1} em ({zx1},{zy1})")

        # Conta quanto da zona é foto vs #c0c0c0
        total_zone = zone_pmap.size
        foto_count = np.sum(zone_pmap == 0)
        c0_count = np.sum(zone_pmap == 1)
        foto_pct = foto_count / total_zone if total_zone > 0 else 0

        # Se >90% é foto → foto preenche toda a zona, retorna direto
        if foto_pct > 0.90:
            print(f"   → Foto preenche {foto_pct:.0%} da zona, retornando inteira")
            return zone_bgr

        # Se >70% é foto → foto quase preenche, trim sutil
        if foto_pct > 0.70:
            photo = self._trim_c0_margins(zone_bgr, zone_pmap)
            print(f"   → Foto {foto_pct:.0%}, trim → {photo.shape[1]}x{photo.shape[0]}")
            return photo

        # Se <70% é foto → tem bastante margem #c0c0c0, trim agressivo
        photo = self._trim_c0_margins(zone_bgr, zone_pmap)
        print(f"   → Foto {foto_pct:.0%}, margens c0 removidas → "
              f"{photo.shape[1]}x{photo.shape[0]}")
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

                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                mask_blue = cv2.inRange(
                    hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
                mask_red1 = cv2.inRange(
                    hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
                mask_red2 = cv2.inRange(
                    hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                blue_bar = self._find_color_bar(mask_blue)
                red_bar = self._find_color_bar(mask_red)

                if blue_bar is not None and red_bar is not None:
                    # Cada barra é 100% independente
                    crop_sample = self._extract_photo(
                        frame_bgr, blue_bar, "AZUL")
                    crop_ng = self._extract_photo(
                        frame_bgr, red_bar, "VERM")

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
                    if blue_bar is not None and red_bar is not None:
                        self.log_updated.emit(
                            "Monitor AOI: LAYOUT DETECTADO! 📡 Analisando...")
                    else:
                        self.log_updated.emit(
                            "Monitor AOI: Aguardando interface da máquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()