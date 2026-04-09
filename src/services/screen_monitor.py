"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot: Tira um único print ao encontrar o layout e encerra a thread.

Captura a imagem COMPLETA entregue pela AOI, REMOVE o fundo cinza da interface,
e extrai as informações de texto (Board, Parts, Value) via OCR.

Debug: salva as regiões de texto recortadas em public/debug_ocr/ para diagnóstico.
"""
import cv2
import numpy as np
import mss
import re
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from src.config.settings import settings

# Pasta de debug para salvar imagens do OCR
DEBUG_DIR = settings.PUBLIC_DIR / "debug_ocr"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

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

    def _find_text_zones(self, frame_bgr, blue_bar, red_bar):
        """
        Encontra as zonas de texto da AOI.
        
        O texto Board/Parts/Value fica NO TOPO do painel da AOI,
        ACIMA das barras coloridas. Pode estar a centenas de pixels acima.
        
        Estratégia: para cada barra (azul e vermelha), pega TODA a coluna
        acima dela (desde o topo da tela ou desde onde começar o painel da AOI).
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        text_zones = []

        # Usa ambas as barras para referência de largura X
        # A barra mais alta (menor Y) indica onde começa o painel da AOI
        bars = [("sample", blue_bar), ("ng", red_bar)]
        
        # Y mais alto entre as duas barras = topo do painel
        top_bar_y = min(blue_bar[1], red_bar[1])

        for bar_name, bar_rect in bars:
            bx, by, bw, bh = bar_rect

            # ===================================================
            # ZONA PRINCIPAL: TUDO acima da barra, mesma coluna X
            # Desde o topo da tela (ou pelo menos 400px acima)
            # até o início da barra colorida
            # ===================================================
            above_y1 = max(0, by - 400)
            above_y2 = by
            
            if above_y2 > above_y1 + 5:
                zone = frame_bgr[above_y1:above_y2, bx:bx+bw].copy()
                if zone.size > 0:
                    text_zones.append((f"{bar_name}_above_full", zone))
                    
                    # Também tenta só a metade inferior (mais perto da barra)
                    mid_y = (above_y1 + above_y2) // 2
                    zone_lower = frame_bgr[mid_y:above_y2, bx:bx+bw].copy()
                    if zone_lower.size > 0:
                        text_zones.append((f"{bar_name}_above_lower", zone_lower))
                    
                    # E só a metade superior (mais longe da barra)
                    zone_upper = frame_bgr[above_y1:mid_y, bx:bx+bw].copy()
                    if zone_upper.size > 0:
                        text_zones.append((f"{bar_name}_above_upper", zone_upper))

            # ===================================================
            # ZONA EXTRA: Acima da barra mas com a coluna X expandida
            # (o texto pode ser mais largo que a barra)
            # ===================================================
            expand_x = 50
            ex1 = max(0, bx - expand_x)
            ex2 = min(frame_w, bx + bw + expand_x)
            
            if above_y2 > above_y1 + 5:
                zone_wide = frame_bgr[above_y1:above_y2, ex1:ex2].copy()
                if zone_wide.size > 0:
                    text_zones.append((f"{bar_name}_above_wide", zone_wide))

        return text_zones

    def _ocr_region(self, image: np.ndarray, zone_name: str = "") -> str:
        """
        Executa OCR com múltiplas estratégias de pré-processamento.
        Otimizado para fontes pequenas do Windows XP.
        """
        if not HAS_TESSERACT or image is None or image.size == 0:
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Escala agressiva para fontes pequenas do Windows XP
        # Objetivo: texto final com ~30-40px de altura
        scale = max(3, min(6, 300 // max(h, 1)))
        gray_big = cv2.resize(gray, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # Sharpening para melhorar bordas de fontes bitmap do XP
        kernel_sharp = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        gray_sharp = cv2.filter2D(gray_big, -1, kernel_sharp)

        # Múltiplas estratégias de binarização
        attempts = []

        # 1. Otsu
        _, otsu = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(("otsu", otsu))

        # 2. Otsu invertido
        _, otsu_inv = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        attempts.append(("otsu_inv", otsu_inv))

        # 3. Adaptativo
        adaptive = cv2.adaptiveThreshold(
            gray_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        attempts.append(("adaptive", adaptive))

        # 4. Threshold fixo (128 — meio do range, bom pra XP)
        _, fixed = cv2.threshold(gray_big, 128, 255, cv2.THRESH_BINARY)
        attempts.append(("fixed128", fixed))

        # 5. Cinza direto com sharpening
        attempts.append(("sharp", gray_sharp))

        # 6. Cinza direto sem processar (às vezes funciona melhor)
        attempts.append(("gray", gray_big))

        keywords = ['board', 'part', 'value']
        best_text = ""
        best_score = 0

        for strat_name, processed in attempts:
            try:
                for psm in [6, 4, 3, 11]:
                    config = f'--psm {psm} --oem 3'
                    raw = pytesseract.image_to_string(processed, config=config)
                    text = raw.strip()

                    if not text:
                        continue

                    text_lower = text.lower()
                    score = sum(1 for kw in keywords if kw in text_lower)

                    if score > best_score:
                        best_score = score
                        best_text = text
                        # Salva debug da melhor tentativa
                        debug_path = DEBUG_DIR / f"ocr_{zone_name}_{strat_name}_psm{psm}.png"
                        cv2.imwrite(str(debug_path), processed)

                    if score >= 2:
                        return text

            except Exception:
                continue

        # Salva debug original sempre
        if zone_name:
            cv2.imwrite(str(DEBUG_DIR / f"ocr_{zone_name}_original.png"), image)
            cv2.imwrite(str(DEBUG_DIR / f"ocr_{zone_name}_scaled.png"), gray_big)
            if best_text:
                (DEBUG_DIR / f"ocr_{zone_name}_result.txt").write_text(
                    best_text, encoding='utf-8')

        return best_text

    def _extract_text_info(self, frame_bgr, blue_bar, red_bar) -> dict:
        """
        Extrai Board, Parts e Value buscando texto ACIMA das barras coloridas.
        """
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}

        if not HAS_TESSERACT:
            info["raw_text"] = "[OCR não disponível]"
            return info

        text_zones = self._find_text_zones(frame_bgr, blue_bar, red_bar)
        print(f"🔍 OCR: Analisando {len(text_zones)} zonas de texto acima das barras...")

        all_raw = []

        for zone_name, zone_img in text_zones:
            raw = self._ocr_region(zone_img, zone_name)

            if not raw:
                continue

            all_raw.append(f"[{zone_name}]: {raw}")
            print(f"📋 OCR [{zone_name}]: {raw[:200]}")

            for line in raw.split('\n'):
                line_clean = line.strip()
                if not line_clean:
                    continue

                # Board (flexível: Board, Roard, 8oard, Hoard)
                if not info["board"]:
                    m = re.search(
                        r'[BbRr8Hh][Oo0][Aa][Rr][Dd]\s*[:\-\.\s]?\s*(.*)',
                        line_clean
                    )
                    if m and m.group(1).strip():
                        info["board"] = m.group(1).strip()

                # Parts (flexível: Parts, Part, Farts, Fart)
                if not info["parts"]:
                    m = re.search(
                        r'[PpFf][Aa][Rr][Tt][Ss]?\s*[:\-\.\s]?\s*(.*)',
                        line_clean
                    )
                    if m and m.group(1).strip():
                        info["parts"] = m.group(1).strip()

                # Value (flexível: Value, Vaiue, Va1ue, Vatue)
                if not info["value"]:
                    m = re.search(
                        r'[Vv][Aa][Ll1iI][Uu][Ee]\s*[:\-\.\s]?\s*(.*)',
                        line_clean
                    )
                    if m and m.group(1).strip():
                        info["value"] = m.group(1).strip()

            if info["board"] and info["parts"] and info["value"]:
                break

        info["raw_text"] = "\n".join(all_raw) if all_raw else "[Nenhum texto encontrado]"

        print(f"📋 OCR Final — Board: '{info['board']}' | "
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
                                # Debug: salva frame anotado
                                debug_annotated = frame_bgr.copy()
                                # Marca barra azul
                                cv2.rectangle(debug_annotated,
                                              (blue_bar[0], blue_bar[1]),
                                              (blue_bar[0]+blue_bar[2], blue_bar[1]+blue_bar[3]),
                                              (255, 0, 0), 3)
                                cv2.putText(debug_annotated, "AZUL",
                                            (blue_bar[0], blue_bar[1]-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                # Marca barra vermelha
                                cv2.rectangle(debug_annotated,
                                              (red_bar[0], red_bar[1]),
                                              (red_bar[0]+red_bar[2], red_bar[1]+red_bar[3]),
                                              (0, 0, 255), 3)
                                cv2.putText(debug_annotated, "VERM",
                                            (red_bar[0], red_bar[1]-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                # Marca quadrados verdes
                                for gb in green_boxes:
                                    cv2.rectangle(debug_annotated,
                                                  (gb[0], gb[1]),
                                                  (gb[0]+gb[2], gb[1]+gb[3]),
                                                  (0, 255, 0), 2)
                                # Marca zona de busca de texto (400px acima da barra)
                                top_y = min(blue_bar[1], red_bar[1])
                                search_y1 = max(0, top_y - 400)
                                cv2.rectangle(debug_annotated,
                                              (blue_bar[0], search_y1),
                                              (blue_bar[0]+blue_bar[2], top_y),
                                              (255, 255, 0), 2)
                                cv2.putText(debug_annotated, "ZONA TEXTO",
                                            (blue_bar[0], search_y1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                                cv2.imwrite(str(DEBUG_DIR / "frame_anotado.png"), debug_annotated)
                                cv2.imwrite(str(DEBUG_DIR / "frame_completo.png"), frame_bgr)

                                print(f"📐 Barra Azul: x={blue_bar[0]} y={blue_bar[1]} "
                                      f"w={blue_bar[2]} h={blue_bar[3]}")
                                print(f"📐 Barra Verm: x={red_bar[0]} y={red_bar[1]} "
                                      f"w={red_bar[2]} h={red_bar[3]}")
                                print(f"📐 Zona texto: y={search_y1} até y={top_y}")

                                # OCR
                                aoi_info = self._extract_text_info(
                                    frame_bgr, blue_bar, red_bar)

                                self.layout_detected.emit(
                                    crop_sample, crop_ng, aoi_info)
                                self.log_updated.emit(
                                    "Monitor AOI: SNAPSHOT CAPTURADO!")
                                print(f"🔍 Debug OCR salvo em: {DEBUG_DIR}")
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