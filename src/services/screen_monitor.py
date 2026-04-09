# src\services\screen_monitor.py
"""
Módulo responsável pela captura de tela e detecção do Layout da AOI.
Modo One-Shot: Tira um único print ao encontrar o layout e encerra a thread.

Estratégia Cross-Scan (Cruzeta): Varre a partir do centro das barras coloridas
em formato de cruz (+) até encontrar a parede cinza (#d4d0c7) do software,
garantindo o recorte cirúrgico e simétrico da foto da placa.
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
            print(f"Tesseract encontrado: {p}")
            break

    if not HAS_TESSERACT:
        import shutil
        auto_path = shutil.which("tesseract")
        if auto_path:
            pytesseract.pytesseract.tesseract_cmd = auto_path
            HAS_TESSERACT = True
            print(f"Tesseract encontrado no PATH: {auto_path}")
        else:
            print("Tesseract NAO encontrado. Ajuste TESSERACT_CMD em settings.py")

except ImportError:
    print("pytesseract nao instalado. OCR desativado.")


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

    def _cross_scan_crop(self, frame_bgr, bar_rect):
        """
        Executa a lógica humana: desce a partir da barra até bater no cinza,
        depois vai para a esquerda e direita até bater no cinza.
        Retorna as coordenadas exatas: (y_topo, y_fundo, x_esq, x_dir)
        """
        bx, by, bw, bh = bar_rect
        frame_h, frame_w = frame_bgr.shape[:2]
        
        # Ponto central da barra
        cx = bx + (bw // 2)
        start_y = by + bh

        # Cor do fundo da AOI: #d4d0c7 (BGR: 199, 208, 212)
        bg_color = np.array([199, 208, 212], dtype=np.float32)
        tolerance = 30 # Distância máxima de cor para ser considerado "Fundo Cinza"

        # ==========================================
        # PASSO 1: DESCER ATÉ ACHAR O FUNDO (Eixo Y)
        # ==========================================
        # Pegamos uma fatia vertical de 10 pixels de largura no centro da barra para evitar ruídos
        slice_w = 5
        x1_slice = max(0, cx - slice_w)
        x2_slice = min(frame_w, cx + slice_w)
        
        col_slice = frame_bgr[start_y:, x1_slice:x2_slice]
        if col_slice.size == 0:
            return None
            
        # Calcula a diferença da fatia para o cinza do fundo
        diff_y = np.linalg.norm(col_slice.astype(np.float32) - bg_color, axis=2)
        # Se algum pixel na linha for DIFERENTE do cinza, é porque é foto/texto (True)
        is_content_y = np.any(diff_y > tolerance, axis=1)
        
        # Encontra blocos contínuos de "Conteúdo" (True)
        runs_y = np.flatnonzero(np.diff(np.r_[np.int8(0), is_content_y.view(np.int8), np.int8(0)])).reshape(-1, 2)
        if len(runs_y) == 0:
            return None
            
        # O maior bloco contínuo de conteúdo abaixo da barra É a foto da placa
        longest_run_y = max(runs_y, key=lambda r: r[1] - r[0])
        
        # Se o bloco for muito pequeno, falhou
        if longest_run_y[1] - longest_run_y[0] < 50:
            return None
            
        top_y = start_y + longest_run_y[0]
        bottom_y = start_y + longest_run_y[1]

        # ==========================================
        # PASSO 2: ESQUERDA E DIREITA (Eixo X)
        # ==========================================
        # Vamos exatamente para o meio vertical da foto que achamos
        cy = (top_y + bottom_y) // 2
        
        # Pegamos uma fatia horizontal de 10 pixels de altura
        y1_slice = max(0, cy - slice_w)
        y2_slice = min(frame_h, cy + slice_w)
        
        row_slice = frame_bgr[y1_slice:y2_slice, :]
        
        diff_x = np.linalg.norm(row_slice.astype(np.float32) - bg_color, axis=2)
        is_content_x = np.any(diff_x > tolerance, axis=0)
        
        runs_x = np.flatnonzero(np.diff(np.r_[np.int8(0), is_content_x.view(np.int8), np.int8(0)])).reshape(-1, 2)
        if len(runs_x) == 0:
            return None
            
        # A largura correta é o bloco de conteúdo que PASSA pelo centro da nossa barra
        valid_runs_x = [r for r in runs_x if r[0] <= cx <= r[1]]
        
        if not valid_runs_x:
            valid_runs_x = [max(runs_x, key=lambda r: r[1] - r[0])]
            
        left_x = valid_runs_x[0][0]
        right_x = valid_runs_x[0][1]

        # Margem de segurança (2px para dentro) para não pegar nenhuma bordinha preta/cinza
        m = 2
        return (top_y + m, bottom_y - m, left_x + m, right_x - m)

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
        """
        Extrai Board, Parts e Value do texto ACIMA das barras coloridas.
        """
        info = {"board": "", "parts": "", "value": "", "raw_text": ""}

        if not HAS_TESSERACT:
            info["raw_text"] = "[OCR nao disponivel]"
            return info

        bx, by, bw, bh = blue_bar
        frame_h, frame_w = frame_bgr.shape[:2]

        above_y1 = max(0, by - 400)
        above_y2 = by

        if above_y2 <= above_y1 + 5:
            info["raw_text"] = "[Zona de texto muito pequena]"
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
            line_clean = line.strip()
            if not line_clean:
                continue

            if not info["board"]:
                m = re.search(
                    r'[BbRr8Hh][Oo0][Aa][Rr][Dd]\s*[:\-\.\s]\s*(\S.*)',
                    line_clean
                )
                if m:
                    info["board"] = m.group(1).strip()

            if not info["parts"]:
                m = re.search(
                    r'[PpFf][Aa][Rr][Tt][Ss]?\s*[:\-\.\s]\s*(.+?)(?:\s+[Bb][Ll][Oo][Cc][Kk]|\s+[Vv][Aa][Ll]|\s*$)',
                    line_clean
                )
                if m:
                    val = m.group(1).strip()
                    val = re.sub(r'\s+$', '', val)
                    if val:
                        info["parts"] = val

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

        print(f"OCR - Board: '{info['board']}' | Parts: '{info['parts']}' | Value: '{info['value']}'")

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
                    # Continua exigindo a presença de caixas verdes na tela para ativar o gatilho de captura
                    green_cnts, _ = cv2.findContours(
                        mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_boxes = [cv2.boundingRect(c) for c in green_cnts if cv2.contourArea(c) > 500]

                    if len(green_boxes) >= 1:
                        
                        # 1. Faz o Cross-Scan na Barra Azul (Sample) para descobrir a geometria exata da foto
                        coords_sample = self._cross_scan_crop(frame_bgr, blue_bar)
                        
                        if coords_sample:
                            sy1, sy2, sx1, sx2 = coords_sample
                            crop_sample = frame_bgr[sy1:sy2, sx1:sx2].copy()
                            
                            # 2. SIMETRIA ABSOLUTA: Aplica as dimensões perfeitas descobertas na Sample para a NG
                            # Isso resolve todos os problemas de recortes pequenos no lado NG.
                            rx, ry, rw, rh = red_bar
                            red_center_x = rx + (rw // 2)
                            
                            photo_width = sx2 - sx1
                            half_width = photo_width // 2
                            
                            nx1 = red_center_x - half_width
                            nx2 = nx1 + photo_width
                            
                            # Garante que não vai tentar recortar fora da tela
                            nx1 = max(0, nx1)
                            nx2 = min(frame_bgr.shape[1], nx2)
                            
                            crop_ng = frame_bgr[sy1:sy2, nx1:nx2].copy()

                            if crop_sample.size > 0 and crop_ng.size > 0:
                                # OCR
                                aoi_info = self._extract_text_info(frame_bgr, blue_bar, red_bar)

                                self.layout_detected.emit(crop_sample, crop_ng, aoi_info)
                                self.log_updated.emit("Monitor AOI: SNAPSHOT CAPTURADO COM SUCESSO!")
                                self.running = False
                                break

                frame_count += 1
                if frame_count % 15 == 0:
                    if has_blue and has_red:
                        self.log_updated.emit("Monitor AOI: LAYOUT DETECTADO! Analisando...")
                    else:
                        self.log_updated.emit("Monitor AOI: Aguardando interface da maquina...")

                self.msleep(int(1000 / settings.SCREEN_CAPTURE_FPS))

    def stop(self):
        self.running = False
        self.wait()