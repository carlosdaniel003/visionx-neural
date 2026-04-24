# src/core/epicenter_extractor.py
import cv2
import numpy as np
import math
from typing import Tuple, List

class EpicenterExtractor:
    """
    Especialista isolado para buscar a caixa de foco da AOI.
    Usa o Radar Euclidiano (Centro para Fora) para ignorar sujeiras
    e a moldura gigante, focando no objeto verde válido mais central.
    """
    @staticmethod
    def extract_focus(sample_crop: np.ndarray, ng_crop: np.ndarray, old_epicenters: list, global_box_info: dict) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Retorna: (Lista de Epicentros Reais, Gabarito Recortado, Teste Recortado)
        """
        real_epicenters = []
        img_h, img_w = sample_crop.shape[:2]
        
        # =====================================================================
        # RADAR EUCLIDIANO: Busca Centro-Para-Fora (Center-Out Search)
        # =====================================================================
        try:
            hsv = cv2.cvtColor(sample_crop, cv2.COLOR_BGR2HSV)
            lower_green = np.array([50, 150, 100])
            upper_green = np.array([75, 255, 255])
            
            mask = cv2.inRange(hsv, lower_green, upper_green)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            center_x, center_y = img_w / 2, img_h / 2
            valid_boxes = []
            
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                # Regras: Maior que 15px (Ignora poeira), Menor que 85% (Ignora Moldura)
                if w > 15 and h > 15 and w < (img_w * 0.85) and h < (img_h * 0.85):
                    box_cx = x + (w / 2)
                    box_cy = y + (h / 2)
                    dist = math.sqrt((center_x - box_cx)**2 + (center_y - box_cy)**2)
                    valid_boxes.append({"box": (x, y, w, h), "dist": dist})
            
            if valid_boxes:
                valid_boxes.sort(key=lambda b: b["dist"]) 
                real_epicenters.append(valid_boxes[0]["box"]) 
                
        except Exception as e:
            print(f"⚠️ Erro no Radar Euclidiano: {e}")

        # Fallback 1: Antigo sistema de hierarquia invertida
        if not real_epicenters:
            if old_epicenters:
                old_epicenters.sort(key=lambda b: b[2] * b[3], reverse=True)
                for (x, y, w, h) in old_epicenters:
                    if w < img_w * 0.90 and h < img_h * 0.90 and w > 20 and h > 20:
                        real_epicenters.append((x, y, w, h))
                        break
            # Fallback 2: Caixa global
            elif global_box_info:
                 x = global_box_info.get("x", 0)
                 y = global_box_info.get("y", 0)
                 w = global_box_info.get("w", img_w)
                 h = global_box_info.get("h", img_h)
                 if 20 < w < img_w * 0.90 and 20 < h < img_h * 0.90:
                     real_epicenters.append((x, y, w, h))

        # =====================================================================
        # RECORTE DO EPICENTRO (Segurança e Validação)
        # =====================================================================
        focus_gab = np.array([])
        focus_ng = np.array([])

        if real_epicenters:
            ex, ey, ew, eh = real_epicenters[0] 
            try:
                pad = 0
                y1 = max(0, ey + pad)
                y2 = min(img_h, ey + eh - pad)
                x1 = max(0, ex + pad)
                x2 = min(img_w, ex + ew - pad)
                
                if y2 > y1 and x2 > x1:
                    focus_gab = sample_crop[y1:y2, x1:x2].copy()
                    focus_ng = ng_crop[y1:y2, x1:x2].copy()
                    
                    if focus_gab.shape != focus_ng.shape:
                        focus_ng = cv2.resize(focus_ng, (focus_gab.shape[1], focus_gab.shape[0]))
            except Exception as e:
                print(f"⚠️ Erro ao fatiar matriz da imagem no Extrator: {e}")

        return real_epicenters, focus_gab, focus_ng