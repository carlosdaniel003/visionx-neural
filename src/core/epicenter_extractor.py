# src/core/epicenter_extractor.py
import cv2
import numpy as np
import math
from typing import Tuple

from src.core.epicenter_line_recovery import (
    RADAR_GREEN_LOWER,
    RADAR_GREEN_UPPER,
    recover_nested_frame_focus,
)

class EpicenterExtractor:
    """
    Especialista isolado para buscar a caixa de foco da AOI.
    Usa o Radar Euclidiano (Centro para Fora) para ignorar sujeiras
    e a moldura gigante, focando no objeto verde válido mais central.
    """
    @staticmethod
    def select_radar_candidate(
        boxes: list,
        image_shape: tuple,
        old_epicenters: list | None = None,
        global_box_info: dict | None = None,
    ) -> tuple[tuple[int, int, int, int] | None, str]:
        """Seleciona o foco pela hierarquia de molduras antes da distância.

        A moldura global não é uma ROI. Ela pode ocupar menos que 85% da
        altura e, ainda assim, parecer perfeitamente centralizada. Só
        elegemos um candidato como epicentro quando existe uma moldura
        interior geometricamente distinta; na ausência de moldura externa
        reconhecida, preservamos o desempate radial para caixas independentes.
        """
        height, width = image_shape[:2]
        candidates = []
        for box in boxes or []:
            try:
                x, y, w, h = (int(value) for value in box[:4])
            except (ValueError, TypeError):
                continue
            if w <= 15 or h <= 15:
                continue
            if w >= width * 0.85 and h >= height * 0.85:
                continue
            candidates.append((x, y, w, h))

        if not candidates:
            return None, "no_contour_candidate"

        def area(box):
            return box[2] * box[3]

        def nested(inner, outer):
            x, y, w, h = inner
            ox, oy, ow, oh = outer
            if area(inner) >= area(outer) * 0.85:
                return False
            margin = max(5, round(min(height, width) * 0.025))
            return (
                x >= ox - margin
                and y >= oy - margin
                and x + w <= ox + ow + margin
                and y + h <= oy + oh + margin
            )

        def distance(box):
            x, y, w, h = box
            return math.hypot(x + w / 2 - width / 2, y + h / 2 - height / 2)

        def iou(first, second):
            ax, ay, aw, ah = first
            bx, by, bw, bh = second
            intersection_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            intersection_h = max(0, min(ay + ah, by + bh) - max(ay, by))
            intersection = intersection_w * intersection_h
            union = area(first) + area(second) - intersection
            return intersection / union if union else 0.0

        global_frame = None
        if isinstance(global_box_info, dict) and global_box_info.get("detected"):
            try:
                global_frame = tuple(
                    int(global_box_info[key]) for key in ("x", "y", "w", "h")
                )
                if global_frame[2] <= 0 or global_frame[3] <= 0:
                    global_frame = None
            except (TypeError, ValueError, KeyError):
                global_frame = None

        if global_frame is not None:
            # Impede que os dois contornos da espessura da moldura externa
            # sejam confundidos com duas regiões distintas.
            inner_candidates = [
                box for box in candidates if nested(box, global_frame)
            ]
        else:
            inner_candidates = [
                box for box in candidates
                if any(nested(box, outer) for outer in candidates if outer != box)
            ]

        if inner_candidates:
            # A AOI pode apresentar três níveis de borda:
            # contorno externo da tela, caixa geral do componente e epicentro.
            # Uma caixa não é foco quando contém outro retângulo verde
            # independente e comprovadamente menor. Contornos duplicados da
            # espessura da mesma linha não contam como outro nível (<15%).
            deepest = [
                candidate for candidate in inner_candidates
                if not any(
                    nested(smaller, candidate)
                    for smaller in inner_candidates
                    if smaller != candidate
                )
            ]
            inner_candidates = deepest or inner_candidates

            # O TESTE desempata epicentros do MESMO nível. Um IoU perfeito
            # com a caixa geral nunca tem precedência sobre uma ROI interna.
            confirmed = []
            for box in inner_candidates:
                matches = [
                    iou(box, tuple(int(value) for value in prior[:4]))
                    for prior in (old_epicenters or [])
                    if len(prior) >= 4
                ]
                similarity = max(matches, default=0.0)
                if similarity >= 0.6:
                    confirmed.append((similarity, box))
            if confirmed:
                confirmed.sort(key=lambda item: (-item[0], distance(item[1])))
                return confirmed[0][1], "inner_frame_confirmed_by_test"
            return min(inner_candidates, key=distance), "inner_frame_hierarchy"

        if global_frame is not None:
            # A maior caixa identificada no TESTE não pode se tornar o
            # epicentro apenas por estar próxima do centro do GABARITO.
            return None, "global_frame_only"

        return min(candidates, key=distance), "center_without_global"

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
            lower_green = RADAR_GREEN_LOWER
            upper_green = RADAR_GREEN_UPPER
            
            mask = cv2.inRange(hsv, lower_green, upper_green)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            candidate, _strategy = EpicenterExtractor.select_radar_candidate(
                [cv2.boundingRect(contour) for contour in cnts],
                sample_crop.shape,
                old_epicenters,
                global_box_info,
            )
            if candidate is not None:
                real_epicenters.append(candidate)
                
        except Exception as e:
            print(f"⚠️ Erro no Radar Euclidiano: {e}")

        # Recupera duas molduras verdes cruzadas/cortadas na borda da AOI.
        # Sem duas molduras independentes, não inventa epicentro.
        if not real_epicenters:
            recovered_focus = recover_nested_frame_focus(sample_crop)
            if recovered_focus is not None:
                real_epicenters.append(recovered_focus)

        # Fallback 1: Antigo sistema de hierarquia invertida
        if not real_epicenters:
            if old_epicenters:
                old_epicenters.sort(key=lambda b: b[2] * b[3], reverse=True)
                for (x, y, w, h) in old_epicenters:
                    oversized_width = w >= img_w * 0.90
                    oversized_height = h >= img_h * 0.90

                    # O epicentro legado já veio da hierarquia interna da AOI.
                    # Só o rejeitamos como moldura se ocupar quase toda a largura
                    # E quase toda a altura ao mesmo tempo.
                    if (
                        w > 20
                        and h > 20
                        and not (oversized_width and oversized_height)
                    ):
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