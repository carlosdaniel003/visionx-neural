# src\core\inspection.py
"""
Módulo de Inspeção de Defeitos v3.
Melhorias: Extração de Múltiplos Quadrados (Mecanismo de Atenção de Epicentros),
filtros de área mínima/máxima mais inteligentes, 
eliminação de ruído de borda, e agrupamento refinado.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.config.settings import settings


def detect_anomalies(img_gabarito: np.ndarray, img_teste: np.ndarray) -> tuple:
    """
    Retorna uma tupla: (lista_de_anomalias, lista_de_epicentros_aoi)
    """
    if img_gabarito.shape != img_teste.shape:
        h, w = img_gabarito.shape[:2]
        img_teste = cv2.resize(img_teste, (w, h))

    h_full, w_full = img_gabarito.shape[:2]

    # ==========================================
    # PASSO A: Caça ao Tesouro (Hierarquia de Caixas Verdes)
    # ==========================================
    hsv_test = cv2.cvtColor(img_teste, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv_test, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    
    # Usando RETR_LIST para capturar caixas dentro de caixas!
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    fx1, fy1, fx2, fy2 = 0, 0, w_full, h_full
    inner_boxes = []

    if contours_green:
        # Pega as caixas pelas dimensões (bounding rect) e não pela massa de pixels
        valid_greens = [cv2.boundingRect(c) for c in contours_green]
        valid_greens = [b for b in valid_greens if b[2] > 10 and b[3] > 10]
        
        # Remove duplicatas (uma linha grossa gera 2 contornos, interno e externo)
        unique_greens = []
        for b in valid_greens:
            is_dup = False
            for u in unique_greens:
                if abs(b[0]-u[0])<10 and abs(b[1]-u[1])<10 and abs(b[2]-u[2])<10 and abs(b[3]-u[3])<10:
                    is_dup = True
                    break
            if not is_dup:
                unique_greens.append(b)

        # Ordena da maior caixa para a menor
        unique_greens.sort(key=lambda b: b[2]*b[3], reverse=True)

        if unique_greens:
            # A MAIOR caixa verde define a Zona de Foco global
            gx, gy, gw, gh = unique_greens[0]
            padding = 40
            fx1 = max(0, gx - padding)
            fy1 = max(0, gy - padding)
            fx2 = min(w_full, gx + gw + padding)
            fy2 = min(h_full, gy + gh + padding)

            # As OUTRAS caixas menores são os "Epicentros" apontados pela AOI
            for (ix, iy, iw, ih) in unique_greens[1:]:
                # Só aceita se for fisicamente menor (evita falsos positivos de caixas quase do mesmo tamanho)
                if (iw * ih) < (gw * gh) * 0.85:
                    inner_boxes.append((ix, iy, iw, ih))

    # Recorta a Zona de Foco
    gab_focus = img_gabarito[fy1:fy2, fx1:fx2]
    test_focus = img_teste[fy1:fy2, fx1:fx2]

    # ==========================================
    # PASSO B: Antídoto (Invisibilidade da linha Verde + Vermelha + Azul)
    # ==========================================
    hsv_focus_test = hsv_test[fy1:fy2, fx1:fx2]
    hsv_focus_gab = cv2.cvtColor(img_gabarito[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2HSV)

    mask_green_test = cv2.inRange(hsv_focus_test, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    mask_green_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    
    mask_red1_test = cv2.inRange(hsv_focus_test, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
    mask_red2_test = cv2.inRange(hsv_focus_test, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
    mask_red_test = cv2.bitwise_or(mask_red1_test, mask_red2_test)
    
    mask_red1_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
    mask_red2_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
    mask_red_gab = cv2.bitwise_or(mask_red1_gab, mask_red2_gab)

    mask_blue_test = cv2.inRange(hsv_focus_test, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
    mask_blue_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)

    mask_ui = mask_green_test
    mask_ui = cv2.bitwise_or(mask_ui, mask_green_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_gab)

    kernel_antidote = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_ui_expanded = cv2.dilate(mask_ui, kernel_antidote, iterations=2)

    mask_ignore = cv2.bitwise_not(mask_ui_expanded)

    # ==========================================
    # PASSO C: Ignora bordas da imagem
    # ==========================================
    h_focus, w_focus = gab_focus.shape[:2]
    border_margin = 8  
    border_mask = np.zeros((h_focus, w_focus), dtype=np.uint8)
    border_mask[border_margin:h_focus-border_margin, border_margin:w_focus-border_margin] = 255
    mask_ignore = cv2.bitwise_and(mask_ignore, border_mask)

    # ==========================================
    # PASSO D: Análise Cirúrgica
    # ==========================================
    gab_blur = cv2.GaussianBlur(gab_focus, (7, 7), 0)
    test_blur = cv2.GaussianBlur(test_focus, (7, 7), 0)

    gray_gab = cv2.cvtColor(gab_blur, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_blur, cv2.COLOR_BGR2GRAY)
    _, diff_ssim_float = ssim(gray_gab, gray_test, full=True)
    diff_ssim_8bit = (diff_ssim_float * 255).astype("uint8")
    _, mask_ssim = cv2.threshold(diff_ssim_8bit, 100, 255, cv2.THRESH_BINARY_INV)

    diff_color = cv2.absdiff(gab_blur, test_blur)
    gray_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    _, mask_color = cv2.threshold(gray_color, 55, 255, cv2.THRESH_BINARY)

    fusion_mask = cv2.bitwise_and(mask_ssim, mask_color)
    fusion_mask = cv2.bitwise_and(fusion_mask, fusion_mask, mask=mask_ignore)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(fusion_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_group)

    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    focus_area = h_focus * w_focus
    anomalies = []

    for cnt in contours_final:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        if area < 150:
            continue
        if area > focus_area * 0.4:
            continue
        aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
        if aspect > 12: 
            continue
        box_area = w_box * h_box
        solidity = area / max(box_area, 1)
        if solidity < 0.15: 
            continue

        anomalies.append((x + fx1, y + fy1, w_box, h_box))

    return anomalies, inner_boxes