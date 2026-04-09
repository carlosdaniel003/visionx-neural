"""
Módulo de Inspeção de Defeitos v2.
Melhorias: filtros de área mínima/máxima mais inteligentes, 
eliminação de ruído de borda, e agrupamento refinado.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.config.settings import settings


def detect_anomalies(img_gabarito: np.ndarray, img_teste: np.ndarray) -> list:
    if img_gabarito.shape != img_teste.shape:
        h, w = img_gabarito.shape[:2]
        img_teste = cv2.resize(img_teste, (w, h))

    h_full, w_full = img_gabarito.shape[:2]

    # ==========================================
    # PASSO A: Caça ao Tesouro (Onde a IoT apontou?)
    # ==========================================
    hsv_test = cv2.cvtColor(img_teste, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv_test, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fx1, fy1, fx2, fy2 = 0, 0, w_full, h_full

    if contours_green:
        largest_green = max(contours_green, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest_green)

        if gw > 10 and gh > 10:
            padding = 40
            fx1 = max(0, gx - padding)
            fy1 = max(0, gy - padding)
            fx2 = min(w_full, gx + gw + padding)
            fy2 = min(h_full, gy + gh + padding)

    # Recorta a Zona de Foco
    gab_focus = img_gabarito[fy1:fy2, fx1:fx2]
    test_focus = img_teste[fy1:fy2, fx1:fx2]

    # ==========================================
    # PASSO B: Antídoto (Invisibilidade da linha Verde + Vermelha + Azul)
    # ==========================================
    hsv_focus_test = hsv_test[fy1:fy2, fx1:fx2]
    hsv_focus_gab = cv2.cvtColor(img_gabarito[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2HSV)

    # Mascara verde em ambas as imagens (a IoT pode ter marcações diferentes)
    mask_green_test = cv2.inRange(hsv_focus_test, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    mask_green_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    
    # Mascara vermelho (barras/indicadores da interface)
    mask_red1_test = cv2.inRange(hsv_focus_test, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
    mask_red2_test = cv2.inRange(hsv_focus_test, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
    mask_red_test = cv2.bitwise_or(mask_red1_test, mask_red2_test)
    
    mask_red1_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER)
    mask_red2_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER)
    mask_red_gab = cv2.bitwise_or(mask_red1_gab, mask_red2_gab)

    # Mascara azul (barras da interface)
    mask_blue_test = cv2.inRange(hsv_focus_test, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)
    mask_blue_gab = cv2.inRange(hsv_focus_gab, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER)

    # Combina TODAS as máscaras de interface (verde + vermelho + azul de ambas imagens)
    mask_ui = mask_green_test
    mask_ui = cv2.bitwise_or(mask_ui, mask_green_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_gab)

    # Dilata para cobrir anti-aliasing ao redor dos pixels coloridos da UI
    kernel_antidote = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_ui_expanded = cv2.dilate(mask_ui, kernel_antidote, iterations=2)

    # Máscara de invisibilidade final
    mask_ignore = cv2.bitwise_not(mask_ui_expanded)

    # ==========================================
    # PASSO C: Ignora bordas da imagem (margem de segurança)
    # ==========================================
    h_focus, w_focus = gab_focus.shape[:2]
    border_margin = 8  # pixels
    border_mask = np.zeros((h_focus, w_focus), dtype=np.uint8)
    border_mask[border_margin:h_focus-border_margin, border_margin:w_focus-border_margin] = 255
    mask_ignore = cv2.bitwise_and(mask_ignore, border_mask)

    # ==========================================
    # PASSO D: Análise Cirúrgica
    # ==========================================
    gab_blur = cv2.GaussianBlur(gab_focus, (7, 7), 0)
    test_blur = cv2.GaussianBlur(test_focus, (7, 7), 0)

    # 1. SSIM
    gray_gab = cv2.cvtColor(gab_blur, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_blur, cv2.COLOR_BGR2GRAY)
    _, diff_ssim_float = ssim(gray_gab, gray_test, full=True)
    diff_ssim_8bit = (diff_ssim_float * 255).astype("uint8")
    _, mask_ssim = cv2.threshold(diff_ssim_8bit, 100, 255, cv2.THRESH_BINARY_INV)

    # 2. Diferença de Cor (threshold mais alto = menos ruído)
    diff_color = cv2.absdiff(gab_blur, test_blur)
    gray_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    _, mask_color = cv2.threshold(gray_color, 55, 255, cv2.THRESH_BINARY)

    # 3. Fusão: exige que AMBOS os motores concordem (AND em vez de OR = menos falsos positivos)
    fusion_mask = cv2.bitwise_and(mask_ssim, mask_color)

    # 4. Aplica o antídoto
    fusion_mask = cv2.bitwise_and(fusion_mask, fusion_mask, mask=mask_ignore)

    # ==========================================
    # Limpeza Morfológica Refinada
    # ==========================================
    # Abre para remover ruído fino (pontos soltos)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(fusion_mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    # Fecha para agrupar regiões próximas
    kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_group)

    # ==========================================
    # Extração de Contornos com Filtros Inteligentes
    # ==========================================
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    focus_area = h_focus * w_focus
    anomalies = []

    for cnt in contours_final:
        area = cv2.contourArea(cnt)
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        # Filtro 1: Área mínima (ignora poeira/ruído) — mais restritivo
        if area < 150:
            continue

        # Filtro 2: Área máxima (se cobre >40% da zona de foco, é erro de alinhamento, não defeito)
        if area > focus_area * 0.4:
            continue

        # Filtro 3: Proporção (aspect ratio) — rejeita linhas finíssimas (1px de largura)
        aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
        if aspect > 12:  # Linhas muito finas (ex: borda residual)
            continue

        # Filtro 4: Solidez (área real vs bounding box) — rejeita contornos espalhados/dispersos
        box_area = w_box * h_box
        solidity = area / max(box_area, 1)
        if solidity < 0.15:  # Contorno muito esparso
            continue

        # Reajusta coordenada local → global
        anomalies.append((x + fx1, y + fy1, w_box, h_box))

    return anomalies