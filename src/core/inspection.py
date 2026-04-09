"""
Módulo de Inspeção de Defeitos (Fusão de Sensores Avançada).
Utiliza "Attention Cropping" baseado na marcação da IoT e aplica um filtro
de invisibilidade na linha verde para evitar falsos positivos.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.config.settings import settings # Importamos para usar a cor verde

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

    # Coordenadas padrão (Fallback: se não achar o verde, analisa a imagem inteira)
    fx1, fy1, fx2, fy2 = 0, 0, w_full, h_full

    if contours_green:
        # Pega a maior marcação verde na tela
        largest_green = max(contours_green, key=cv2.contourArea)
        gx, gy, gw, gh = cv2.boundingRect(largest_green)

        # Só confia se a caixa verde tiver um tamanho coerente (> 10px)
        if gw > 10 and gh > 10:
            # ==========================================
            # PASSO B: Expansão de Contexto (Zona de Foco)
            # ==========================================
            padding = 40  # Margem extra de 40 pixels ao redor do defeito para dar contexto à IA
            fx1 = max(0, gx - padding)
            fy1 = max(0, gy - padding)
            fx2 = min(w_full, gx + gw + padding)
            fy2 = min(h_full, gy + gh + padding)

    # Recorta a imagem limitando-se apenas à Zona de Foco
    gab_focus = img_gabarito[fy1:fy2, fx1:fx2]
    test_focus = img_teste[fy1:fy2, fx1:fx2]

    # ==========================================
    # PASSO C: O Antídoto (Invisibilidade da linha Verde)
    # ==========================================
    hsv_focus = hsv_test[fy1:fy2, fx1:fx2]
    mask_green_focus = cv2.inRange(hsv_focus, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
    
    # Dilatamos a linha verde levemente para cobrir a "sombra" (anti-aliasing) ao redor dos pixels verdes
    kernel_antidote = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_green_expanded = cv2.dilate(mask_green_focus, kernel_antidote, iterations=1)
    
    # Criamos a máscara de invisibilidade: Onde for PRETO (0), a matemática vai ignorar!
    mask_ignore = cv2.bitwise_not(mask_green_expanded)

    # ==========================================
    # PASSO D: Análise Cirúrgica (Apenas na Zona de Foco)
    # ==========================================
    gab_blur = cv2.GaussianBlur(gab_focus, (7, 7), 0)
    test_blur = cv2.GaussianBlur(test_focus, (7, 7), 0)

    # 1. SSIM
    gray_gab = cv2.cvtColor(gab_blur, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_blur, cv2.COLOR_BGR2GRAY)
    _, diff_ssim_float = ssim(gray_gab, gray_test, full=True)
    diff_ssim_8bit = (diff_ssim_float * 255).astype("uint8")
    _, mask_ssim = cv2.threshold(diff_ssim_8bit, 120, 255, cv2.THRESH_BINARY_INV)

    # 2. Cor
    diff_color = cv2.absdiff(gab_blur, test_blur)
    gray_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    _, mask_color = cv2.threshold(gray_color, 45, 255, cv2.THRESH_BINARY)

    # 3. Fusão dos Motores
    fusion_mask = cv2.bitwise_or(mask_ssim, mask_color)

    # 4. APLICAÇÃO DO ANTÍDOTO: Apaga da memória qualquer diferença que tenha ocorrido em cima da linha verde
    fusion_mask = cv2.bitwise_and(fusion_mask, fusion_mask, mask=mask_ignore)

    # ==========================================
    # Limpeza e Agrupamento Morfológico
    # ==========================================
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(fusion_mask, cv2.MORPH_OPEN, kernel_clean)

    kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_group)
    mask_final = cv2.dilate(mask_final, kernel_group, iterations=1)

    # ==========================================
    # Extração e Reajuste Global de Coordenadas
    # ==========================================
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anomalies = []
    for cnt in contours_final:
        area = cv2.contourArea(cnt)
        if area >= 80: # Ignora poeira isolada menor que 80 pixels quadrados
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            # Reajusta a coordenada "local" da Zona de Foco para a coordenada "Global" da imagem da IoT!
            anomalies.append((x + fx1, y + fy1, w_box, h_box))

    return anomalies