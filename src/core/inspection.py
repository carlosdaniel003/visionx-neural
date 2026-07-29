# src/core/inspection.py
"""Detecção inicial de diferenças e caixas de interface da AOI."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.config.settings import settings
from src.core.epicenter_extractor import EpicenterExtractor


def detect_anomalies(img_gabarito: np.ndarray, img_teste: np.ndarray) -> tuple:
    """Detecta diferenças e retorna caixas da AOI em coordenadas absolutas."""
    if img_gabarito.shape != img_teste.shape:
        height, width = img_gabarito.shape[:2]
        img_teste = cv2.resize(img_teste, (width, height))

    full_height, full_width = img_gabarito.shape[:2]

    # ---------------------------------------------------------------
    # Passo A: caixas verdes usando a assinatura da sobreposição AOI
    # ---------------------------------------------------------------
    mask_green_gabarito = EpicenterExtractor._green_mask(img_gabarito)
    mask_green_teste = EpicenterExtractor._green_mask(img_teste)
    mask_green = cv2.bitwise_or(mask_green_gabarito, mask_green_teste)
    contours_green, _ = cv2.findContours(
        mask_green,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    focus_x1, focus_y1, focus_x2, focus_y2 = 0, 0, full_width, full_height
    inner_boxes = []
    global_box_info = {
        "x": 0,
        "y": 0,
        "w": full_width,
        "h": full_height,
    }

    if contours_green:
        valid_greens = [cv2.boundingRect(contour) for contour in contours_green]
        valid_greens = [
            box
            for box in valid_greens
            if box[2] > 10 and box[3] > 10
        ]

        unique_greens = []
        for box in valid_greens:
            duplicate = any(
                abs(box[0] - current[0]) < 10
                and abs(box[1] - current[1]) < 10
                and abs(box[2] - current[2]) < 10
                and abs(box[3] - current[3]) < 10
                for current in unique_greens
            )
            if not duplicate:
                unique_greens.append(box)

        unique_greens.sort(key=lambda box: box[2] * box[3], reverse=True)
        if unique_greens:
            global_x, global_y, global_width, global_height = unique_greens[0]
            global_box_info = {
                "x": int(global_x),
                "y": int(global_y),
                "w": int(global_width),
                "h": int(global_height),
            }

            padding = 40
            focus_x1 = max(0, global_x - padding)
            focus_y1 = max(0, global_y - padding)
            focus_x2 = min(full_width, global_x + global_width + padding)
            focus_y2 = min(full_height, global_y + global_height + padding)

            global_area = global_width * global_height
            for inner_x, inner_y, inner_width, inner_height in unique_greens[1:]:
                inner_area = inner_width * inner_height
                is_smaller = inner_area < global_area * 0.85
                is_contained = (
                    inner_x >= global_x - 20
                    and inner_y >= global_y - 20
                    and inner_x + inner_width <= global_x + global_width + 20
                    and inner_y + inner_height <= global_y + global_height + 20
                )
                if is_smaller and is_contained:
                    inner_boxes.append(
                        (
                            int(inner_x),
                            int(inner_y),
                            int(inner_width),
                            int(inner_height),
                        )
                    )

            inner_boxes.sort(key=lambda box: box[2] * box[3])

    gab_focus = img_gabarito[focus_y1:focus_y2, focus_x1:focus_x2]
    test_focus = img_teste[focus_y1:focus_y2, focus_x1:focus_x2]

    # ---------------------------------------------------------------
    # Passo B: invisibilidade das linhas de interface
    # ---------------------------------------------------------------
    hsv_focus_test = cv2.cvtColor(test_focus, cv2.COLOR_BGR2HSV)
    hsv_focus_gab = cv2.cvtColor(gab_focus, cv2.COLOR_BGR2HSV)

    mask_green_test = EpicenterExtractor._green_mask(test_focus)
    mask_green_gab = EpicenterExtractor._green_mask(gab_focus)

    mask_red1_test = cv2.inRange(
        hsv_focus_test,
        settings.COLOR_RED1_LOWER,
        settings.COLOR_RED1_UPPER,
    )
    mask_red2_test = cv2.inRange(
        hsv_focus_test,
        settings.COLOR_RED2_LOWER,
        settings.COLOR_RED2_UPPER,
    )
    mask_red_test = cv2.bitwise_or(mask_red1_test, mask_red2_test)

    mask_red1_gab = cv2.inRange(
        hsv_focus_gab,
        settings.COLOR_RED1_LOWER,
        settings.COLOR_RED1_UPPER,
    )
    mask_red2_gab = cv2.inRange(
        hsv_focus_gab,
        settings.COLOR_RED2_LOWER,
        settings.COLOR_RED2_UPPER,
    )
    mask_red_gab = cv2.bitwise_or(mask_red1_gab, mask_red2_gab)

    mask_blue_test = cv2.inRange(
        hsv_focus_test,
        settings.COLOR_BLUE_LOWER,
        settings.COLOR_BLUE_UPPER,
    )
    mask_blue_gab = cv2.inRange(
        hsv_focus_gab,
        settings.COLOR_BLUE_LOWER,
        settings.COLOR_BLUE_UPPER,
    )

    mask_ui = cv2.bitwise_or(mask_green_test, mask_green_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_red_gab)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_test)
    mask_ui = cv2.bitwise_or(mask_ui, mask_blue_gab)

    kernel_antidote = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_ui_expanded = cv2.dilate(mask_ui, kernel_antidote, iterations=2)
    mask_ignore = cv2.bitwise_not(mask_ui_expanded)

    # ---------------------------------------------------------------
    # Passo C: ignora bordas do foco
    # ---------------------------------------------------------------
    focus_height, focus_width = gab_focus.shape[:2]
    border_margin = 8
    border_mask = np.zeros((focus_height, focus_width), dtype=np.uint8)
    if focus_height > border_margin * 2 and focus_width > border_margin * 2:
        border_mask[
            border_margin : focus_height - border_margin,
            border_margin : focus_width - border_margin,
        ] = 255
    mask_ignore = cv2.bitwise_and(mask_ignore, border_mask)

    # ---------------------------------------------------------------
    # Passo D: análise de diferenças
    # ---------------------------------------------------------------
    gab_blur = cv2.GaussianBlur(gab_focus, (3, 3), 0)
    test_blur = cv2.GaussianBlur(test_focus, (3, 3), 0)

    gray_gab = cv2.cvtColor(gab_blur, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_blur, cv2.COLOR_BGR2GRAY)
    _, diff_ssim_float = ssim(gray_gab, gray_test, full=True)
    diff_ssim_8bit = (diff_ssim_float * 255).astype("uint8")
    _, mask_ssim = cv2.threshold(
        diff_ssim_8bit,
        100,
        255,
        cv2.THRESH_BINARY_INV,
    )

    diff_color = cv2.absdiff(gab_blur, test_blur)
    gray_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    _, mask_color = cv2.threshold(gray_color, 80, 255, cv2.THRESH_BINARY)

    fusion_mask = cv2.bitwise_and(mask_ssim, mask_color)
    fusion_mask = cv2.bitwise_and(fusion_mask, fusion_mask, mask=mask_ignore)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(
        fusion_mask,
        cv2.MORPH_OPEN,
        kernel_clean,
        iterations=2,
    )
    kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_group)

    contours_final, _ = cv2.findContours(
        mask_final,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    focus_area = max(1, focus_height * focus_width)
    anomalies = []
    for contour in contours_final:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area < 40:
            continue
        if area > focus_area * 0.4:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 25:
            continue
        box_area = width * height
        solidity = area / max(box_area, 1)
        if solidity < 0.05:
            continue
        anomalies.append(
            (
                x + focus_x1,
                y + focus_y1,
                width,
                height,
            )
        )

    return anomalies, inner_boxes, global_box_info, gab_focus, test_focus
