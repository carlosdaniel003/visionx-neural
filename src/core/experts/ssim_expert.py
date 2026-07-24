"""Especialista em textura e manchas usando SSIM e diferenças visuais."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class SSIMExpert:
    """Compara uma ROI de referência com a mesma ROI da imagem de teste."""

    @staticmethod
    def _safe_pair(gab: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Copia e iguala as dimensões sem alterar a imagem de referência."""
        gab_copy = gab.copy()
        test_copy = test.copy()
        if gab_copy.shape != test_copy.shape:
            test_copy = cv2.resize(
                test_copy,
                (gab_copy.shape[1], gab_copy.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return gab_copy, test_copy

    @staticmethod
    def _analysis_pair(gab: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Garante tamanho mínimo para o SSIM sem modificar os recortes exibidos."""
        height, width = gab.shape[:2]
        min_dimension = min(height, width)
        if min_dimension >= 7:
            return gab.copy(), test.copy()

        scale = 7.0 / max(min_dimension, 1)
        analysis_width = max(7, int(round(width * scale)))
        analysis_height = max(7, int(round(height * scale)))
        size = (analysis_width, analysis_height)
        return (
            cv2.resize(gab, size, interpolation=cv2.INTER_CUBIC),
            cv2.resize(test, size, interpolation=cv2.INTER_CUBIC),
        )

    def analyze(
        self,
        crop_gab: np.ndarray,
        crop_test: np.ndarray,
        full_gab: np.ndarray | None = None,
        full_test: np.ndarray | None = None,
        box_x: int = 0,
        box_y: int = 0,
        box_w: int = 0,
        box_h: int = 0,
        aoi_epicenters: list | None = None,
        canonical_focus: bool = False,
        focus_box: tuple[int, int, int, int] | None = None,
    ) -> dict:
        try:
            if (
                crop_gab is None
                or crop_test is None
                or crop_gab.size == 0
                or crop_test.size == 0
            ):
                return {
                    "is_defect": False,
                    "local_score": 0,
                    "reason": "Imagem nula",
                    "global_boxes": [],
                }

            is_epicenter = bool(canonical_focus)
            focus_source = "epicenter_extractor" if canonical_focus else "raw_anomaly"
            focus_gab = crop_gab.copy()
            focus_test = crop_test.copy()
            resolved_focus_box = focus_box

            if (
                not canonical_focus
                and aoi_epicenters
                and full_gab is not None
                and full_test is not None
            ):
                for ex, ey, ew, eh in aoi_epicenters:
                    x_right = min(box_x + box_w, ex + ew)
                    x_left = max(box_x, ex)
                    y_bottom = min(box_y + box_h, ey + eh)
                    y_top = max(box_y, ey)

                    if x_right <= x_left or y_bottom <= y_top:
                        continue

                    height, width = full_gab.shape[:2]
                    x1, x2 = max(0, ex), min(width, ex + ew)
                    y1, y2 = max(0, ey), min(height, ey + eh)
                    if x2 > x1 and y2 > y1:
                        focus_gab = full_gab[y1:y2, x1:x2].copy()
                        focus_test = full_test[y1:y2, x1:x2].copy()
                        resolved_focus_box = (x1, y1, x2 - x1, y2 - y1)
                        is_epicenter = True
                        focus_source = "aoi_intersection"
                    break

            focus_gab, focus_test = self._safe_pair(focus_gab, focus_test)
            display_gab = focus_gab.copy()
            display_test = focus_test.copy()

            analysis_gab, analysis_test = self._analysis_pair(display_gab, display_test)
            gab = cv2.GaussianBlur(analysis_gab, (3, 3), 0)
            test = cv2.GaussianBlur(analysis_test, (3, 3), 0)
            gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

            ssim_score, ssim_map = ssim(gray_gab, gray_test, full=True)
            diff = cv2.absdiff(gray_gab, gray_test).astype(np.float32) / 255.0

            mean_diff = float(np.mean(diff))
            pct_changed = float(np.mean(diff > 0.15))
            edge_change = float(
                np.mean(
                    cv2.absdiff(
                        cv2.Canny(gray_gab, 70, 180),
                        cv2.Canny(gray_test, 70, 180),
                    )
                    > 0
                )
            )
            hist_corr = float(
                cv2.compareHist(
                    cv2.calcHist([gray_gab], [0], None, [64], [0, 256]),
                    cv2.calcHist([gray_test], [0], None, [64], [0, 256]),
                    cv2.HISTCMP_CORREL,
                )
            )

            structural_heat = np.clip((1.0 - ssim_map) * 255.0, 0, 255)
            pixel_heat = np.clip(diff * 255.0, 0, 255)
            heat_map = np.maximum(structural_heat, pixel_heat).astype(np.uint8)
            display_height, display_width = display_gab.shape[:2]
            if heat_map.shape != (display_height, display_width):
                heat_map = cv2.resize(
                    heat_map,
                    (display_width, display_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            ctx_score, ctx_reason = 0.3, ""
            if full_gab is not None and full_test is not None and box_w > 0:
                full_height, full_width = full_gab.shape[:2]
                expand = max(box_w, box_h)
                ctx_g = full_gab[
                    max(0, box_y - expand) : min(full_height, box_y + box_h + expand),
                    max(0, box_x - expand) : min(full_width, box_x + box_w + expand),
                ]
                ctx_t = full_test[
                    max(0, box_y - expand) : min(full_height, box_y + box_h + expand),
                    max(0, box_x - expand) : min(full_width, box_x + box_w + expand),
                ]

                if ctx_g.size > 0 and ctx_t.size > 0:
                    if ctx_g.shape != ctx_t.shape:
                        ctx_t = cv2.resize(
                            ctx_t,
                            (ctx_g.shape[1], ctx_g.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )

                    _, context_map = ssim(
                        cv2.cvtColor(cv2.resize(ctx_g, (96, 96)), cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(cv2.resize(ctx_t, (96, 96)), cv2.COLOR_BGR2GRAY),
                        full=True,
                    )
                    _, threshold = cv2.threshold(
                        ((1.0 - context_map) * 255).astype(np.uint8),
                        100,
                        255,
                        cv2.THRESH_BINARY,
                    )
                    contours, _ = cv2.findContours(
                        threshold,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    is_localized = len(contours) <= 3
                    ctx_score = 0.6 if is_localized else 0.20
                    base_reason = "concentrada" if is_localized else "espalhada"
                    if is_epicenter:
                        ctx_score = min(1.0, ctx_score + 0.20)
                        ctx_reason = f"Foco Validado | Diferença {base_reason}"
                    else:
                        ctx_reason = f"Diferença {base_reason}"

            global_boxes = []
            diff_edges = None
            if (
                full_gab is not None
                and full_test is not None
                and full_gab.shape == full_test.shape
            ):
                full_gray_gab = cv2.cvtColor(full_gab, cv2.COLOR_BGR2GRAY)
                full_gray_test = cv2.cvtColor(full_test, cv2.COLOR_BGR2GRAY)
                blur_gab = cv2.GaussianBlur(full_gray_gab, (7, 7), 0)
                blur_test = cv2.GaussianBlur(full_gray_test, (7, 7), 0)
                edges_gab = cv2.Canny(blur_gab, 50, 150)
                edges_test = cv2.Canny(blur_test, 50, 150)
                tolerance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                edges_gab_dilated = cv2.dilate(edges_gab, tolerance_kernel, iterations=1)
                diff_edges = cv2.bitwise_xor(edges_test, edges_gab_dilated)
                macro_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                macro_threshold = cv2.morphologyEx(
                    diff_edges,
                    cv2.MORPH_CLOSE,
                    macro_kernel,
                )
                macro_threshold = cv2.morphologyEx(
                    macro_threshold,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                )
                macro_contours, _ = cv2.findContours(
                    macro_threshold,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in macro_contours:
                    if cv2.contourArea(contour) > 600:
                        global_boxes.append(cv2.boundingRect(contour))

            local_score = sum(
                [
                    max(0, (0.85 - ssim_score) / 0.85) * 0.35,
                    min(1.0, mean_diff / 0.25) * 0.20,
                    min(1.0, pct_changed / 0.40) * 0.20,
                    min(1.0, edge_change / 0.25) * 0.15,
                    max(0, (0.80 - hist_corr) / 0.80) * 0.10,
                ]
            )
            local_score = max(0.0, min(1.0, local_score))
            if is_epicenter:
                local_score = min(1.0, local_score * 1.30)

            if global_boxes:
                local_score = max(local_score, 0.95)
                ctx_reason = "DANO ESTRUTURAL MASSIVO DETECTADO"

            return {
                "local_score": float(local_score),
                "ctx_score": float(ctx_score),
                "ssim": float(ssim_score),
                "mean_diff": float(mean_diff),
                "pct_changed": float(pct_changed),
                "edge_change": float(edge_change),
                "hist_corr": float(hist_corr),
                "ctx_reason": ctx_reason,
                "is_epicenter": is_epicenter,
                "global_boxes": global_boxes,
                "heat_map_raw": heat_map,
                "macro_edges": diff_edges,
                "crop_gab": display_gab,
                "crop_test": display_test,
                "full_test": full_test,
                "focus_source": focus_source,
                "focus_box": resolved_focus_box,
                "focus_width": int(display_width),
                "focus_height": int(display_height),
            }
        except Exception as exc:
            print(f"⚠️ Erro no SSIMExpert: {exc}")
            return {
                "local_score": 0,
                "ctx_score": 0,
                "ssim": 1,
                "mean_diff": 0,
                "pct_changed": 0,
                "edge_change": 0,
                "hist_corr": 1,
                "ctx_reason": "",
                "is_epicenter": False,
                "global_boxes": [],
            }
