"""Especialista em textura, SSIM e mapa seletivo de excesso de adesivo."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class SSIMExpert:
    """Compara exatamente a mesma ROI do gabarito e da imagem de teste."""

    @staticmethod
    def _safe_pair(gab: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        """Amplia somente ROIs menores que o limite mínimo do SSIM."""
        height, width = gab.shape[:2]
        minimum = min(height, width)
        if minimum >= 7:
            return gab.copy(), test.copy()

        scale = 7.0 / max(minimum, 1)
        size = (
            max(7, int(round(width * scale))),
            max(7, int(round(height * scale))),
        )
        return (
            cv2.resize(gab, size, interpolation=cv2.INTER_CUBIC),
            cv2.resize(test, size, interpolation=cv2.INTER_CUBIC),
        )

    @staticmethod
    def _is_adhesive_category(category: str) -> bool:
        normalized = str(category or "").strip().upper()
        return any(
            token in normalized
            for token in ("MUCH ADHESIVE", "ADHESIVE", "ADESIVO", "COLA")
        )

    @staticmethod
    def _warm_hue_affinity(hue: np.ndarray) -> np.ndarray:
        """Afinidade contínua para vermelho, laranja queimado e marrom."""
        hue = hue.astype(np.float32)
        orange_red = np.clip(1.0 - np.abs(hue - 10.0) / 22.0, 0.0, 1.0)
        wrapped_red = np.clip(
            1.0 - np.minimum(np.abs(hue - 179.0), np.abs(hue + 1.0)) / 12.0,
            0.0,
            1.0,
        )
        return np.maximum(orange_red, wrapped_red)

    @classmethod
    def _adhesive_likelihood(cls, image_bgr: np.ndarray) -> np.ndarray:
        """Probabilidade cromática de cola marrom/vermelha escura."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        blue, green, red = cv2.split(image_bgr.astype(np.float32))
        hue, saturation, value = cv2.split(hsv)

        warm = cls._warm_hue_affinity(hue)
        saturation_weight = 0.25 + 0.75 * np.clip(
            (saturation - 25.0) / 150.0,
            0.0,
            1.0,
        )
        darkness = np.clip((205.0 - value) / 170.0, 0.0, 1.0)
        red_dominance = np.clip(
            (red - np.maximum(green, blue) + 10.0) / 100.0,
            0.0,
            1.0,
        )
        lab_red = np.clip((lab[:, :, 1] - 128.0) / 55.0, 0.0, 1.0)

        return np.clip(
            warm
            * saturation_weight
            * (0.25 + 0.75 * darkness)
            * (0.45 + 0.30 * red_dominance + 0.25 * lab_red),
            0.0,
            1.0,
        )

    @staticmethod
    def _copper_likelihood(image_bgr: np.ndarray) -> np.ndarray:
        """Probabilidade de padding/cobre laranja brilhante."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hue, saturation, value = cv2.split(hsv)

        orange = np.clip(1.0 - np.abs(hue - 13.0) / 18.0, 0.0, 1.0)
        saturation_weight = np.clip((saturation - 55.0) / 140.0, 0.0, 1.0)
        brightness = np.clip((value - 105.0) / 135.0, 0.0, 1.0)
        lab_yellow = np.clip((lab[:, :, 2] - 132.0) / 65.0, 0.0, 1.0)

        return np.clip(
            orange
            * saturation_weight
            * brightness
            * (0.55 + 0.45 * lab_yellow),
            0.0,
            1.0,
        )

    @staticmethod
    def _metal_likelihood(image_bgr: np.ndarray) -> np.ndarray:
        """Probabilidade de lateral metálica, solda ou região prata/branca."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        blue, green, red = cv2.split(image_bgr.astype(np.float32))
        _, saturation, value = cv2.split(hsv)

        spread = np.maximum.reduce([blue, green, red]) - np.minimum.reduce(
            [blue, green, red]
        )
        neutral = np.clip(1.0 - spread / 60.0, 0.0, 1.0)
        low_saturation = np.clip(1.0 - saturation / 80.0, 0.0, 1.0)
        brightness = np.clip((value - 95.0) / 150.0, 0.0, 1.0)
        return np.clip(neutral * low_saturation * brightness, 0.0, 1.0)

    @staticmethod
    def _align_reference_to_test(
        gab: np.ndarray,
        test: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        """Alinha o gabarito ao teste para eliminar calor causado por translação."""
        height, width = gab.shape[:2]
        if min(height, width) < 16:
            return gab.copy(), (0.0, 0.0), 0.0

        template = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        template = cv2.GaussianBlur(template, (5, 5), 0)
        moving = cv2.GaussianBlur(moving, (5, 5), 0)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            60,
            1e-5,
        )

        try:
            correlation, warp = cv2.findTransformECC(
                template,
                moving,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
                None,
                3,
            )
            dx = float(warp[0, 2])
            dy = float(warp[1, 2])
            if abs(dx) > width * 0.20 or abs(dy) > height * 0.20:
                return gab.copy(), (0.0, 0.0), 0.0

            aligned = cv2.warpAffine(
                gab,
                warp,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned, (dx, dy), float(correlation)
        except cv2.error:
            return gab.copy(), (0.0, 0.0), 0.0

    @classmethod
    def _build_adhesive_heatmap(
        cls,
        aligned_gab: np.ndarray,
        test: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Reconstrói o excesso de adesivo em coordenadas da imagem de teste.

        Favorece material marrom/vermelho escuro que apareceu no teste,
        principalmente quando encobre cobre do gabarito. Cobre estável,
        metal estável, corpo estável e bordas coincidentes são suprimidos.
        """
        hsv_gab = cv2.cvtColor(aligned_gab, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_test = cv2.cvtColor(test, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab_gab = cv2.cvtColor(aligned_gab, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_test = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)

        adhesive_gab = cls._adhesive_likelihood(aligned_gab)
        adhesive_test = cls._adhesive_likelihood(test)
        copper_gab = cls._copper_likelihood(aligned_gab)
        copper_test = cls._copper_likelihood(test)
        metal_gab = cls._metal_likelihood(aligned_gab)
        metal_test = cls._metal_likelihood(test)

        delta_lab = np.linalg.norm(lab_test - lab_gab, axis=2) / 110.0
        changed = np.clip((delta_lab - 0.06) / 0.75, 0.0, 1.0)
        dark_gain = np.clip(
            (hsv_gab[:, :, 2] - hsv_test[:, :, 2] - 4.0) / 105.0,
            0.0,
            1.0,
        )
        red_gain = np.clip(
            (lab_test[:, :, 1] - lab_gab[:, :, 1] - 2.0) / 45.0,
            0.0,
            1.0,
        )

        adhesive_excess = np.clip(
            adhesive_test - adhesive_gab - 0.05,
            0.0,
            1.0,
        )
        covered_copper = (
            copper_gab
            * (1.0 - copper_test)
            * adhesive_test
            * np.maximum(dark_gain, changed)
        )
        dark_adhesive_change = (
            adhesive_test
            * changed
            * np.maximum(dark_gain, 0.40 * red_gain)
        )
        raw_heat = (
            0.48 * adhesive_excess
            + 0.37 * covered_copper
            + 0.15 * dark_adhesive_change
        )

        stable_copper = np.minimum(copper_gab, copper_test)
        stable_metal = np.minimum(metal_gab, metal_test)
        gray_gab = cv2.cvtColor(aligned_gab, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_gab = cv2.dilate(cv2.Canny(gray_gab, 60, 160), edge_kernel)
        edges_test = cv2.dilate(cv2.Canny(gray_test, 60, 160), edge_kernel)
        stable_edges = np.minimum(
            edges_gab.astype(np.float32) / 255.0,
            edges_test.astype(np.float32) / 255.0,
        )

        raw_heat *= 1.0 - 0.82 * stable_copper
        raw_heat *= 1.0 - 0.75 * stable_metal
        raw_heat *= 1.0 - 0.35 * stable_edges
        raw_heat *= 0.20 + 0.80 * changed

        height, _ = raw_heat.shape
        vertical_position = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        raw_heat *= 0.68 + 0.52 * vertical_position
        raw_heat = cv2.GaussianBlur(np.clip(raw_heat, 0.0, 1.0), (5, 5), 0)

        active = (raw_heat > 0.045).astype(np.uint8) * 255
        active = cv2.morphologyEx(
            active,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        active = cv2.morphologyEx(
            active,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        raw_heat *= active.astype(np.float32) / 255.0

        positive = raw_heat[raw_heat > 0.01]
        if positive.size == 0:
            return np.zeros(raw_heat.shape, dtype=np.uint8), {
                "adhesive_peak": 0.0,
                "adhesive_coverage": 0.0,
                "adhesive_evidence": 0.0,
                "adhesive_centroid": None,
            }

        low = float(np.percentile(positive, 25))
        high = float(np.percentile(positive, 98))
        normalized = np.clip(
            (raw_heat - low) / max(high - low, 1e-6),
            0.0,
            1.0,
        )
        normalized[normalized < 0.06] = 0.0
        heat_map = (normalized * 255.0).astype(np.uint8)

        strong_mask = heat_map >= 96
        coverage = float(np.mean(strong_mask))
        weights = heat_map.astype(np.float32)
        total_weight = float(weights.sum())
        centroid = None
        if total_weight > 0:
            y_coordinates, x_coordinates = np.indices(heat_map.shape)
            centroid = (
                float((x_coordinates * weights).sum() / total_weight),
                float((y_coordinates * weights).sum() / total_weight),
            )

        return heat_map, {
            "adhesive_peak": float(heat_map.max() / 255.0),
            "adhesive_coverage": coverage,
            "adhesive_evidence": high,
            "adhesive_centroid": centroid,
        }

    @staticmethod
    def _build_generic_heatmap(
        ssim_map: np.ndarray,
        pixel_diff: np.ndarray,
        display_size: tuple[int, int],
    ) -> np.ndarray:
        structural_heat = np.clip((1.0 - ssim_map) * 255.0, 0.0, 255.0)
        pixel_heat = np.clip(pixel_diff * 255.0, 0.0, 255.0)
        heat_map = np.maximum(structural_heat, pixel_heat).astype(np.uint8)
        display_width, display_height = display_size
        if heat_map.shape != (display_height, display_width):
            heat_map = cv2.resize(
                heat_map,
                (display_width, display_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return heat_map

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
        defect_category: str = "",
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

            category_requests_adhesive = self._is_adhesive_category(defect_category)
            adhesive_reference, alignment_shift, alignment_score = (
                self._align_reference_to_test(display_gab, display_test)
            )
            adhesive_heat_map, adhesive_metrics = self._build_adhesive_heatmap(
                adhesive_reference,
                display_test,
            )
            adhesive_mode = category_requests_adhesive or (
                adhesive_metrics["adhesive_evidence"] >= 0.05
                and adhesive_metrics["adhesive_coverage"] >= 0.001
            )

            analysis_reference = (
                adhesive_reference if category_requests_adhesive else display_gab
            )
            analysis_gab, analysis_test = self._analysis_pair(
                analysis_reference,
                display_test,
            )
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

            display_height, display_width = display_gab.shape[:2]
            if adhesive_mode:
                heat_map = adhesive_heat_map
                heat_map_mode = "adhesive_excess"
            else:
                heat_map = self._build_generic_heatmap(
                    ssim_map,
                    diff,
                    (display_width, display_height),
                )
                heat_map_mode = "generic_ssim"

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
                    localized = len(contours) <= 3
                    ctx_score = 0.6 if localized else 0.20
                    base_reason = "concentrada" if localized else "espalhada"
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
                tolerance_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (5, 5),
                )
                edges_gab_dilated = cv2.dilate(
                    edges_gab,
                    tolerance_kernel,
                    iterations=1,
                )
                diff_edges = cv2.bitwise_xor(edges_test, edges_gab_dilated)
                macro_threshold = cv2.morphologyEx(
                    diff_edges,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)),
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
                "heat_map_mode": heat_map_mode,
                "macro_edges": diff_edges,
                "crop_gab": display_gab,
                "crop_test": display_test,
                "full_test": full_test,
                "focus_source": focus_source,
                "focus_box": resolved_focus_box,
                "focus_width": int(display_width),
                "focus_height": int(display_height),
                "alignment_shift": alignment_shift,
                "alignment_score": float(alignment_score),
                **adhesive_metrics,
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
