"""Comparador estrutural do epicentro usando alinhamento e XOR de contornos."""

from __future__ import annotations

import cv2
import numpy as np


class SilkExpert:
    """
    Compara a estrutura visual da mesma ROI no gabarito e no teste.

    O motor não tenta interpretar cor ou material. Ele extrai contornos, alinha
    as duas imagens e separa estrutura coincidente, extra e ausente.
    """

    @staticmethod
    def _empty_result(reason: str = "") -> dict:
        return {
            "is_defect": False,
            "silk_error_pct": 0.0,
            "tolerance": 0.08,
            "pct_changed": 0.0,
            "extra_pct": 0.0,
            "missing_pct": 0.0,
            "matched_pct": 1.0,
            "dx": 0.0,
            "dy": 0.0,
            "alignment_score": 0.0,
            "reason": reason,
            "bounding_box": None,
        }

    @staticmethod
    def _safe_pair(
        full_gab: np.ndarray,
        full_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        gab = full_gab.copy()
        test = full_test.copy()
        if gab.shape != test.shape:
            test = cv2.resize(
                test,
                (gab.shape[1], gab.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return gab, test

    @staticmethod
    def _extract_roi(
        full_gab: np.ndarray,
        full_test: np.ndarray,
        aoi_epicenters: list | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
        height, width = full_gab.shape[:2]
        x1, y1, x2, y2 = 0, 0, width, height

        if aoi_epicenters:
            ex, ey, ew, eh = aoi_epicenters[0]
            candidate_x1 = max(0, int(ex))
            candidate_y1 = max(0, int(ey))
            candidate_x2 = min(width, int(ex + ew))
            candidate_y2 = min(height, int(ey + eh))
            if candidate_x2 > candidate_x1 and candidate_y2 > candidate_y1:
                x1, y1, x2, y2 = (
                    candidate_x1,
                    candidate_y1,
                    candidate_x2,
                    candidate_y2,
                )
        else:
            crop_x = int(width * 0.22)
            crop_y = int(height * 0.18)
            if crop_x < width // 2 and crop_y < height // 2:
                x1, y1 = crop_x, crop_y
                x2, y2 = width - crop_x, height - crop_y

        roi_gab = full_gab[y1:y2, x1:x2].copy()
        roi_test = full_test[y1:y2, x1:x2].copy()
        return roi_gab, roi_test, (x1, y1), (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _align_test_to_reference(
        roi_gab: np.ndarray,
        roi_test: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        """Alinha o teste ao gabarito por translação, sem mudar a ROI."""
        height, width = roi_gab.shape[:2]
        if min(height, width) < 16:
            return roi_test.copy(), (0.0, 0.0), 0.0

        template = cv2.cvtColor(roi_gab, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        template = cv2.GaussianBlur(template, (5, 5), 0)
        moving = cv2.GaussianBlur(moving, (5, 5), 0)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            80,
            1e-6,
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
                return roi_test.copy(), (0.0, 0.0), 0.0

            aligned = cv2.warpAffine(
                roi_test,
                warp,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned, (dx, dy), float(correlation)
        except cv2.error:
            return roi_test.copy(), (0.0, 0.0), 0.0

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    @classmethod
    def _extract_structure(cls, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extrai contornos estruturais, evitando máscaras preenchidas que faziam
        o XOR ocupar praticamente toda a ROI.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(gray))
        low = int(max(20, min(130, median * 0.66)))
        high = int(max(low + 25, min(230, median * 1.33)))

        edges = cv2.Canny(gray, low, high)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        min_area = max(2, int(edges.size * 0.0004))
        return cls._remove_small_components(edges, min_area)

    @staticmethod
    def _paint_mask(
        image_bgr: np.ndarray,
        mask: np.ndarray,
        color_bgr: tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        output = image_bgr.copy().astype(np.float32)
        selected = mask > 0
        if np.any(selected):
            color = np.asarray(color_bgr, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + color * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @classmethod
    def _build_visualizations(
        cls,
        roi_gab: np.ndarray,
        roi_test_aligned: np.ndarray,
        mask_gab: np.ndarray,
        mask_test: np.ndarray,
        match_mask: np.ndarray,
        extra_mask: np.ndarray,
        missing_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference_view = cls._paint_mask(
            roi_gab,
            cv2.dilate(mask_gab, np.ones((2, 2), np.uint8)),
            (0, 210, 255),
            0.78,
        )
        test_view = cls._paint_mask(
            roi_test_aligned,
            cv2.dilate(mask_test, np.ones((2, 2), np.uint8)),
            (255, 210, 40),
            0.78,
        )

        difference_view = (roi_test_aligned.astype(np.float32) * 0.72).astype(np.uint8)
        visible_match = cv2.dilate(
            match_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        visible_extra = cv2.dilate(
            extra_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        visible_missing = cv2.dilate(
            missing_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        difference_view = cls._paint_mask(
            difference_view,
            visible_match,
            (70, 190, 90),
            0.38,
        )
        difference_view = cls._paint_mask(
            difference_view,
            visible_missing,
            (0, 220, 255),
            0.92,
        )
        difference_view = cls._paint_mask(
            difference_view,
            visible_extra,
            (45, 65, 255),
            0.94,
        )

        for mask, color in (
            (visible_extra, (30, 30, 255)),
            (visible_missing, (0, 220, 255)),
        ):
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if cv2.contourArea(contour) >= 3:
                    cv2.drawContours(
                        difference_view,
                        [contour],
                        -1,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )

        return reference_view, test_view, difference_view

    def analyze(
        self,
        full_gab: np.ndarray,
        full_test: np.ndarray,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
    ) -> dict:
        try:
            if (
                full_gab is None
                or full_test is None
                or full_gab.size == 0
                or full_test.size == 0
            ):
                return self._empty_result()

            full_gab, full_test = self._safe_pair(full_gab, full_test)
            full_height, full_width = full_gab.shape[:2]
            roi_gab, roi_test, offset, roi_box = self._extract_roi(
                full_gab,
                full_test,
                aoi_epicenters,
            )
            if roi_gab.size == 0 or roi_test.size == 0:
                return self._empty_result("ROI inválida para comparação estrutural")

            height, width = roi_gab.shape[:2]
            if height < 5 or width < 5:
                return self._empty_result("ROI muito pequena para comparação estrutural")

            roi_test_aligned, shift, alignment_score = self._align_test_to_reference(
                roi_gab,
                roi_test,
            )
            dx, dy = shift

            mask_gab = self._extract_structure(roi_gab)
            mask_test = self._extract_structure(roi_test_aligned)

            tolerance_radius = max(
                1,
                min(3, int(round(min(height, width) * 0.012))),
            )
            tolerance_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (tolerance_radius * 2 + 1, tolerance_radius * 2 + 1),
            )
            mask_gab_tolerant = cv2.dilate(mask_gab, tolerance_kernel)
            mask_test_tolerant = cv2.dilate(mask_test, tolerance_kernel)

            extra_mask = cv2.bitwise_and(
                mask_test,
                cv2.bitwise_not(mask_gab_tolerant),
            )
            missing_mask = cv2.bitwise_and(
                mask_gab,
                cv2.bitwise_not(mask_test_tolerant),
            )

            component_min_area = max(2, int(height * width * 0.0005))
            extra_mask = self._remove_small_components(
                extra_mask,
                component_min_area,
            )
            missing_mask = self._remove_small_components(
                missing_mask,
                component_min_area,
            )

            border_ignore = max(2, min(5, int(round(min(height, width) * 0.025))))
            for mask in (extra_mask, missing_mask):
                mask[:border_ignore, :] = 0
                mask[-border_ignore:, :] = 0
                mask[:, :border_ignore] = 0
                mask[:, -border_ignore:] = 0

            match_mask = cv2.bitwise_and(mask_gab, mask_test_tolerant)
            diff_mask = cv2.bitwise_or(extra_mask, missing_mask)
            structure_union = cv2.bitwise_or(mask_gab, mask_test)

            structure_pixels = max(cv2.countNonZero(structure_union), 1)
            extra_pixels = cv2.countNonZero(extra_mask)
            missing_pixels = cv2.countNonZero(missing_mask)
            wrong_pixels = extra_pixels + missing_pixels

            error_pct = float(wrong_pixels / structure_pixels)
            extra_pct = float(extra_pixels / structure_pixels)
            missing_pct = float(missing_pixels / structure_pixels)
            matched_pct = float(max(0.0, 1.0 - error_pct))

            tolerance = 0.08
            if aoi_info:
                value_text = str(aoi_info.get("value", "")).upper()
                category_text = str(aoi_info.get("category", "")).upper()
                if "SHIFT" in value_text or "SHIFT" in category_text or "SIFT" in value_text:
                    tolerance = 0.15

            reference_view, test_view, difference_view = self._build_visualizations(
                roi_gab,
                roi_test_aligned,
                mask_gab,
                mask_test,
                match_mask,
                extra_mask,
                missing_mask,
            )

            result = {
                "is_defect": False,
                "silk_error_pct": error_pct,
                "tolerance": tolerance,
                "pct_changed": error_pct,
                "extra_pct": extra_pct,
                "missing_pct": missing_pct,
                "matched_pct": matched_pct,
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "alignment_score": float(alignment_score),
                "reason": "",
                "bounding_box": None,
                "roi_box": roi_box,
                "roi_width": int(width),
                "roi_height": int(height),
                "comparison_mode": "structural_xor",
                "mask_gab": mask_gab,
                "mask_test": mask_test,
                "match_mask": match_mask,
                "extra_mask": extra_mask,
                "missing_mask": missing_mask,
                "diff_mask": diff_mask,
                "roi_gab": roi_gab,
                "roi_test_aligned": roi_test_aligned,
                "reference_view": reference_view,
                "test_view": test_view,
                "difference_view": difference_view,
            }

            is_critical = error_pct > tolerance
            contours, _ = cv2.findContours(
                cv2.dilate(
                    diff_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            significant = [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= max(4, component_min_area)
            ]

            if is_critical and significant:
                largest = max(significant, key=cv2.contourArea)
                local_x, local_y, box_width, box_height = cv2.boundingRect(largest)
                offset_x, offset_y = offset
                margin = 4
                real_x = max(0, local_x + offset_x - margin)
                real_y = max(0, local_y + offset_y - margin)
                real_width = min(
                    full_width - real_x,
                    box_width + margin * 2,
                )
                real_height = min(
                    full_height - real_y,
                    box_height + margin * 2,
                )

                result["is_defect"] = True
                result["bounding_box"] = (
                    real_x,
                    real_y,
                    real_width,
                    real_height,
                )
                result["reason"] = (
                    f"DIVERGÊNCIA ESTRUTURAL NO FOCO ({error_pct:.1%}; "
                    f"extra {extra_pct:.1%}; ausente {missing_pct:.1%})"
                )
            elif error_pct > 0:
                result["reason"] = (
                    f"Diferenças estruturais dentro da tolerância ({error_pct:.1%})"
                )
            else:
                result["reason"] = "Estrutura do teste coincide com o gabarito"

            return result
        except Exception as exc:
            print(f"⚠️ Erro no SilkExpert: {exc}")
            return self._empty_result()
