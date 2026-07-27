"""Motor dedicado ao fluxo, expansão e vazamento de adesivo."""

from __future__ import annotations

import cv2
import numpy as np


class ShiftExpert:
    """
    Compatibilidade: o nome ShiftExpert permanece, mas dx/dy agora medem o
    deslocamento do centro da massa adesiva, não do componente inteiro.
    """

    MODE = "adhesive_flow"
    TERMS = ("MUCH ADHESIVE", "ADHESIVE", "ADESIVO", "COLA", "GLUE")

    @classmethod
    def _is_adhesive_context(cls, info: dict | None) -> bool:
        if not info:
            return False
        text = " ".join(str(info.get(k, "")) for k in ("category", "value", "parts")).upper()
        return any(term in text for term in cls.TERMS)

    @staticmethod
    def _empty(active=False, reason="") -> dict:
        return {
            "is_defect": False,
            "adhesive_is_defect": False,
            "shift_active": active,
            "comparison_mode": ShiftExpert.MODE,
            "shift_pixels": 0.0,
            "shift_pct": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "adhesive_shift_pixels": 0.0,
            "adhesive_shift_pct": 0.0,
            "adhesive_dx": 0.0,
            "adhesive_dy": 0.0,
            "tolerance": 0.32,
            "adhesive_tolerance": 0.32,
            "adhesive_score": 0.0,
            "excess_coverage": 0.0,
            "padding_overlap": 0.0,
            "area_growth_ratio": 0.0,
            "spread_growth_ratio": 0.0,
            "lower_leakage_ratio": 0.0,
            "reference_area_pct": 0.0,
            "test_area_pct": 0.0,
            "alignment_score": 0.0,
            "alignment_shift": (0.0, 0.0),
            "adhesive_alignment_score": 0.0,
            "adhesive_alignment_shift": (0.0, 0.0),
            "adhesive_direction": "ESTÁVEL",
            "reason": reason,
            "adhesive_reason": reason,
            "bounding_box": None,
            "roi_box": None,
            "roi_width": 0,
            "roi_height": 0,
            "reference_centroid": None,
            "test_centroid": None,
            "reference_mask": None,
            "test_mask": None,
            "excess_mask": None,
            "padding_overlap_mask": None,
            "reference_view": None,
            "test_view": None,
            "flow_view": None,
        }

    @staticmethod
    def _safe_pair(ref: np.ndarray, test: np.ndarray):
        ref, test = ref.copy(), test.copy()
        if ref.shape != test.shape:
            test = cv2.resize(test, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        return ref, test

    @staticmethod
    def _roi(ref: np.ndarray, test: np.ndarray, epicenters):
        h, w = ref.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h
        if epicenters:
            x, y, rw, rh = epicenters[0]
            cx1, cy1 = max(0, int(x)), max(0, int(y))
            cx2, cy2 = min(w, int(x + rw)), min(h, int(y + rh))
            if cx2 > cx1 and cy2 > cy1:
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
        return (
            ref[y1:y2, x1:x2].copy(),
            test[y1:y2, x1:x2].copy(),
            (x1, y1),
            (x1, y1, x2 - x1, y2 - y1),
        )

    @staticmethod
    def _align(ref: np.ndarray, test: np.ndarray):
        h, w = ref.shape[:2]
        if min(h, w) < 16:
            return ref.copy(), (0.0, 0.0), 0.0
        template = cv2.GaussianBlur(
            cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (5, 5), 0,
        )
        moving = cv2.GaussianBlur(
            cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (5, 5), 0,
        )
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        try:
            score, warp = cv2.findTransformECC(
                template, moving, warp, cv2.MOTION_TRANSLATION, criteria, None, 3
            )
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            if abs(dx) > w * 0.20 or abs(dy) > h * 0.20:
                return ref.copy(), (0.0, 0.0), 0.0
            aligned = cv2.warpAffine(
                ref, warp, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned, (dx, dy), float(score)
        except cv2.error:
            return ref.copy(), (0.0, 0.0), 0.0

    @staticmethod
    def _warm(hue: np.ndarray):
        hue = hue.astype(np.float32)
        orange = np.clip(1.0 - np.abs(hue - 10.0) / 22.0, 0.0, 1.0)
        red = np.clip(
            1.0 - np.minimum(np.abs(hue - 179.0), np.abs(hue + 1.0)) / 12.0,
            0.0, 1.0,
        )
        return np.maximum(orange, red)

    @classmethod
    def _adhesive(cls, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        blue, green, red = cv2.split(image.astype(np.float32))
        hue, saturation, value = cv2.split(hsv)
        warm = cls._warm(hue)
        sat = 0.05 + 0.95 * np.clip((saturation - 25.0) / 130.0, 0.0, 1.0)
        dark = np.clip((210.0 - value) / 175.0, 0.0, 1.0)
        red_dom = np.clip((red - np.maximum(green, blue) + 12.0) / 105.0, 0.0, 1.0)
        lab_red = np.clip((lab[:, :, 1] - 128.0) / 55.0, 0.0, 1.0)
        return np.clip(
            warm * sat
            * (0.08 + 0.92 * np.power(dark, 1.35))
            * (0.18 + 0.52 * red_dom + 0.30 * lab_red),
            0.0, 1.0,
        )

    @staticmethod
    def _copper(image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        hue, saturation, value = cv2.split(hsv)
        orange = np.clip(1.0 - np.abs(hue - 13.0) / 18.0, 0.0, 1.0)
        sat = np.clip((saturation - 50.0) / 145.0, 0.0, 1.0)
        bright = np.clip((value - 100.0) / 140.0, 0.0, 1.0)
        yellow = np.clip((lab[:, :, 2] - 130.0) / 68.0, 0.0, 1.0)
        return np.clip(orange * sat * bright * (0.52 + 0.48 * yellow), 0.0, 1.0)

    @staticmethod
    def _metal(image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        b, g, r = cv2.split(image.astype(np.float32))
        _, saturation, value = cv2.split(hsv)
        spread = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])
        neutral = np.clip(1.0 - spread / 60.0, 0.0, 1.0)
        low_sat = np.clip(1.0 - saturation / 80.0, 0.0, 1.0)
        bright = np.clip((value - 90.0) / 155.0, 0.0, 1.0)
        return np.clip(neutral * low_sat * bright, 0.0, 1.0)

    @classmethod
    def _material(cls, image: np.ndarray):
        return np.clip(
            cls._adhesive(image)
            * (1.0 - 0.72 * cls._copper(image))
            * (1.0 - 0.78 * cls._metal(image)),
            0.0, 1.0,
        )

    @staticmethod
    def _clean(mask: np.ndarray, min_area: int):
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                output[labels == label] = 255
        return cv2.morphologyEx(
            output, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

    @staticmethod
    def _centroid(prob: np.ndarray):
        weights = np.clip(prob.astype(np.float64), 0.0, None)
        total = float(weights.sum())
        if total <= 1e-9:
            return None
        yy, xx = np.indices(weights.shape)
        return (
            float((xx * weights).sum() / total),
            float((yy * weights).sum() / total),
        )

    @staticmethod
    def _spread(prob: np.ndarray, centroid):
        if centroid is None:
            return 0.0
        weights = np.clip(prob.astype(np.float64), 0.0, None)
        total = float(weights.sum())
        if total <= 1e-9:
            return 0.0
        yy, xx = np.indices(weights.shape)
        cx, cy = centroid
        return float(np.sqrt((((xx - cx) ** 2 + (yy - cy) ** 2) * weights).sum() / total))

    @staticmethod
    def _direction(dx: float, dy: float, threshold=1.0):
        if abs(dx) < threshold and abs(dy) < threshold:
            return "ESTÁVEL"
        parts = []
        if dy > threshold:
            parts.append("BAIXO")
        elif dy < -threshold:
            parts.append("CIMA")
        if dx > threshold:
            parts.append("DIREITA")
        elif dx < -threshold:
            parts.append("ESQUERDA")
        return " + ".join(parts)

    @staticmethod
    def _paint(image, mask, color, alpha):
        output = image.copy().astype(np.float32)
        selected = mask > 0
        if np.any(selected):
            c = np.asarray(color, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + c * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @classmethod
    def _views(cls, ref, test, ref_mask, test_mask, excess, on_padding, stable, ref_c, test_c):
        ref_view = cls._paint(
            ref, cv2.dilate(ref_mask, np.ones((3, 3), np.uint8)), (60, 210, 80), 0.68
        )
        test_view = cls._paint(
            test, cv2.dilate(test_mask, np.ones((3, 3), np.uint8)), (255, 200, 30), 0.68
        )
        flow = (test.astype(np.float32) * 0.68).astype(np.uint8)
        flow = cls._paint(flow, cv2.dilate(stable, np.ones((3, 3), np.uint8)), (70, 190, 90), 0.40)
        flow = cls._paint(flow, cv2.dilate(excess, np.ones((5, 5), np.uint8)), (35, 55, 255), 0.90)
        flow = cls._paint(flow, cv2.dilate(on_padding, np.ones((5, 5), np.uint8)), (0, 220, 255), 0.95)

        for mask, color in ((excess, (20, 20, 255)), (on_padding, (0, 220, 255))):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) >= 3:
                    cv2.drawContours(flow, [contour], -1, color, 1, lineType=cv2.LINE_AA)

        if ref_c is not None:
            cv2.drawMarker(
                flow, tuple(int(round(v)) for v in ref_c), (80, 255, 80),
                cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA,
            )
        if test_c is not None:
            cv2.drawMarker(
                flow, tuple(int(round(v)) for v in test_c), (0, 220, 255),
                cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA,
            )
        if ref_c is not None and test_c is not None:
            cv2.arrowedLine(
                flow,
                tuple(int(round(v)) for v in ref_c),
                tuple(int(round(v)) for v in test_c),
                (0, 220, 255), 1, cv2.LINE_AA, tipLength=0.25,
            )
        return ref_view, test_view, flow

    def analyze(
        self,
        full_gab: np.ndarray,
        full_test: np.ndarray,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
    ) -> dict:
        try:
            if not self._is_adhesive_context(aoi_info):
                return self._empty(False, "Motor reservado para categorias de adesivo")
            if (
                full_gab is None or full_test is None
                or full_gab.size == 0 or full_test.size == 0
            ):
                return self._empty(True, "Imagem nula")

            full_gab, full_test = self._safe_pair(full_gab, full_test)
            ref, test, offset, roi_box = self._roi(full_gab, full_test, aoi_epicenters)
            if ref.size == 0 or test.size == 0:
                return self._empty(True, "ROI inválida")
            h, w = ref.shape[:2]
            if min(h, w) < 7:
                return self._empty(True, "ROI muito pequena para analisar adesivo")

            aligned, align_shift, align_score = self._align(ref, test)
            ref_prob, test_prob = self._material(aligned), self._material(test)
            copper_ref = self._copper(aligned)

            lab_ref = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_test = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
            changed = np.clip(
                (np.linalg.norm(lab_test - lab_ref, axis=2) / 110.0 - 0.05) / 0.70,
                0.0, 1.0,
            )

            min_area = max(3, int(h * w * 0.0007))
            ref_mask = self._clean((ref_prob >= 0.22).astype(np.uint8) * 255, min_area)
            test_mask = self._clean((test_prob >= 0.22).astype(np.uint8) * 255, min_area)
            ref_tolerant = cv2.dilate(
                ref_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )

            excess_prob = np.clip(test_prob - ref_prob - 0.025, 0.0, 1.0)
            excess_prob *= 0.18 + 0.82 * changed
            raw_excess = (
                (excess_prob >= 0.025)
                | ((test_mask > 0) & (ref_tolerant == 0) & (changed > 0.08))
            )
            excess = self._clean(raw_excess.astype(np.uint8) * 255, min_area)
            stable = cv2.bitwise_and(test_mask, ref_tolerant)
            on_padding = cv2.bitwise_and(
                excess, (copper_ref >= 0.16).astype(np.uint8) * 255
            )

            pixels = float(h * w)
            ref_area = float(cv2.countNonZero(ref_mask))
            test_area = float(cv2.countNonZero(test_mask))
            excess_area = float(cv2.countNonZero(excess))
            padding_area = float(cv2.countNonZero(on_padding))

            ref_mass = ref_prob * (ref_mask.astype(np.float32) / 255.0)
            test_mass = test_prob * (test_mask.astype(np.float32) / 255.0)
            ref_c, test_c = self._centroid(ref_mass), self._centroid(test_mass)
            if ref_c is None and test_c is not None:
                ref_c = (w / 2.0, h / 2.0)
            dx = float(test_c[0] - ref_c[0]) if ref_c and test_c else 0.0
            dy = float(test_c[1] - ref_c[1]) if ref_c and test_c else 0.0
            shift_pixels = float(np.hypot(dx, dy))
            shift_pct = shift_pixels / max(w, h, 1)

            ref_spread = self._spread(ref_mass, ref_c)
            test_spread = self._spread(test_mass, test_c)
            spread_growth = max(0.0, (test_spread - ref_spread) / max(ref_spread, 1.0))
            area_growth = max(
                0.0, (test_area - ref_area) / max(ref_area, pixels * 0.005)
            )
            excess_coverage, padding_overlap = excess_area / pixels, padding_area / pixels
            lower_ratio = (
                float(cv2.countNonZero(excess[h // 2 :, :]) / max(excess_area, 1.0))
                if excess_area > 0 else 0.0
            )

            score = float(np.clip(
                0.40 * min(1.0, excess_coverage / 0.10)
                + 0.27 * min(1.0, padding_overlap / 0.045)
                + 0.18 * min(1.0, area_growth / 1.0)
                + 0.10 * min(1.0, spread_growth / 0.65)
                + 0.05 * lower_ratio,
                0.0, 1.0,
            ))
            tolerance = 0.32
            is_defect = bool(
                score > tolerance
                and (excess_coverage > 0.004 or padding_overlap > 0.002)
            )

            ref_view, test_view, flow_view = self._views(
                aligned, test, ref_mask, test_mask, excess, on_padding, stable, ref_c, test_c
            )
            bounding_box = None
            contours, _ = cv2.findContours(
                cv2.dilate(excess, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            significant = [c for c in contours if cv2.contourArea(c) >= max(4, min_area)]
            if significant:
                x, y, bw, bh = cv2.boundingRect(max(significant, key=cv2.contourArea))
                ox, oy = offset
                bounding_box = (max(0, x + ox - 4), max(0, y + oy - 4), bw + 8, bh + 8)

            direction = self._direction(dx, dy)
            if is_defect:
                reason = (
                    f"ADESIVO EXCEDENTE ({score:.0%}): excesso {excess_coverage:.1%}, "
                    f"padding {padding_overlap:.1%}, expansão {area_growth:.0%}, "
                    f"fluxo {direction}"
                )
            elif excess_coverage > 0:
                reason = (
                    f"Adesivo dentro da tolerância: excesso {excess_coverage:.1%}, "
                    f"score {score:.0%}"
                )
            else:
                reason = "Distribuição do adesivo coincide com o gabarito"

            return {
                "is_defect": is_defect,
                "adhesive_is_defect": is_defect,
                "shift_active": True,
                "comparison_mode": self.MODE,
                "shift_pixels": round(shift_pixels, 2),
                "shift_pct": float(shift_pct),
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "adhesive_shift_pixels": round(shift_pixels, 2),
                "adhesive_shift_pct": float(shift_pct),
                "adhesive_dx": round(dx, 2),
                "adhesive_dy": round(dy, 2),
                "tolerance": tolerance,
                "adhesive_tolerance": tolerance,
                "adhesive_score": score,
                "excess_coverage": float(excess_coverage),
                "padding_overlap": float(padding_overlap),
                "area_growth_ratio": float(area_growth),
                "spread_growth_ratio": float(spread_growth),
                "lower_leakage_ratio": float(lower_ratio),
                "reference_area_pct": float(ref_area / pixels),
                "test_area_pct": float(test_area / pixels),
                "alignment_score": float(align_score),
                "alignment_shift": tuple(float(v) for v in align_shift),
                "adhesive_alignment_score": float(align_score),
                "adhesive_alignment_shift": tuple(float(v) for v in align_shift),
                "adhesive_direction": direction,
                "reason": reason,
                "adhesive_reason": reason,
                "bounding_box": bounding_box if is_defect else None,
                "roi_box": roi_box,
                "roi_width": int(w),
                "roi_height": int(h),
                "reference_centroid": ref_c,
                "test_centroid": test_c,
                "reference_mask": ref_mask,
                "test_mask": test_mask,
                "excess_mask": excess,
                "padding_overlap_mask": on_padding,
                "reference_view": ref_view,
                "test_view": test_view,
                "flow_view": flow_view,
            }
        except Exception as exc:
            print(f"⚠️ Erro no Adhesive Shift Expert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")
