"""Motor de expectativa visual do patch para a categoria FALTANDO.

A ROI enviada pela AOI é tratada como uma amostra visual estratégica. O motor
não tenta descobrir o componente inteiro: ele verifica se o conteúdo recebido
na mesma posição continua compatível com o patch do gabarito.
"""

from __future__ import annotations

import cv2
import numpy as np


class ROIPatchExpectationExpert:
    MODE = "roi_patch_expectation"
    CATEGORIES = frozenset({"FALTANDO", "MISSING"})
    TOLERANCE = 0.36

    @classmethod
    def _is_active_category(cls, info: dict | None) -> bool:
        if not isinstance(info, dict):
            return False
        category = " ".join(str(info.get("category", "")).upper().split())
        return category in cls.CATEGORIES

    @classmethod
    def _empty(cls, active: bool = False, reason: str = "") -> dict:
        return {
            "missing_active": bool(active),
            "missing_comparison_mode": cls.MODE,
            "missing_is_defect": False,
            "missing_score": 0.0,
            "missing_tolerance": cls.TOLERANCE,
            "missing_expectation_mode": "patch",
            "missing_patch_type": "unknown",
            "missing_classification": "SEM DADOS",
            "missing_structure_loss": 0.0,
            "missing_extra_structure": 0.0,
            "missing_edge_mismatch": 0.0,
            "missing_coverage": 0.0,
            "missing_changed_coverage": 0.0,
            "missing_appearance_loss": 0.0,
            "missing_background_exposure": 0.0,
            "missing_presence_retention": 1.0,
            "missing_direct_similarity": 1.0,
            "missing_best_similarity": 0.0,
            "missing_displacement_dx": 0.0,
            "missing_displacement_dy": 0.0,
            "missing_displacement_pixels": 0.0,
            "missing_displacement_pct": 0.0,
            "missing_reference_distinctness": 0.0,
            "missing_residual_mean": 0.0,
            "missing_residual_p90": 0.0,
            "missing_residual_peak": 0.0,
            "missing_alignment_score": 0.0,
            "missing_alignment_shift": (0.0, 0.0),
            "missing_roi_box": None,
            "missing_roi_width": 0,
            "missing_roi_height": 0,
            "component_expected_mask": None,
            "component_missing_mask": None,
            "component_matched_mask": None,
            "roi_anomaly_mask": None,
            "missing_residual_map": None,
            "missing_heatmap": None,
            "missing_reference_view": None,
            "missing_test_view": None,
            "missing_reconstruction_view": None,
            "missing_bounding_box": None,
            "missing_reason": reason,
        }

    @staticmethod
    def _safe_pair(reference: np.ndarray, test: np.ndarray):
        reference = reference.copy()
        test = test.copy()
        if reference.shape != test.shape:
            test = cv2.resize(
                test,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return reference, test

    @staticmethod
    def _roi_box(image: np.ndarray, epicenters):
        height, width = image.shape[:2]
        if epicenters:
            x, y, roi_width, roi_height = epicenters[0]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, x1 + max(1, int(round(roi_width))))
            y2 = min(height, y1 + max(1, int(round(roi_height))))
            if x2 > x1 and y2 > y1:
                return x1, y1, x2 - x1, y2 - y1
        return 0, 0, width, height

    @staticmethod
    def _crop(image: np.ndarray, box):
        x, y, width, height = box
        return image[y : y + height, x : x + width].copy()

    @staticmethod
    def _ring_mask(shape, box, scale: float = 0.45):
        image_height, image_width = shape[:2]
        x, y, width, height = box
        margin_x = max(5, int(round(width * scale)))
        margin_y = max(5, int(round(height * scale)))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image_width, x + width + margin_x)
        y2 = min(image_height, y + height + margin_y)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        inner_margin_x = max(2, int(round(width * 0.08)))
        inner_margin_y = max(2, int(round(height * 0.08)))
        ix1 = max(0, x - inner_margin_x)
        iy1 = max(0, y - inner_margin_y)
        ix2 = min(image_width, x + width + inner_margin_x)
        iy2 = min(image_height, y + height + inner_margin_y)
        mask[iy1:iy2, ix1:ix2] = 0
        return mask

    @classmethod
    def _normalize_illumination(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        roi_box,
    ) -> np.ndarray:
        """Corrige apenas tendência de iluminação medida fora da ROI."""
        mask = cls._ring_mask(full_reference.shape, roi_box)
        selected = mask > 0
        if int(np.count_nonzero(selected)) < 24:
            return full_test.copy()
        reference_lab = cv2.cvtColor(full_reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(full_test, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = np.median(reference_lab[selected] - test_lab[selected], axis=0)
        delta = np.clip(delta, [-16.0, -7.0, -7.0], [16.0, 7.0, 7.0])
        corrected = np.clip(test_lab + delta.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _auto_edges(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(gray))
        lower = int(max(14, 0.56 * median))
        upper = int(min(245, max(lower + 24, 1.44 * median)))
        return cv2.Canny(gray, lower, upper)

    @staticmethod
    def _shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        return cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    @classmethod
    def _local_color_residual(
        cls,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> np.ndarray:
        """Compara cada pixel com uma vizinhança mínima do gabarito.

        Isso tolera blur e jitter de borda, mas não permite que uma estrutura
        realmente deslocada atravesse a ROI sem produzir divergência.
        """
        height, width = reference.shape[:2]
        radius = max(1, min(3, int(round(min(height, width) * 0.025))))
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_lab = cv2.GaussianBlur(reference_lab, (3, 3), 0)
        test_lab = cv2.GaussianBlur(test_lab, (3, 3), 0)
        best = np.full((height, width), np.inf, dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = cls._shifted(reference_lab, dx, dy)
                distance = np.linalg.norm(test_lab - shifted, axis=2) / 62.0
                best = np.minimum(best, distance.astype(np.float32))
        return np.clip(best, 0.0, 1.0)

    @classmethod
    def _edge_metrics(cls, reference: np.ndarray, test: np.ndarray):
        reference_edges = cls._auto_edges(reference)
        test_edges = cls._auto_edges(test)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reference_tolerant = cv2.dilate(reference_edges, kernel)
        test_tolerant = cv2.dilate(test_edges, kernel)
        missing_edges = cv2.bitwise_and(reference_edges, cv2.bitwise_not(test_tolerant))
        extra_edges = cv2.bitwise_and(test_edges, cv2.bitwise_not(reference_tolerant))
        edge_anomaly = cv2.dilate(
            cv2.bitwise_or(missing_edges, extra_edges),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        reference_count = float(cv2.countNonZero(reference_edges))
        test_count = float(cv2.countNonZero(test_edges))
        total = max(reference_count + test_count, 1.0)
        missing_ratio = float(cv2.countNonZero(missing_edges) / max(reference_count, 1.0))
        extra_ratio = float(cv2.countNonZero(extra_edges) / max(test_count, 1.0))
        mismatch = float(
            (cv2.countNonZero(missing_edges) + cv2.countNonZero(extra_edges)) / total
        )
        return edge_anomaly, missing_ratio, extra_ratio, mismatch

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        height, width = mask.shape[:2]
        minimum_area = max(4, int(round(height * width * 0.003)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        output = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                output[labels == label] = 255
        return output

    @classmethod
    def _residual_and_mask(cls, reference: np.ndarray, test: np.ndarray):
        color_residual = cls._local_color_residual(reference, test)
        edge_anomaly, missing_edges, extra_edges, edge_mismatch = cls._edge_metrics(
            reference,
            test,
        )
        edge_layer = (edge_anomaly > 0).astype(np.float32)
        residual = np.clip(0.82 * color_residual + 0.18 * edge_layer, 0.0, 1.0)

        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        texture = float(np.std(reference_gray) / 64.0)
        edge_density = float(np.mean(cls._auto_edges(reference) > 0))
        threshold = 0.27
        if texture > 0.65 or edge_density > 0.14:
            threshold = 0.31
        mask = (residual >= threshold).astype(np.uint8) * 255
        mask = cls._clean_mask(mask)
        return residual, mask, missing_edges, extra_edges, edge_mismatch, texture, edge_density

    @staticmethod
    def _direct_similarity(
        residual: np.ndarray,
        coverage: float,
        edge_mismatch: float,
    ) -> float:
        mean_residual = float(np.mean(residual))
        p90 = float(np.percentile(residual, 90))
        loss = 0.50 * coverage + 0.27 * mean_residual + 0.13 * p90 + 0.10 * edge_mismatch
        return float(np.clip(1.0 - loss, 0.0, 1.0))

    @classmethod
    def _patch_distinctness(cls, reference: np.ndarray) -> float:
        lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        spread = float(np.mean(np.std(lab.reshape(-1, 3), axis=0)) / 42.0)
        edges = float(np.mean(cls._auto_edges(reference) > 0) / 0.15)
        return float(np.clip(0.58 * spread + 0.42 * edges, 0.0, 1.0))

    @classmethod
    def _nearby_match(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        roi_box,
        distinctness: float,
    ):
        if distinctness < 0.20:
            return 0.0, 0.0, 0.0, None
        x, y, width, height = roi_box
        template = cls._crop(full_reference, roi_box)
        image_height, image_width = full_test.shape[:2]
        margin_x = max(5, int(round(width * 0.65)))
        margin_y = max(5, int(round(height * 0.65)))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image_width, x + width + margin_x)
        y2 = min(image_height, y + height + margin_y)
        search = full_test[y1:y2, x1:x2]
        if search.shape[0] < height or search.shape[1] < width:
            return 0.0, 0.0, 0.0, None

        template_gray = cv2.equalizeHist(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
        search_gray = cv2.equalizeHist(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY))
        template_edge = cls._auto_edges(template)
        search_edge = cls._auto_edges(search)
        maps = []
        if float(np.std(template_gray)) > 3.0:
            maps.append(cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED))
        if cv2.countNonZero(template_edge) >= 8:
            maps.append(cv2.matchTemplate(search_edge, template_edge, cv2.TM_CCOEFF_NORMED))
        if not maps:
            return 0.0, 0.0, 0.0, None
        combined = np.mean(np.stack(maps, axis=0), axis=0)
        _, maximum, _, location = cv2.minMaxLoc(combined.astype(np.float32))
        best_x = x1 + int(location[0])
        best_y = y1 + int(location[1])
        dx = float(best_x - x)
        dy = float(best_y - y)
        return (
            float(np.clip(maximum, 0.0, 1.0)),
            dx,
            dy,
            (best_x, best_y, width, height),
        )

    @classmethod
    def _background_replacement_signal(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        reference_roi: np.ndarray,
        test_roi: np.ndarray,
        roi_box,
    ) -> float:
        mask = cls._ring_mask(full_reference.shape, roi_box)
        selected = mask > 0
        if int(np.count_nonzero(selected)) < 24:
            return 0.0
        reference_lab = cv2.cvtColor(full_reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(full_test, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_roi_lab = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_roi_lab = cv2.cvtColor(test_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_reference = np.median(reference_lab[selected], axis=0)
        background_test = np.median(test_lab[selected], axis=0)
        reference_distance = float(
            np.linalg.norm(np.median(reference_roi_lab.reshape(-1, 3), axis=0) - background_reference)
            / 85.0
        )
        test_distance = float(
            np.linalg.norm(np.median(test_roi_lab.reshape(-1, 3), axis=0) - background_test)
            / 85.0
        )
        return float(np.clip(reference_distance - test_distance, 0.0, 1.0))

    @staticmethod
    def _score(
        coverage: float,
        residual_mean: float,
        residual_p90: float,
        edge_mismatch: float,
    ) -> float:
        coverage_signal = float(np.clip((coverage - 0.035) / 0.52, 0.0, 1.0))
        intensity_signal = float(np.clip((residual_mean - 0.045) / 0.44, 0.0, 1.0))
        peak_signal = float(np.clip((residual_p90 - 0.16) / 0.68, 0.0, 1.0))
        edge_signal = float(np.clip(edge_mismatch / 0.62, 0.0, 1.0))
        score = (
            0.50 * coverage_signal
            + 0.24 * intensity_signal
            + 0.14 * peak_signal
            + 0.12 * edge_signal
        )
        if coverage > 0.62 and residual_mean > 0.24:
            score = max(score, 0.68)
        return float(np.clip(score, 0.0, 1.0))

    @classmethod
    def _classification(
        cls,
        is_defect: bool,
        patch_type: str,
        coverage: float,
        background_signal: float,
        best_similarity: float,
        displacement_pixels: float,
        roi_diagonal: float,
    ) -> str:
        if not is_defect:
            return "ROI CONFORME"
        if (
            best_similarity >= 0.52
            and displacement_pixels >= max(3.0, roi_diagonal * 0.055)
        ):
            return "DESLOCAMENTO PROVÁVEL"
        if background_signal > 0.30 and coverage > 0.28:
            return "CONTEÚDO ESPERADO AUSENTE"
        if patch_type == "homogeneous":
            return "CONTEÚDO INESPERADO NA ROI"
        if coverage < 0.66:
            return "DIVERGÊNCIA PARCIAL NA ROI"
        return "QUEBRA DA EXPECTATIVA VISUAL"

    @staticmethod
    def _paint(image: np.ndarray, mask: np.ndarray, color, alpha: float):
        output = image.astype(np.float32).copy()
        selected = mask > 0
        if np.any(selected):
            paint = np.asarray(color, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + paint * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @classmethod
    def _views(
        cls,
        reference: np.ndarray,
        test: np.ndarray,
        residual: np.ndarray,
        anomaly_mask: np.ndarray,
    ):
        reference_view = reference.copy()
        test_view = test.copy()
        reconstruction = (test.astype(np.float32) * 0.78).astype(np.uint8)

        heat_source = np.clip(residual * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat_source, cv2.COLORMAP_TURBO)
        selected = anomaly_mask > 0
        if np.any(selected):
            reconstruction[selected] = (
                reconstruction[selected].astype(np.float32) * 0.20
                + heatmap[selected].astype(np.float32) * 0.80
            ).astype(np.uint8)

        contours, _ = cv2.findContours(
            anomaly_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(test_view, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.drawContours(reconstruction, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        height, width = anomaly_mask.shape[:2]
        cv2.rectangle(reference_view, (0, 0), (width - 1, height - 1), (0, 220, 255), 1)
        cv2.rectangle(test_view, (0, 0), (width - 1, height - 1), (0, 220, 255), 1)

        if np.any(selected):
            weighted = residual * selected.astype(np.float32)
            _, _, _, maximum_location = cv2.minMaxLoc(weighted.astype(np.float32))
            cv2.drawMarker(
                reconstruction,
                maximum_location,
                (255, 230, 40),
                cv2.MARKER_CROSS,
                10,
                1,
                cv2.LINE_AA,
            )
        return reference_view, test_view, reconstruction, heatmap

    @staticmethod
    def _bounding_box(mask: np.ndarray, roi_box):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        height, width = mask.shape[:2]
        significant = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= max(4, height * width * 0.003)
        ]
        if not significant:
            return None
        combined = np.vstack(significant)
        x, y, box_width, box_height = cv2.boundingRect(combined)
        offset_x, offset_y, _, _ = roi_box
        margin = 3
        return (
            max(0, offset_x + x - margin),
            max(0, offset_y + y - margin),
            box_width + margin * 2,
            box_height + margin * 2,
        )

    def analyze(
        self,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
    ) -> dict:
        try:
            if not self._is_active_category(aoi_info):
                return self._empty(False, "Motor reservado para a categoria FALTANDO")
            if (
                full_reference is None
                or full_test is None
                or full_reference.size == 0
                or full_test.size == 0
            ):
                return self._empty(True, "Imagem nula")

            full_reference, full_test = self._safe_pair(full_reference, full_test)
            roi_box = self._roi_box(full_reference, aoi_epicenters)
            reference = self._crop(full_reference, roi_box)
            normalized_full_test = self._normalize_illumination(
                full_reference,
                full_test,
                roi_box,
            )
            test = self._crop(normalized_full_test, roi_box)
            raw_test = self._crop(full_test, roi_box)
            if reference.size == 0 or test.size == 0 or min(reference.shape[:2]) < 8:
                return self._empty(True, "ROI inválida ou pequena demais")

            (
                residual,
                anomaly_mask,
                structure_loss,
                extra_structure,
                edge_mismatch,
                texture,
                edge_density,
            ) = self._residual_and_mask(reference, test)
            coverage = float(np.mean(anomaly_mask > 0))
            residual_mean = float(np.mean(residual[anomaly_mask > 0])) if coverage > 0 else float(np.mean(residual))
            residual_p90 = float(np.percentile(residual, 90))
            residual_peak = float(np.max(residual))
            direct_similarity = self._direct_similarity(residual, coverage, edge_mismatch)
            distinctness = self._patch_distinctness(reference)
            patch_type = (
                "homogeneous"
                if texture < 0.48 and edge_density < 0.10
                else "structured"
            )
            best_similarity, dx, dy, best_box = self._nearby_match(
                full_reference,
                normalized_full_test,
                roi_box,
                distinctness,
            )
            displacement_pixels = float(np.hypot(dx, dy))
            roi_diagonal = max(float(np.hypot(reference.shape[1], reference.shape[0])), 1.0)
            displacement_pct = float(displacement_pixels / roi_diagonal)
            background_signal = self._background_replacement_signal(
                full_reference,
                normalized_full_test,
                reference,
                test,
                roi_box,
            )
            score = self._score(
                coverage,
                residual_mean,
                residual_p90,
                edge_mismatch,
            )
            is_defect = bool(
                score > self.TOLERANCE
                and (
                    coverage > 0.095
                    or edge_mismatch > 0.30
                    or residual_mean > 0.31
                )
            )
            classification = self._classification(
                is_defect,
                patch_type,
                coverage,
                background_signal,
                best_similarity,
                displacement_pixels,
                roi_diagonal,
            )
            matched_mask = cv2.bitwise_not(anomaly_mask)
            reference_view, test_view, reconstruction, heatmap = self._views(
                reference,
                raw_test,
                residual,
                anomaly_mask,
            )
            bounding_box = self._bounding_box(anomaly_mask, roi_box) if is_defect else None

            patch_label = "HOMOGÊNEO" if patch_type == "homogeneous" else "ESTRUTURADO"
            if is_defect:
                reason = (
                    f"QUEBRA DA EXPECTATIVA VISUAL DA ROI ({score:.0%}) • {classification}: "
                    f"patch {patch_label.lower()}, área divergente {coverage:.0%}, "
                    f"intensidade {residual_mean:.0%}, bordas incompatíveis {edge_mismatch:.0%}"
                )
                if classification == "DESLOCAMENTO PROVÁVEL":
                    reason += f", possível deslocamento X:{dx:+.1f}px Y:{dy:+.1f}px"
            else:
                reason = (
                    f"ROI compatível com o patch do gabarito: similaridade "
                    f"{direct_similarity:.0%}, divergência {coverage:.0%}"
                )

            height, width = reference.shape[:2]
            expected_mask = np.full((height, width), 255, dtype=np.uint8)
            return {
                "missing_active": True,
                "missing_comparison_mode": self.MODE,
                "missing_is_defect": is_defect,
                "missing_score": score,
                "missing_tolerance": self.TOLERANCE,
                "missing_expectation_mode": "patch",
                "missing_patch_type": patch_type,
                "missing_classification": classification,
                "missing_structure_loss": structure_loss,
                "missing_extra_structure": extra_structure,
                "missing_edge_mismatch": edge_mismatch,
                "missing_coverage": coverage,
                "missing_changed_coverage": coverage,
                "missing_appearance_loss": float(1.0 - direct_similarity),
                "missing_background_exposure": background_signal,
                "missing_presence_retention": float(1.0 - coverage),
                "missing_direct_similarity": direct_similarity,
                "missing_best_similarity": best_similarity,
                "missing_displacement_dx": dx,
                "missing_displacement_dy": dy,
                "missing_displacement_pixels": displacement_pixels,
                "missing_displacement_pct": displacement_pct,
                "missing_reference_distinctness": distinctness,
                "missing_residual_mean": residual_mean,
                "missing_residual_p90": residual_p90,
                "missing_residual_peak": residual_peak,
                "missing_alignment_score": 0.0,
                "missing_alignment_shift": (0.0, 0.0),
                "missing_roi_box": roi_box,
                "missing_roi_width": int(width),
                "missing_roi_height": int(height),
                "component_expected_mask": expected_mask,
                "component_missing_mask": anomaly_mask,
                "component_matched_mask": matched_mask,
                "roi_anomaly_mask": anomaly_mask,
                "missing_residual_map": residual,
                "missing_heatmap": heatmap,
                "missing_reference_view": reference_view,
                "missing_test_view": test_view,
                "missing_reconstruction_view": reconstruction,
                "missing_best_match_box": best_box,
                "missing_bounding_box": bounding_box,
                "missing_reason": reason,
            }
        except Exception as exc:
            print(f"Erro no ROIPatchExpectationExpert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")


__all__ = ["ROIPatchExpectationExpert"]
