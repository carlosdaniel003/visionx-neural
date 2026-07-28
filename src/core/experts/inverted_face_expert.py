"""Especialista de assinatura da face para a categoria INVERTIDO.

A ROI da AOI é tratada como uma testemunha visual da orientação correta. O
motor compara a estrutura esperada, a topologia, a direção dominante e sinais
de uma face alternativa sem reposicionar o conteúdo interno da ROI.
"""

from __future__ import annotations

import cv2
import numpy as np


class InvertedFaceExpert:
    MODE = "inverted_face_signature"
    CATEGORIES = frozenset({"INVERTIDO", "INVERTED", "REVERSE", "UP SIDE DOWN"})
    TOLERANCE = 0.43
    ORIENTATION_BINS = 12
    GRID_SIDE = 6

    @classmethod
    def _is_active_category(cls, info: dict | None) -> bool:
        if not isinstance(info, dict):
            return False
        category = " ".join(str(info.get("category", "")).upper().split())
        return category in cls.CATEGORIES

    @classmethod
    def _empty(cls, active: bool = False, reason: str = "") -> dict:
        zeros_hist = [0.0] * cls.ORIENTATION_BINS
        zeros_grid = [[0.0] * cls.GRID_SIDE for _ in range(cls.GRID_SIDE)]
        return {
            "inverted_active": bool(active),
            "inverted_comparison_mode": cls.MODE,
            "inverted_is_defect": False,
            "inverted_score": 0.0,
            "inverted_tolerance": cls.TOLERANCE,
            "inverted_classification": "SEM DADOS",
            "inverted_signature_strength": 0.0,
            "inverted_direct_similarity": 1.0,
            "inverted_feature_loss": 0.0,
            "inverted_extra_structure": 0.0,
            "inverted_topology_mismatch": 0.0,
            "inverted_orientation_mismatch": 0.0,
            "inverted_alternate_face_signal": 0.0,
            "inverted_transform_gain": 0.0,
            "inverted_best_transform": "none",
            "inverted_best_transform_similarity": 0.0,
            "inverted_expected_angle": 0.0,
            "inverted_observed_angle": 0.0,
            "inverted_expected_orientation_strength": 0.0,
            "inverted_observed_orientation_strength": 0.0,
            "inverted_changed_coverage": 0.0,
            "inverted_orientation_hist_reference": zeros_hist,
            "inverted_orientation_hist_test": zeros_hist,
            "inverted_edge_grid_reference": zeros_grid,
            "inverted_edge_grid_test": zeros_grid,
            "inverted_polarity_grid_reference": zeros_grid,
            "inverted_polarity_grid_test": zeros_grid,
            "inverted_roi_box": None,
            "inverted_roi_width": 0,
            "inverted_roi_height": 0,
            "inverted_expected_mask": None,
            "inverted_observed_mask": None,
            "inverted_missing_mask": None,
            "inverted_extra_mask": None,
            "inverted_anomaly_mask": None,
            "inverted_residual_map": None,
            "inverted_reference_view": None,
            "inverted_test_view": None,
            "inverted_reconstruction_view": None,
            "inverted_bounding_box": None,
            "inverted_reason": reason,
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
    def _ring_mask(shape, box, scale: float = 0.42):
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
        inner_x = max(2, int(round(width * 0.08)))
        inner_y = max(2, int(round(height * 0.08)))
        mask[
            max(0, y - inner_y) : min(image_height, y + height + inner_y),
            max(0, x - inner_x) : min(image_width, x + width + inner_x),
        ] = 0
        return mask

    @classmethod
    def _normalize_illumination(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        roi_box,
    ) -> np.ndarray:
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
    def _gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(gray)

    @classmethod
    def _edges(cls, image: np.ndarray) -> np.ndarray:
        gray = cls._gray(image)
        median = float(np.median(gray))
        lower = int(max(12, 0.52 * median))
        upper = int(min(245, max(lower + 24, 1.48 * median)))
        edges = cv2.Canny(gray, lower, upper)
        return cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

    @staticmethod
    def _shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        matrix = np.asarray(
            [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    @classmethod
    def _local_residual(cls, reference: np.ndarray, test: np.ndarray) -> np.ndarray:
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
                distance = np.linalg.norm(test_lab - shifted, axis=2) / 66.0
                best = np.minimum(best, distance.astype(np.float32))
        return np.clip(best, 0.0, 1.0)

    @classmethod
    def _orientation_histogram(cls, image: np.ndarray):
        gray = cls._gray(image).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        orientation = np.mod(angle, 180.0)
        positive = magnitude[magnitude > 1e-6]
        if positive.size == 0:
            return np.zeros(cls.ORIENTATION_BINS, np.float32), 0.0, 0.0
        threshold = float(np.percentile(positive, 58))
        selected = magnitude >= max(threshold, 0.025)
        histogram = np.zeros(cls.ORIENTATION_BINS, dtype=np.float32)
        bin_width = 180.0 / cls.ORIENTATION_BINS
        indices = np.floor(orientation[selected] / bin_width).astype(np.int32)
        indices = np.clip(indices, 0, cls.ORIENTATION_BINS - 1)
        np.add.at(histogram, indices, magnitude[selected])
        total = float(histogram.sum())
        if total <= 1e-8:
            return histogram, 0.0, 0.0
        histogram /= total
        dominant_index = int(np.argmax(histogram))
        dominant_angle = (dominant_index + 0.5) * bin_width
        dominant_strength = float(histogram[dominant_index])
        return histogram, float(dominant_angle), dominant_strength

    @staticmethod
    def _histogram_mismatch(reference_hist: np.ndarray, test_hist: np.ndarray) -> float:
        ref_norm = float(np.linalg.norm(reference_hist))
        test_norm = float(np.linalg.norm(test_hist))
        if ref_norm <= 1e-8 and test_norm <= 1e-8:
            return 0.0
        if ref_norm <= 1e-8 or test_norm <= 1e-8:
            return 1.0
        cosine = float(
            np.dot(reference_hist, test_hist) / (ref_norm * test_norm)
        )
        return float(np.clip(1.0 - cosine, 0.0, 1.0))

    @classmethod
    def _grids(cls, image: np.ndarray, edges: np.ndarray):
        edge_grid = cv2.resize(
            (edges > 0).astype(np.float32),
            (cls.GRID_SIDE, cls.GRID_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        gray = cls._gray(image).astype(np.float32)
        median = float(np.median(gray))
        spread = max(float(np.percentile(gray, 90) - np.percentile(gray, 10)), 18.0)
        polarity = np.clip((gray - median) / spread * 0.5 + 0.5, 0.0, 1.0)
        polarity_grid = cv2.resize(
            polarity,
            (cls.GRID_SIDE, cls.GRID_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        return edge_grid, polarity_grid

    @staticmethod
    def _grid_mismatch(
        reference_edge_grid,
        test_edge_grid,
        reference_polarity_grid,
        test_polarity_grid,
    ) -> float:
        edge_delta = float(np.mean(np.abs(reference_edge_grid - test_edge_grid)))
        polarity_delta = float(
            np.mean(np.abs(reference_polarity_grid - test_polarity_grid))
        )
        return float(
            np.clip(0.62 * edge_delta / 0.32 + 0.38 * polarity_delta / 0.30, 0.0, 1.0)
        )

    @classmethod
    def _edge_evidence(cls, reference_edges: np.ndarray, test_edges: np.ndarray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reference_tolerant = cv2.dilate(reference_edges, kernel)
        test_tolerant = cv2.dilate(test_edges, kernel)
        missing = cv2.bitwise_and(reference_edges, cv2.bitwise_not(test_tolerant))
        extra = cv2.bitwise_and(test_edges, cv2.bitwise_not(reference_tolerant))
        matched = cv2.bitwise_and(reference_edges, test_tolerant)
        reference_count = max(float(cv2.countNonZero(reference_edges)), 1.0)
        test_count = max(float(cv2.countNonZero(test_edges)), 1.0)
        feature_loss = float(cv2.countNonZero(missing) / reference_count)
        extra_structure = float(cv2.countNonZero(extra) / test_count)
        return missing, extra, matched, feature_loss, extra_structure

    @classmethod
    def _signature_strength(cls, reference: np.ndarray, edges: np.ndarray) -> float:
        gray = cls._gray(reference)
        contrast = float(np.std(gray) / 52.0)
        edge_density = float(np.mean(edges > 0) / 0.13)
        dynamic_range = float(
            (np.percentile(gray, 92) - np.percentile(gray, 8)) / 100.0
        )
        return float(
            np.clip(0.42 * contrast + 0.38 * edge_density + 0.20 * dynamic_range, 0.0, 1.0)
        )

    @classmethod
    def _descriptor_similarity(cls, reference: np.ndarray, test: np.ndarray) -> float:
        if reference.shape != test.shape:
            test = cv2.resize(test, (reference.shape[1], reference.shape[0]))
        reference_edges = cls._edges(reference)
        test_edges = cls._edges(test)
        missing, extra, _, feature_loss, extra_structure = cls._edge_evidence(
            reference_edges,
            test_edges,
        )
        ref_edge_grid, ref_polarity = cls._grids(reference, reference_edges)
        test_edge_grid, test_polarity = cls._grids(test, test_edges)
        topology = cls._grid_mismatch(
            ref_edge_grid,
            test_edge_grid,
            ref_polarity,
            test_polarity,
        )
        reference_gray = cls._gray(reference).astype(np.float32)
        test_gray = cls._gray(test).astype(np.float32)
        correlation = 0.0
        if float(np.std(reference_gray)) > 2.0 and float(np.std(test_gray)) > 2.0:
            candidate = float(
                np.corrcoef(reference_gray.reshape(-1), test_gray.reshape(-1))[0, 1]
            )
            if np.isfinite(candidate):
                correlation = float(np.clip((candidate + 1.0) / 2.0, 0.0, 1.0))
        edge_similarity = float(np.clip(1.0 - 0.55 * feature_loss - 0.45 * extra_structure, 0.0, 1.0))
        topology_similarity = float(1.0 - topology)
        _ = missing, extra
        return float(
            np.clip(
                0.42 * edge_similarity
                + 0.36 * topology_similarity
                + 0.22 * correlation,
                0.0,
                1.0,
            )
        )

    @classmethod
    def _transform_evidence(cls, reference: np.ndarray, test: np.ndarray):
        height, width = reference.shape[:2]
        transforms = {
            "rot180": cv2.rotate(reference, cv2.ROTATE_180),
            "flip_horizontal": cv2.flip(reference, 1),
            "flip_vertical": cv2.flip(reference, 0),
            "rot90_clockwise": cv2.resize(
                cv2.rotate(reference, cv2.ROTATE_90_CLOCKWISE),
                (width, height),
                interpolation=cv2.INTER_AREA,
            ),
            "rot90_counterclockwise": cv2.resize(
                cv2.rotate(reference, cv2.ROTATE_90_COUNTERCLOCKWISE),
                (width, height),
                interpolation=cv2.INTER_AREA,
            ),
        }
        direct_similarity = cls._descriptor_similarity(reference, test)
        best_name = "none"
        best_similarity = 0.0
        for name, transformed in transforms.items():
            similarity = cls._descriptor_similarity(transformed, test)
            if similarity > best_similarity:
                best_name = name
                best_similarity = similarity
        gain = float(np.clip(best_similarity - direct_similarity, 0.0, 1.0))
        return direct_similarity, best_name, float(best_similarity), gain

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
        minimum_area = max(4, int(round(height * width * 0.0025)))
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
    def _anomaly_mask(
        cls,
        residual: np.ndarray,
        missing_mask: np.ndarray,
        extra_mask: np.ndarray,
        signature_strength: float,
    ) -> np.ndarray:
        residual_threshold = 0.31 if signature_strength >= 0.35 else 0.38
        color_mask = (residual >= residual_threshold).astype(np.uint8) * 255
        edge_mask = cv2.dilate(
            cv2.bitwise_or(missing_mask, extra_mask),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        combined = cv2.bitwise_or(color_mask, edge_mask)
        return cls._clean_mask(combined)

    @staticmethod
    def _score(
        feature_loss: float,
        topology_mismatch: float,
        orientation_mismatch: float,
        alternate_face_signal: float,
        signature_strength: float,
    ) -> float:
        score = (
            0.34 * feature_loss
            + 0.25 * topology_mismatch
            + 0.20 * orientation_mismatch
            + 0.21 * alternate_face_signal
        )
        score *= 0.68 + 0.32 * signature_strength
        if feature_loss > 0.66 and topology_mismatch > 0.44:
            score = max(score, 0.73)
        if alternate_face_signal > 0.68:
            score = max(score, 0.70)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _classification(
        is_defect: bool,
        signature_strength: float,
        feature_loss: float,
        topology_mismatch: float,
        orientation_mismatch: float,
        alternate_face_signal: float,
        transform_gain: float,
        best_transform_similarity: float,
    ) -> str:
        if not is_defect:
            return "ASSINATURA FRACA" if signature_strength < 0.22 else "ROI CONFORME"
        if transform_gain > 0.16 and best_transform_similarity > 0.56:
            return "ASSINATURA INVERTIDA FORTE"
        if alternate_face_signal > 0.66 and topology_mismatch > 0.38:
            return "FACE ALTERNATIVA PROVÁVEL"
        if orientation_mismatch > 0.48 and orientation_mismatch >= topology_mismatch:
            return "ORIENTAÇÃO DIVERGENTE"
        if feature_loss > 0.58:
            return "MARCA ESPERADA AUSENTE"
        return "ASSINATURA DA FACE DIVERGENTE"

    @staticmethod
    def _draw_orientation(
        image: np.ndarray,
        angle_degrees: float,
        strength: float,
        color,
    ) -> None:
        if strength < 0.10:
            return
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        length = max(8, int(min(width, height) * 0.34))
        radians = np.deg2rad(angle_degrees)
        delta_x = int(round(np.cos(radians) * length))
        delta_y = int(round(np.sin(radians) * length))
        start = (center[0] - delta_x, center[1] - delta_y)
        end = (center[0] + delta_x, center[1] + delta_y)
        cv2.arrowedLine(image, start, end, color, 1, cv2.LINE_AA, tipLength=0.20)

    @classmethod
    def _views(
        cls,
        reference: np.ndarray,
        test: np.ndarray,
        reference_edges: np.ndarray,
        test_edges: np.ndarray,
        matched_mask: np.ndarray,
        missing_mask: np.ndarray,
        extra_mask: np.ndarray,
        anomaly_mask: np.ndarray,
        expected_angle: float,
        observed_angle: float,
        expected_strength: float,
        observed_strength: float,
    ):
        reference_view = (reference.astype(np.float32) * 0.78).astype(np.uint8)
        test_view = (test.astype(np.float32) * 0.78).astype(np.uint8)
        reconstruction = (test.astype(np.float32) * 0.64).astype(np.uint8)

        reference_view[reference_edges > 0] = (0, 220, 255)
        test_view[test_edges > 0] = (255, 210, 40)
        reconstruction[matched_mask > 0] = (60, 190, 80)
        reconstruction[missing_mask > 0] = (0, 220, 255)
        reconstruction[extra_mask > 0] = (40, 55, 255)

        contours, _ = cv2.findContours(
            anomaly_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(reconstruction, contours, -1, (20, 20, 255), 1, cv2.LINE_AA)
        cls._draw_orientation(
            reference_view,
            expected_angle,
            expected_strength,
            (70, 235, 110),
        )
        cls._draw_orientation(
            test_view,
            observed_angle,
            observed_strength,
            (255, 215, 45),
        )
        cls._draw_orientation(
            reconstruction,
            expected_angle,
            expected_strength,
            (70, 235, 110),
        )
        cls._draw_orientation(
            reconstruction,
            observed_angle,
            observed_strength,
            (255, 215, 45),
        )
        return reference_view, test_view, reconstruction

    @staticmethod
    def _bounding_box(mask: np.ndarray, roi_box):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        height, width = mask.shape[:2]
        significant = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= max(4, height * width * 0.0025)
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
                return self._empty(False, "Motor reservado para a categoria INVERTIDO")
            if (
                full_reference is None
                or full_test is None
                or full_reference.size == 0
                or full_test.size == 0
            ):
                return self._empty(True, "Imagem nula")

            full_reference, full_test = self._safe_pair(full_reference, full_test)
            roi_box = self._roi_box(full_reference, aoi_epicenters)
            normalized_test = self._normalize_illumination(
                full_reference,
                full_test,
                roi_box,
            )
            reference = self._crop(full_reference, roi_box)
            test = self._crop(normalized_test, roi_box)
            raw_test = self._crop(full_test, roi_box)
            if reference.size == 0 or test.size == 0 or min(reference.shape[:2]) < 8:
                return self._empty(True, "ROI inválida ou pequena demais")

            reference_edges = self._edges(reference)
            test_edges = self._edges(test)
            (
                missing_mask,
                extra_mask,
                matched_mask,
                feature_loss,
                extra_structure,
            ) = self._edge_evidence(reference_edges, test_edges)

            reference_hist, expected_angle, expected_strength = self._orientation_histogram(
                reference
            )
            test_hist, observed_angle, observed_strength = self._orientation_histogram(test)
            orientation_mismatch = self._histogram_mismatch(reference_hist, test_hist)

            reference_edge_grid, reference_polarity_grid = self._grids(
                reference,
                reference_edges,
            )
            test_edge_grid, test_polarity_grid = self._grids(test, test_edges)
            topology_mismatch = self._grid_mismatch(
                reference_edge_grid,
                test_edge_grid,
                reference_polarity_grid,
                test_polarity_grid,
            )

            direct_similarity, best_transform, best_transform_similarity, transform_gain = (
                self._transform_evidence(reference, test)
            )
            signature_strength = self._signature_strength(reference, reference_edges)
            transform_signal = float(np.clip(transform_gain / 0.28, 0.0, 1.0))
            alternate_face_signal = float(
                np.clip(
                    0.44 * extra_structure
                    + 0.30 * topology_mismatch
                    + 0.16 * feature_loss
                    + 0.10 * transform_signal,
                    0.0,
                    1.0,
                )
            )

            score = self._score(
                feature_loss,
                topology_mismatch,
                orientation_mismatch,
                alternate_face_signal,
                signature_strength,
            )
            is_defect = bool(
                score > self.TOLERANCE
                and signature_strength > 0.12
                and (
                    feature_loss > 0.28
                    or topology_mismatch > 0.30
                    or orientation_mismatch > 0.40
                    or alternate_face_signal > 0.44
                )
            )
            classification = self._classification(
                is_defect,
                signature_strength,
                feature_loss,
                topology_mismatch,
                orientation_mismatch,
                alternate_face_signal,
                transform_gain,
                best_transform_similarity,
            )

            residual = self._local_residual(reference, test)
            anomaly_mask = self._anomaly_mask(
                residual,
                missing_mask,
                extra_mask,
                signature_strength,
            )
            changed_coverage = float(np.mean(anomaly_mask > 0))
            reference_view, test_view, reconstruction_view = self._views(
                reference,
                raw_test,
                reference_edges,
                test_edges,
                matched_mask,
                missing_mask,
                extra_mask,
                anomaly_mask,
                expected_angle,
                observed_angle,
                expected_strength,
                observed_strength,
            )
            bounding_box = self._bounding_box(anomaly_mask, roi_box) if is_defect else None

            if is_defect:
                reason = (
                    f"ASSINATURA DA FACE DIVERGENTE ({score:.0%}) • {classification}: "
                    f"marca esperada perdida {feature_loss:.0%}, topologia {topology_mismatch:.0%}, "
                    f"orientação {orientation_mismatch:.0%}, face alternativa {alternate_face_signal:.0%}"
                )
                if transform_gain > 0.10:
                    reason += (
                        f", transformação provável {best_transform} "
                        f"({best_transform_similarity:.0%})"
                    )
            else:
                reason = (
                    f"Assinatura da face compatível: similaridade {direct_similarity:.0%}, "
                    f"força da assinatura {signature_strength:.0%}"
                )

            height, width = reference.shape[:2]
            return {
                "inverted_active": True,
                "inverted_comparison_mode": self.MODE,
                "inverted_is_defect": is_defect,
                "inverted_score": score,
                "inverted_tolerance": self.TOLERANCE,
                "inverted_classification": classification,
                "inverted_signature_strength": signature_strength,
                "inverted_direct_similarity": direct_similarity,
                "inverted_feature_loss": feature_loss,
                "inverted_extra_structure": extra_structure,
                "inverted_topology_mismatch": topology_mismatch,
                "inverted_orientation_mismatch": orientation_mismatch,
                "inverted_alternate_face_signal": alternate_face_signal,
                "inverted_transform_gain": transform_gain,
                "inverted_best_transform": best_transform,
                "inverted_best_transform_similarity": best_transform_similarity,
                "inverted_expected_angle": expected_angle,
                "inverted_observed_angle": observed_angle,
                "inverted_expected_orientation_strength": expected_strength,
                "inverted_observed_orientation_strength": observed_strength,
                "inverted_changed_coverage": changed_coverage,
                "inverted_orientation_hist_reference": reference_hist.tolist(),
                "inverted_orientation_hist_test": test_hist.tolist(),
                "inverted_edge_grid_reference": reference_edge_grid.tolist(),
                "inverted_edge_grid_test": test_edge_grid.tolist(),
                "inverted_polarity_grid_reference": reference_polarity_grid.tolist(),
                "inverted_polarity_grid_test": test_polarity_grid.tolist(),
                "inverted_roi_box": roi_box,
                "inverted_roi_width": int(width),
                "inverted_roi_height": int(height),
                "inverted_expected_mask": reference_edges,
                "inverted_observed_mask": test_edges,
                "inverted_missing_mask": missing_mask,
                "inverted_extra_mask": extra_mask,
                "inverted_anomaly_mask": anomaly_mask,
                "inverted_residual_map": residual,
                "inverted_reference_view": reference_view,
                "inverted_test_view": test_view,
                "inverted_reconstruction_view": reconstruction_view,
                "inverted_bounding_box": bounding_box,
                "inverted_reason": reason,
            }
        except Exception as exc:
            print(f"Erro no InvertedFaceExpert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")


__all__ = ["InvertedFaceExpert"]
