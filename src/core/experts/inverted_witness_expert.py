"""Especialista de marca testemunha para a categoria INVERTIDO.

A ROI da AOI aponta uma marca visual estratégica da face correta: inscrição,
anel, linha, triângulo, canto ou outra estrutura local. O motor mede se essa
marca permaneceu na posição esperada, desapareceu, mudou de orientação ou foi
substituída por uma face alternativa.
"""

from __future__ import annotations

import cv2
import numpy as np


class InvertedWitnessExpert:
    MODE = "inverted_face_signature"
    CATEGORIES = frozenset({"INVERTIDO", "INVERTED", "REVERSE", "UP SIDE DOWN"})
    TOLERANCE = 0.43
    ORIENTATION_BINS = 12
    GRID_SIDE = 8

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
            "inverted_test_signature_strength": 0.0,
            "inverted_direct_similarity": 1.0,
            "inverted_witness_retention": 1.0,
            "inverted_witness_loss": 0.0,
            "inverted_feature_loss": 0.0,
            "inverted_extra_structure": 0.0,
            "inverted_topology_mismatch": 0.0,
            "inverted_orientation_mismatch": 0.0,
            "inverted_alternate_face_signal": 0.0,
            "inverted_transform_gain": 0.0,
            "inverted_best_transform": "none",
            "inverted_best_transform_similarity": 0.0,
            "inverted_relocation_similarity": 0.0,
            "inverted_relocation_gain": 0.0,
            "inverted_relocation_dx": 0.0,
            "inverted_relocation_dy": 0.0,
            "inverted_relocation_pixels": 0.0,
            "inverted_expected_angle": 0.0,
            "inverted_observed_angle": 0.0,
            "inverted_expected_orientation_strength": 0.0,
            "inverted_observed_orientation_strength": 0.0,
            "inverted_changed_coverage": 0.0,
            "inverted_witness_coverage": 0.0,
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
            "inverted_witness_mask": None,
            "inverted_retained_mask": None,
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
        padding_x = max(2, int(round(width * 0.08)))
        padding_y = max(2, int(round(height * 0.08)))
        mask[
            max(0, y - padding_y) : min(image_height, y + height + padding_y),
            max(0, x - padding_x) : min(image_width, x + width + padding_x),
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
    def _robust_unit(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
        scale = float(np.percentile(values, percentile))
        if scale <= 1e-7:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip(values / scale, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _visual_maps(cls, image: np.ndarray):
        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_u8 = cv2.GaussianBlur(gray_u8, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4)).apply(gray_u8)
        gray = gray.astype(np.float32) / 255.0

        sigma = max(1.2, min(gray.shape[:2]) / 11.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigma)
        local_contrast = cls._robust_unit(np.abs(gray - background))
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cls._robust_unit(cv2.magnitude(grad_x, grad_y))

        height, width = gray.shape[:2]
        interior = np.zeros((height, width), dtype=np.float32)
        margin_x = max(2, int(round(width * 0.05)))
        margin_y = max(2, int(round(height * 0.05)))
        if width > margin_x * 2 and height > margin_y * 2:
            interior[margin_y : height - margin_y, margin_x : width - margin_x] = 1.0
        else:
            interior[:] = 1.0

        saliency = (0.58 * gradient + 0.42 * local_contrast) * interior
        polarity = np.clip((gray - background) / 0.25, -1.0, 1.0) * interior
        return gray, saliency, gradient * interior, polarity, interior

    @staticmethod
    def _clean_mask(mask: np.ndarray, minimum_ratio: float = 0.0015) -> np.ndarray:
        mask = mask.astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        height, width = mask.shape[:2]
        minimum_area = max(3, int(round(height * width * minimum_ratio)))
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
    def _witness_mask(cls, saliency: np.ndarray, interior: np.ndarray) -> np.ndarray:
        values = saliency[interior > 0]
        if values.size == 0:
            return np.zeros(saliency.shape, dtype=np.uint8)
        threshold = max(0.22, float(np.percentile(values, 70)))
        mask = ((saliency >= threshold) & (interior > 0)).astype(np.uint8) * 255
        return cls._clean_mask(mask)

    @classmethod
    def _orientation_histogram(
        cls,
        polarity: np.ndarray,
        witness_mask: np.ndarray,
    ):
        grad_x = cv2.Sobel(polarity, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(polarity, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        positive = magnitude[(witness_mask > 0) & (magnitude > 1e-7)]
        histogram = np.zeros(cls.ORIENTATION_BINS, dtype=np.float32)
        if positive.size == 0:
            return histogram, 0.0, 0.0
        threshold = float(np.percentile(positive, 40))
        selected = (witness_mask > 0) & (magnitude >= max(threshold, 0.02))
        orientation = np.mod(angle, 180.0)
        bin_width = 180.0 / cls.ORIENTATION_BINS
        indices = np.floor(orientation[selected] / bin_width).astype(np.int32)
        indices = np.clip(indices, 0, cls.ORIENTATION_BINS - 1)
        np.add.at(histogram, indices, magnitude[selected])
        total = float(histogram.sum())
        if total <= 1e-8:
            return histogram, 0.0, 0.0
        histogram /= total
        dominant_index = int(np.argmax(histogram))
        return (
            histogram,
            float((dominant_index + 0.5) * bin_width),
            float(histogram[dominant_index]),
        )

    @staticmethod
    def _histogram_mismatch(reference_hist, test_hist, reference_strength, test_strength):
        if reference_strength < 0.11 or test_strength < 0.11:
            return 0.0
        denominator = float(np.linalg.norm(reference_hist) * np.linalg.norm(test_hist))
        if denominator <= 1e-8:
            return 0.0
        cosine = float(np.dot(reference_hist, test_hist) / denominator)
        return float(np.clip(1.0 - cosine, 0.0, 1.0))

    @classmethod
    def _witness_retention(
        cls,
        reference_saliency: np.ndarray,
        test_saliency: np.ndarray,
        reference_polarity: np.ndarray,
        test_polarity: np.ndarray,
        witness_mask: np.ndarray,
    ):
        selected = witness_mask > 0
        if not np.any(selected):
            ones = np.ones(reference_saliency.shape, dtype=np.float32)
            return 1.0, ones
        saliency_ratio = np.minimum(
            test_saliency / (reference_saliency + 0.08),
            1.0,
        )
        polarity_difference = np.abs(test_polarity - reference_polarity)
        polarity_similarity = np.exp(-polarity_difference / 0.35)
        retention_map = np.clip(
            0.60 * saliency_ratio + 0.40 * polarity_similarity,
            0.0,
            1.0,
        ).astype(np.float32)
        return float(np.mean(retention_map[selected])), retention_map

    @classmethod
    def _edge_evidence(
        cls,
        reference_gradient: np.ndarray,
        test_gradient: np.ndarray,
        expected_mask: np.ndarray,
        observed_mask: np.ndarray,
    ):
        reference_edges = (
            (reference_gradient > 0.28) & (expected_mask > 0)
        ).astype(np.uint8) * 255
        test_edges = (
            (test_gradient > 0.28) & (observed_mask > 0)
        ).astype(np.uint8) * 255
        distance = cv2.distanceTransform(
            (test_edges == 0).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        expected_values = distance[reference_edges > 0]
        edge_retention = (
            float(np.mean(expected_values <= 2.2))
            if expected_values.size
            else 1.0
        )
        tolerance = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reference_tolerant = cv2.dilate(reference_edges, tolerance)
        test_tolerant = cv2.dilate(test_edges, tolerance)
        missing = cv2.bitwise_and(reference_edges, cv2.bitwise_not(test_tolerant))
        extra = cv2.bitwise_and(test_edges, cv2.bitwise_not(reference_tolerant))
        matched = cv2.bitwise_and(reference_edges, test_tolerant)
        return reference_edges, test_edges, missing, extra, matched, edge_retention

    @classmethod
    def _grids(
        cls,
        witness_mask: np.ndarray,
        polarity: np.ndarray,
    ):
        edge_grid = cv2.resize(
            (witness_mask > 0).astype(np.float32),
            (cls.GRID_SIDE, cls.GRID_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        polarity_grid = cv2.resize(
            (polarity + 1.0) * 0.5,
            (cls.GRID_SIDE, cls.GRID_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        return edge_grid, polarity_grid

    @staticmethod
    def _grid_mismatch(reference_edge, test_edge, reference_polarity, test_polarity):
        edge_delta = float(np.mean(np.abs(reference_edge - test_edge)))
        polarity_delta = float(np.mean(np.abs(reference_polarity - test_polarity)))
        return float(
            np.clip(
                0.65 * edge_delta / 0.28 + 0.35 * polarity_delta / 0.22,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _masked_correlation(reference_gray, test_gray, mask):
        selected = mask > 0
        if int(np.count_nonzero(selected)) < 6:
            return 1.0
        reference_values = reference_gray[selected]
        test_values = test_gray[selected]
        if float(np.std(reference_values)) <= 1e-5 or float(np.std(test_values)) <= 1e-5:
            difference = float(np.mean(np.abs(reference_values - test_values)))
            return float(np.clip(1.0 - difference / 0.30, 0.0, 1.0))
        correlation = float(np.corrcoef(reference_values, test_values)[0, 1])
        if not np.isfinite(correlation):
            return 0.0
        return float(np.clip((correlation + 1.0) * 0.5, 0.0, 1.0))

    @classmethod
    def _descriptor_similarity(cls, reference: np.ndarray, test: np.ndarray) -> float:
        if reference.shape != test.shape:
            test = cv2.resize(
                test,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        ref_gray, ref_saliency, ref_gradient, ref_polarity, ref_interior = cls._visual_maps(reference)
        test_gray, test_saliency, test_gradient, test_polarity, test_interior = cls._visual_maps(test)
        expected = cls._witness_mask(ref_saliency, ref_interior)
        observed = cls._witness_mask(test_saliency, test_interior)
        retention, _ = cls._witness_retention(
            ref_saliency,
            test_saliency,
            ref_polarity,
            test_polarity,
            expected,
        )
        _, _, _, _, _, edge_retention = cls._edge_evidence(
            ref_gradient,
            test_gradient,
            expected,
            observed,
        )
        ref_edge_grid, ref_polarity_grid = cls._grids(expected, ref_polarity)
        test_edge_grid, test_polarity_grid = cls._grids(observed, test_polarity)
        topology = cls._grid_mismatch(
            ref_edge_grid,
            test_edge_grid,
            ref_polarity_grid,
            test_polarity_grid,
        )
        correlation = cls._masked_correlation(ref_gray, test_gray, expected)
        return float(
            np.clip(
                0.40 * retention
                + 0.25 * edge_retention
                + 0.20 * (1.0 - topology)
                + 0.15 * correlation,
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

    @classmethod
    def _relocation_evidence(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        roi_box,
        reference_saliency: np.ndarray,
        witness_mask: np.ndarray,
        direct_similarity: float,
    ):
        x, y, width, height = roi_box
        image_height, image_width = full_test.shape[:2]
        margin_x = max(5, int(round(width * 0.85)))
        margin_y = max(5, int(round(height * 0.85)))
        search_x1 = max(0, x - margin_x)
        search_y1 = max(0, y - margin_y)
        search_x2 = min(image_width, x + width + margin_x)
        search_y2 = min(image_height, y + height + margin_y)
        search_image = full_test[search_y1:search_y2, search_x1:search_x2]
        if (
            search_image.shape[0] < height
            or search_image.shape[1] < width
            or int(cv2.countNonZero(witness_mask)) < 5
        ):
            return 0.0, 0.0, 0.0, 0.0

        _, search_saliency, _, _, _ = cls._visual_maps(search_image)
        template = (reference_saliency * (witness_mask > 0)).astype(np.float32)
        if float(np.std(template)) <= 1e-5 or float(np.std(search_saliency)) <= 1e-5:
            return 0.0, 0.0, 0.0, 0.0
        result = cv2.matchTemplate(
            search_saliency.astype(np.float32),
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, maximum, _, location = cv2.minMaxLoc(result)
        similarity = float(np.clip((maximum + 1.0) * 0.5, 0.0, 1.0))
        best_x = search_x1 + int(location[0])
        best_y = search_y1 + int(location[1])
        dx = float(best_x - x)
        dy = float(best_y - y)
        pixels = float(np.hypot(dx, dy))
        minimum_offset = max(2.0, min(width, height) * 0.055)
        gain = (
            float(np.clip(similarity - direct_similarity, 0.0, 1.0))
            if pixels >= minimum_offset
            else 0.0
        )
        return similarity, gain, dx, dy

    @staticmethod
    def _score(
        witness_loss: float,
        feature_loss: float,
        topology_mismatch: float,
        orientation_mismatch: float,
        alternate_face_signal: float,
        signature_strength: float,
        relocation_signal: float,
    ) -> float:
        score = (
            0.40 * witness_loss
            + 0.22 * feature_loss
            + 0.17 * topology_mismatch
            + 0.09 * orientation_mismatch
            + 0.08 * alternate_face_signal
            + 0.04 * relocation_signal
        )
        score *= 0.76 + 0.24 * signature_strength
        if witness_loss > 0.55 and feature_loss > 0.34:
            score = max(score, 0.67)
        if witness_loss > 0.68:
            score = max(score, 0.72)
        if orientation_mismatch > 0.50 and witness_loss > 0.28:
            score = max(score, 0.62)
        if relocation_signal > 0.56:
            score = max(score, 0.68)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _classification(
        is_defect: bool,
        signature_strength: float,
        witness_loss: float,
        feature_loss: float,
        test_signature_strength: float,
        orientation_mismatch: float,
        alternate_face_signal: float,
        relocation_signal: float,
        transform_gain: float,
        best_transform_similarity: float,
    ) -> str:
        if not is_defect:
            return "ASSINATURA FRACA" if signature_strength < 0.20 else "ROI CONFORME"
        if relocation_signal > 0.56:
            return "MARCA TESTEMUNHA DESLOCADA"
        if transform_gain > 0.16 and best_transform_similarity > 0.56:
            return "ASSINATURA INVERTIDA FORTE"
        if witness_loss > 0.62 and test_signature_strength < signature_strength * 0.88:
            return "MARCA TESTEMUNHA AUSENTE"
        if orientation_mismatch > 0.48:
            return "ORIENTAÇÃO DA MARCA DIVERGENTE"
        if alternate_face_signal > 0.58:
            return "FACE ALTERNATIVA PROVÁVEL"
        if feature_loss > 0.52:
            return "MARCA ESPERADA AUSENTE"
        return "ASSINATURA DA FACE DIVERGENTE"

    @staticmethod
    def _paint(image, mask, color, alpha):
        output = image.astype(np.float32).copy()
        selected = mask > 0
        if np.any(selected):
            target = np.asarray(color, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + target * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_orientation(image, angle_degrees, strength, color):
        if strength < 0.11:
            return
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        length = max(8, int(min(width, height) * 0.31))
        radians = np.deg2rad(angle_degrees)
        dx = int(round(np.cos(radians) * length))
        dy = int(round(np.sin(radians) * length))
        cv2.arrowedLine(
            image,
            (center[0] - dx, center[1] - dy),
            (center[0] + dx, center[1] + dy),
            color,
            1,
            cv2.LINE_AA,
            tipLength=0.20,
        )

    @classmethod
    def _views(
        cls,
        reference,
        raw_test,
        witness_mask,
        observed_mask,
        retained_mask,
        missing_mask,
        extra_mask,
        anomaly_mask,
        expected_angle,
        observed_angle,
        expected_strength,
        observed_strength,
    ):
        reference_view = (reference.astype(np.float32) * 0.76).astype(np.uint8)
        test_view = (raw_test.astype(np.float32) * 0.76).astype(np.uint8)
        reconstruction = (raw_test.astype(np.float32) * 0.68).astype(np.uint8)
        reference_view = cls._paint(reference_view, witness_mask, (0, 220, 255), 0.48)
        test_view = cls._paint(test_view, observed_mask, (255, 205, 45), 0.36)
        reconstruction = cls._paint(reconstruction, retained_mask, (65, 190, 80), 0.55)
        reconstruction = cls._paint(reconstruction, missing_mask, (0, 220, 255), 0.86)
        reconstruction = cls._paint(reconstruction, extra_mask, (35, 55, 255), 0.88)
        contours, _ = cv2.findContours(
            anomaly_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(reconstruction, contours, -1, (20, 20, 255), 1, cv2.LINE_AA)
        cls._draw_orientation(reference_view, expected_angle, expected_strength, (70, 235, 110))
        cls._draw_orientation(test_view, observed_angle, observed_strength, (255, 215, 45))
        cls._draw_orientation(reconstruction, expected_angle, expected_strength, (70, 235, 110))
        cls._draw_orientation(reconstruction, observed_angle, observed_strength, (255, 215, 45))
        return reference_view, test_view, reconstruction

    @staticmethod
    def _bounding_box(mask, roi_box):
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
        x, y, box_width, box_height = cv2.boundingRect(np.vstack(significant))
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

            (
                reference_gray,
                reference_saliency,
                reference_gradient,
                reference_polarity,
                reference_interior,
            ) = self._visual_maps(reference)
            (
                test_gray,
                test_saliency,
                test_gradient,
                test_polarity,
                test_interior,
            ) = self._visual_maps(test)
            witness_mask = self._witness_mask(reference_saliency, reference_interior)
            observed_mask = self._witness_mask(test_saliency, test_interior)

            witness_retention, retention_map = self._witness_retention(
                reference_saliency,
                test_saliency,
                reference_polarity,
                test_polarity,
                witness_mask,
            )
            witness_loss = float(1.0 - witness_retention)
            (
                reference_edges,
                test_edges,
                edge_missing,
                edge_extra,
                edge_matched,
                edge_retention,
            ) = self._edge_evidence(
                reference_gradient,
                test_gradient,
                witness_mask,
                observed_mask,
            )
            edge_loss = float(1.0 - edge_retention)
            feature_loss = float(np.clip(0.62 * edge_loss + 0.38 * witness_loss, 0.0, 1.0))

            reference_edge_grid, reference_polarity_grid = self._grids(
                witness_mask,
                reference_polarity,
            )
            test_edge_grid, test_polarity_grid = self._grids(
                observed_mask,
                test_polarity,
            )
            topology_mismatch = self._grid_mismatch(
                reference_edge_grid,
                test_edge_grid,
                reference_polarity_grid,
                test_polarity_grid,
            )

            reference_hist, expected_angle, expected_strength = self._orientation_histogram(
                reference_polarity,
                witness_mask,
            )
            test_hist, observed_angle, observed_strength = self._orientation_histogram(
                test_polarity,
                observed_mask,
            )
            orientation_mismatch = self._histogram_mismatch(
                reference_hist,
                test_hist,
                expected_strength,
                observed_strength,
            )

            expected_selected = witness_mask > 0
            observed_selected = observed_mask > 0
            signature_strength = (
                float(np.mean(reference_saliency[expected_selected]))
                if np.any(expected_selected)
                else 0.0
            )
            test_signature_strength = (
                float(np.mean(test_saliency[observed_selected]))
                if np.any(observed_selected)
                else 0.0
            )
            witness_coverage = float(np.mean(expected_selected))

            tolerance_mask = cv2.dilate(
                witness_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            extra_mask = cv2.bitwise_and(observed_mask, cv2.bitwise_not(tolerance_mask))
            extra_structure = float(
                cv2.countNonZero(extra_mask) / max(cv2.countNonZero(observed_mask), 1)
            )

            missing_pixels = (
                (witness_mask > 0)
                & ((retention_map < 0.48) | (edge_missing > 0))
            )
            missing_mask = self._clean_mask(missing_pixels.astype(np.uint8) * 255)
            retained_mask = cv2.bitwise_and(
                witness_mask,
                cv2.bitwise_not(missing_mask),
            )

            direct_similarity, best_transform, best_transform_similarity, transform_gain = (
                self._transform_evidence(reference, test)
            )
            relocation_similarity, relocation_gain, relocation_dx, relocation_dy = (
                self._relocation_evidence(
                    full_reference,
                    normalized_test,
                    roi_box,
                    reference_saliency,
                    witness_mask,
                    direct_similarity,
                )
            )
            relocation_pixels = float(np.hypot(relocation_dx, relocation_dy))
            relocation_signal = float(
                np.clip(
                    0.65 * relocation_gain / 0.24
                    + 0.35 * min(1.0, relocation_pixels / max(min(reference.shape[:2]) * 0.32, 1.0)),
                    0.0,
                    1.0,
                )
                if relocation_gain > 0.04
                else 0.0
            )
            transform_signal = float(np.clip(transform_gain / 0.28, 0.0, 1.0))
            alternate_face_signal = float(
                np.clip(
                    0.34 * extra_structure
                    + 0.30 * witness_loss
                    + 0.20 * topology_mismatch
                    + 0.10 * transform_signal
                    + 0.06 * relocation_signal,
                    0.0,
                    1.0,
                )
            )

            score = self._score(
                witness_loss,
                feature_loss,
                topology_mismatch,
                orientation_mismatch,
                alternate_face_signal,
                signature_strength,
                relocation_signal,
            )
            is_defect = bool(
                score > self.TOLERANCE
                and signature_strength > 0.14
                and (
                    witness_loss > 0.30
                    or feature_loss > 0.30
                    or topology_mismatch > 0.34
                    or orientation_mismatch > 0.42
                    or alternate_face_signal > 0.46
                    or relocation_signal > 0.50
                )
            )
            classification = self._classification(
                is_defect,
                signature_strength,
                witness_loss,
                feature_loss,
                test_signature_strength,
                orientation_mismatch,
                alternate_face_signal,
                relocation_signal,
                transform_gain,
                best_transform_similarity,
            )

            residual = np.clip(
                0.60 * (1.0 - retention_map)
                + 0.25 * np.abs(reference_polarity - test_polarity)
                + 0.15 * np.abs(reference_saliency - test_saliency),
                0.0,
                1.0,
            ).astype(np.float32)
            color_mask = (residual > 0.34).astype(np.uint8) * 255
            anomaly_mask = self._clean_mask(
                cv2.bitwise_or(
                    color_mask,
                    cv2.bitwise_or(missing_mask, extra_mask),
                )
            )
            changed_coverage = float(np.mean(anomaly_mask > 0))
            reference_view, test_view, reconstruction_view = self._views(
                reference,
                raw_test,
                witness_mask,
                observed_mask,
                retained_mask,
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
                    f"MARCA TESTEMUNHA DIVERGENTE ({score:.0%}) • {classification}: "
                    f"retenção {witness_retention:.0%}, perda da marca {feature_loss:.0%}, "
                    f"topologia {topology_mismatch:.0%}, orientação {orientation_mismatch:.0%}"
                )
                if relocation_signal > 0.40:
                    reason += (
                        f", possível deslocamento X:{relocation_dx:+.1f}px "
                        f"Y:{relocation_dy:+.1f}px"
                    )
                elif transform_gain > 0.10:
                    reason += (
                        f", transformação provável {best_transform} "
                        f"({best_transform_similarity:.0%})"
                    )
            else:
                reason = (
                    f"Marca testemunha preservada: retenção {witness_retention:.0%}, "
                    f"similaridade direta {direct_similarity:.0%}"
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
                "inverted_test_signature_strength": test_signature_strength,
                "inverted_direct_similarity": direct_similarity,
                "inverted_witness_retention": witness_retention,
                "inverted_witness_loss": witness_loss,
                "inverted_feature_loss": feature_loss,
                "inverted_extra_structure": extra_structure,
                "inverted_topology_mismatch": topology_mismatch,
                "inverted_orientation_mismatch": orientation_mismatch,
                "inverted_alternate_face_signal": alternate_face_signal,
                "inverted_transform_gain": transform_gain,
                "inverted_best_transform": best_transform,
                "inverted_best_transform_similarity": best_transform_similarity,
                "inverted_relocation_similarity": relocation_similarity,
                "inverted_relocation_gain": relocation_gain,
                "inverted_relocation_dx": relocation_dx,
                "inverted_relocation_dy": relocation_dy,
                "inverted_relocation_pixels": relocation_pixels,
                "inverted_expected_angle": expected_angle,
                "inverted_observed_angle": observed_angle,
                "inverted_expected_orientation_strength": expected_strength,
                "inverted_observed_orientation_strength": observed_strength,
                "inverted_changed_coverage": changed_coverage,
                "inverted_witness_coverage": witness_coverage,
                "inverted_orientation_hist_reference": reference_hist.tolist(),
                "inverted_orientation_hist_test": test_hist.tolist(),
                "inverted_edge_grid_reference": reference_edge_grid.tolist(),
                "inverted_edge_grid_test": test_edge_grid.tolist(),
                "inverted_polarity_grid_reference": reference_polarity_grid.tolist(),
                "inverted_polarity_grid_test": test_polarity_grid.tolist(),
                "inverted_roi_box": roi_box,
                "inverted_roi_width": int(width),
                "inverted_roi_height": int(height),
                "inverted_expected_mask": witness_mask,
                "inverted_observed_mask": observed_mask,
                "inverted_witness_mask": witness_mask,
                "inverted_retained_mask": retained_mask,
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
            print(f"Erro no InvertedWitnessExpert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")


__all__ = ["InvertedWitnessExpert"]
