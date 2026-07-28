"""Motor de expectativa visual da ROI para a categoria FALTANDO.

A ROI da AOI representa uma expectativa visual exata. Ela pode indicar:
- uma região que deveria conter somente o fundo da placa; ou
- uma região que deveria conter uma estrutura/componente específico.

O motor detecta ocupação indevida, ausência, deslocamento, presença parcial e
estrutura/orientação divergente sem tentar transformar todo caso em simples
"componente ausente".
"""

from __future__ import annotations

import cv2
import numpy as np


class ROIExpectationExpert:
    MODE = "roi_expectation"
    CATEGORIES = frozenset({"FALTANDO", "MISSING"})
    TOLERANCE = 0.40
    BACKGROUND_TOLERANCE = 0.34

    @classmethod
    def _is_missing_context(cls, info: dict | None) -> bool:
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
            "missing_expectation_mode": "unknown",
            "missing_classification": "SEM DADOS",
            "missing_structure_loss": 0.0,
            "missing_extra_structure": 0.0,
            "missing_coverage": 0.0,
            "missing_changed_coverage": 0.0,
            "missing_appearance_loss": 0.0,
            "missing_background_exposure": 0.0,
            "missing_presence_retention": 1.0,
            "missing_direct_similarity": 1.0,
            "missing_best_similarity": 1.0,
            "missing_displacement_dx": 0.0,
            "missing_displacement_dy": 0.0,
            "missing_displacement_pixels": 0.0,
            "missing_displacement_pct": 0.0,
            "missing_reference_distinctness": 0.0,
            "missing_alignment_score": 0.0,
            "missing_alignment_shift": (0.0, 0.0),
            "missing_roi_box": None,
            "missing_roi_width": 0,
            "missing_roi_height": 0,
            "component_expected_mask": None,
            "component_missing_mask": None,
            "component_matched_mask": None,
            "roi_anomaly_mask": None,
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
    def _extract_roi(reference: np.ndarray, test: np.ndarray, epicenters):
        height, width = reference.shape[:2]
        x1, y1, x2, y2 = 0, 0, width, height
        if epicenters:
            x, y, roi_width, roi_height = epicenters[0]
            candidate_x1 = max(0, int(x))
            candidate_y1 = max(0, int(y))
            candidate_x2 = min(width, int(x + roi_width))
            candidate_y2 = min(height, int(y + roi_height))
            if candidate_x2 > candidate_x1 and candidate_y2 > candidate_y1:
                x1, y1, x2, y2 = (
                    candidate_x1,
                    candidate_y1,
                    candidate_x2,
                    candidate_y2,
                )
        box = (x1, y1, x2 - x1, y2 - y1)
        return (
            reference[y1:y2, x1:x2].copy(),
            test[y1:y2, x1:x2].copy(),
            (x1, y1),
            box,
        )

    @staticmethod
    def _ring_pixels(image: np.ndarray, roi_box, scale: float = 0.30):
        x, y, width, height = (int(value) for value in roi_box)
        image_height, image_width = image.shape[:2]
        margin_x = max(4, int(round(width * scale)))
        margin_y = max(4, int(round(height * scale)))
        outer_x1 = max(0, x - margin_x)
        outer_y1 = max(0, y - margin_y)
        outer_x2 = min(image_width, x + width + margin_x)
        outer_y2 = min(image_height, y + height + margin_y)
        outer = image[outer_y1:outer_y2, outer_x1:outer_x2]
        if outer.size == 0:
            return np.empty((0, 3), dtype=np.uint8)

        mask = np.ones(outer.shape[:2], dtype=np.uint8)
        inner_x1 = x - outer_x1
        inner_y1 = y - outer_y1
        inner_x2 = min(mask.shape[1], inner_x1 + width)
        inner_y2 = min(mask.shape[0], inner_y1 + height)
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        pixels = outer[mask > 0]
        return pixels.reshape(-1, 3) if pixels.size else np.empty((0, 3), dtype=np.uint8)

    @staticmethod
    def _auto_edges(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(gray))
        lower = int(max(16, 0.58 * median))
        upper = int(min(240, max(lower + 22, 1.42 * median)))
        edges = cv2.Canny(gray, lower, upper)
        return cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

    @staticmethod
    def _micro_align(reference: np.ndarray, test: np.ndarray):
        """Corrige somente jitter pequeno; deslocamento real continua sendo defeito."""
        height, width = reference.shape[:2]
        if min(height, width) < 16:
            return test.copy(), (0.0, 0.0), 0.0

        reference_gray = cv2.GaussianBlur(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (5, 5),
            0,
        )
        test_gray = cv2.GaussianBlur(
            cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            (5, 5),
            0,
        )
        border_mask = np.full((height, width), 255, dtype=np.uint8)
        margin_x = max(3, int(width * 0.24))
        margin_y = max(3, int(height * 0.24))
        if width > margin_x * 2 and height > margin_y * 2:
            border_mask[margin_y : height - margin_y, margin_x : width - margin_x] = 0

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            55,
            1e-5,
        )
        try:
            score, warp = cv2.findTransformECC(
                reference_gray,
                test_gray,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
                border_mask,
                3,
            )
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            jitter_limit = max(1.5, min(width, height) * 0.035)
            if abs(dx) > jitter_limit or abs(dy) > jitter_limit:
                return test.copy(), (0.0, 0.0), float(score)
            aligned = cv2.warpAffine(
                test,
                warp,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101,
            )
            return aligned, (dx, dy), float(score)
        except cv2.error:
            return test.copy(), (0.0, 0.0), 0.0

    @staticmethod
    def _illumination_normalize(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Compensa alteração global de luz sem remover diferenças estruturais."""
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        height, width = reference.shape[:2]
        border = max(2, int(round(min(height, width) * 0.12)))
        mask = np.zeros((height, width), dtype=bool)
        mask[:border] = True
        mask[-border:] = True
        mask[:, :border] = True
        mask[:, -border:] = True
        delta = np.median(reference_lab[mask] - test_lab[mask], axis=0)
        # Limita a correção para não converter uma troca real de material em luz.
        delta = np.clip(delta, [-18.0, -8.0, -8.0], [18.0, 8.0, 8.0])
        normalized = np.clip(test_lab + delta, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    @classmethod
    def _reference_expectation(
        cls,
        full_reference: np.ndarray,
        reference_roi: np.ndarray,
        roi_box,
    ):
        edges = cls._auto_edges(reference_roi)
        edge_density = float(np.mean(edges > 0))
        gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
        texture = float(np.std(gray) / 64.0)

        ring_bgr = cls._ring_pixels(full_reference, roi_box)
        roi_lab = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        if ring_bgr.size:
            ring_lab = cv2.cvtColor(
                ring_bgr.reshape(-1, 1, 3),
                cv2.COLOR_BGR2LAB,
            ).reshape(-1, 3).astype(np.float32)
            background = np.median(ring_lab, axis=0)
            color_distance = float(
                np.linalg.norm(np.median(roi_lab.reshape(-1, 3), axis=0) - background)
                / 95.0
            )
            ring_gray = cv2.cvtColor(
                ring_bgr.reshape(-1, 1, 3),
                cv2.COLOR_BGR2GRAY,
            ).reshape(-1)
            ring_texture = float(np.std(ring_gray) / 64.0)
            texture_contrast = abs(texture - ring_texture)
        else:
            background = np.median(roi_lab.reshape(-1, 3), axis=0)
            color_distance = 0.0
            texture_contrast = texture

        distinctness = float(
            np.clip(
                0.58 * color_distance
                + 0.24 * min(1.0, edge_density / 0.16)
                + 0.18 * min(1.0, texture_contrast),
                0.0,
                1.0,
            )
        )
        expected_background = bool(
            ring_bgr.size
            and distinctness < 0.19
            and edge_density < 0.10
            and texture < 0.48
        )
        return (
            "background" if expected_background else "structure",
            distinctness,
            background,
            edge_density,
        )

    @staticmethod
    def _expected_structure_mask(
        reference: np.ndarray,
        background_lab: np.ndarray,
        reference_edges: np.ndarray,
    ) -> np.ndarray:
        height, width = reference.shape[:2]
        lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
        edge_body = cv2.dilate(
            reference_edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        candidate = ((distance > 15.0) | (edge_body > 0)).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        minimum_area = max(8, int(height * width * 0.006))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        output = np.zeros_like(candidate)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                output[labels == label] = 255
        if cv2.countNonZero(output) < minimum_area:
            output[:] = 255
        return output

    @staticmethod
    def _difference_map(reference: np.ndarray, test: np.ndarray):
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        difference = np.linalg.norm(test_lab - reference_lab, axis=2) / 100.0
        difference = cv2.GaussianBlur(difference.astype(np.float32), (5, 5), 0)
        changed = (difference > 0.15).astype(np.uint8) * 255
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        return np.clip(difference, 0.0, 1.0), changed

    @classmethod
    def _structural_metrics(cls, reference: np.ndarray, test: np.ndarray):
        reference_edges = cls._auto_edges(reference)
        test_edges = cls._auto_edges(test)
        tolerance = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reference_tolerant = cv2.dilate(reference_edges, tolerance)
        test_tolerant = cv2.dilate(test_edges, tolerance)
        missing_edges = cv2.bitwise_and(reference_edges, cv2.bitwise_not(test_tolerant))
        extra_edges = cv2.bitwise_and(test_edges, cv2.bitwise_not(reference_tolerant))
        reference_count = max(float(cv2.countNonZero(reference_edges)), 1.0)
        test_count = max(float(cv2.countNonZero(test_edges)), 1.0)
        structure_loss = float(cv2.countNonZero(missing_edges) / reference_count)
        extra_structure = float(cv2.countNonZero(extra_edges) / test_count)
        union = reference_count + test_count
        mismatch = float(cv2.countNonZero(missing_edges) + cv2.countNonZero(extra_edges))
        edge_similarity = float(np.clip(1.0 - mismatch / max(union, 1.0), 0.0, 1.0))
        return (
            reference_edges,
            test_edges,
            missing_edges,
            extra_edges,
            structure_loss,
            extra_structure,
            edge_similarity,
        )

    @staticmethod
    def _appearance_similarity(reference: np.ndarray, test: np.ndarray, difference):
        mean_similarity = float(np.clip(1.0 - np.mean(difference) * 1.35, 0.0, 1.0))
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_std = float(np.std(reference_gray))
        test_std = float(np.std(test_gray))
        if ref_std > 2.0 and test_std > 2.0:
            correlation = float(np.corrcoef(reference_gray.reshape(-1), test_gray.reshape(-1))[0, 1])
            correlation = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
        else:
            correlation = mean_similarity
        return float(np.clip(0.62 * mean_similarity + 0.38 * correlation, 0.0, 1.0))

    @classmethod
    def _search_expected_structure(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        roi_box,
    ):
        x, y, width, height = (int(value) for value in roi_box)
        template = full_reference[y : y + height, x : x + width]
        image_height, image_width = full_test.shape[:2]
        margin_x = max(4, int(round(width * 0.50)))
        margin_y = max(4, int(round(height * 0.50)))
        search_x1 = max(0, x - margin_x)
        search_y1 = max(0, y - margin_y)
        search_x2 = min(image_width, x + width + margin_x)
        search_y2 = min(image_height, y + height + margin_y)
        search = full_test[search_y1:search_y2, search_x1:search_x2]
        if (
            template.size == 0
            or search.size == 0
            or search.shape[0] < template.shape[0]
            or search.shape[1] < template.shape[1]
        ):
            return 0.0, 0.0, 0.0, None

        template_gray = cv2.equalizeHist(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
        search_gray = cv2.equalizeHist(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY))
        scores = []
        locations = []

        if float(np.std(template_gray)) > 2.0:
            result_gray = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, gray_score, _, gray_location = cv2.minMaxLoc(result_gray)
            scores.append(max(0.0, float(gray_score)))
            locations.append(gray_location)

        template_edges = cls._auto_edges(template)
        search_edges = cls._auto_edges(search)
        if cv2.countNonZero(template_edges) >= 6:
            result_edges = cv2.matchTemplate(search_edges, template_edges, cv2.TM_CCOEFF_NORMED)
            _, edge_score, _, edge_location = cv2.minMaxLoc(result_edges)
            scores.append(max(0.0, float(edge_score)))
            locations.append(edge_location)

        if not scores:
            return 0.0, 0.0, 0.0, None
        best_index = int(np.argmax(scores))
        best_location = locations[best_index]
        best_score = float(np.clip(np.mean(scores), 0.0, 1.0))
        best_x = search_x1 + int(best_location[0])
        best_y = search_y1 + int(best_location[1])
        dx = float(best_x - x)
        dy = float(best_y - y)
        return best_score, dx, dy, (best_x, best_y, width, height)

    @staticmethod
    def _background_exposure(
        reference: np.ndarray,
        test: np.ndarray,
        expected_mask: np.ndarray,
        background_lab: np.ndarray,
    ) -> float:
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_distance = np.linalg.norm(reference_lab - background_lab.reshape(1, 1, 3), axis=2)
        test_distance = np.linalg.norm(test_lab - background_lab.reshape(1, 1, 3), axis=2)
        expected = expected_mask > 0
        exposed = expected & (reference_distance > 18.0) & (test_distance + 7.0 < reference_distance)
        expected_area = max(int(np.count_nonzero(expected)), 1)
        return float(np.count_nonzero(exposed) / expected_area)

    @staticmethod
    def _paint(image: np.ndarray, mask: np.ndarray, color, alpha: float):
        output = image.astype(np.float32).copy()
        selected = mask > 0
        if np.any(selected):
            paint = np.asarray(color, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + paint * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @classmethod
    def _build_views(
        cls,
        reference: np.ndarray,
        raw_test: np.ndarray,
        expectation_mode: str,
        expected_mask: np.ndarray,
        matched_mask: np.ndarray,
        anomaly_mask: np.ndarray,
        displacement,
        best_similarity: float,
    ):
        reference_view = reference.copy()
        test_view = raw_test.copy()
        reconstruction = (raw_test.astype(np.float32) * 0.72).astype(np.uint8)

        if expectation_mode == "structure":
            reference_view = cls._paint(reference_view, expected_mask, (70, 190, 90), 0.25)
            test_view = cls._paint(test_view, matched_mask, (255, 185, 35), 0.42)
            reconstruction = cls._paint(reconstruction, matched_mask, (70, 190, 90), 0.34)
        else:
            stable = cv2.bitwise_not(anomaly_mask)
            reference_view = cls._paint(reference_view, stable, (70, 190, 90), 0.12)

        reconstruction = cls._paint(reconstruction, anomaly_mask, (35, 55, 255), 0.90)
        test_view = cls._paint(test_view, anomaly_mask, (255, 185, 35), 0.26)

        boundary_mask = expected_mask if expectation_mode == "structure" else np.full(anomaly_mask.shape, 255, np.uint8)
        contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(reference_view, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.drawContours(reconstruction, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        anomaly_contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(reconstruction, anomaly_contours, -1, (20, 20, 255), 1, cv2.LINE_AA)

        dx, dy = displacement
        if expectation_mode == "structure" and best_similarity >= 0.45 and np.hypot(dx, dy) >= 2.0:
            center = (reference.shape[1] // 2, reference.shape[0] // 2)
            destination = (
                int(np.clip(center[0] + dx, 0, reference.shape[1] - 1)),
                int(np.clip(center[1] + dy, 0, reference.shape[0] - 1)),
            )
            cv2.arrowedLine(reconstruction, center, destination, (255, 210, 30), 2, cv2.LINE_AA, tipLength=0.22)
        return reference_view, test_view, reconstruction

    @staticmethod
    def _bounding_box(mask: np.ndarray, offset):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape[:2]
        significant = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= max(4, height * width * 0.002)
        ]
        if not significant:
            return None
        combined = np.vstack(significant)
        x, y, box_width, box_height = cv2.boundingRect(combined)
        offset_x, offset_y = offset
        margin = 4
        return (
            max(0, x + offset_x - margin),
            max(0, y + offset_y - margin),
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
            if not self._is_missing_context(aoi_info):
                return self._empty(False, "Motor reservado para a categoria FALTANDO")
            if (
                full_reference is None
                or full_test is None
                or full_reference.size == 0
                or full_test.size == 0
            ):
                return self._empty(True, "Imagem nula")

            full_reference, full_test = self._safe_pair(full_reference, full_test)
            reference, raw_test, offset, roi_box = self._extract_roi(
                full_reference,
                full_test,
                aoi_epicenters,
            )
            if reference.size == 0 or raw_test.size == 0:
                return self._empty(True, "ROI inválida")
            height, width = reference.shape[:2]
            if min(height, width) < 9:
                return self._empty(True, "ROI pequena demais para comparação")

            expectation_mode, distinctness, background_lab, _ = self._reference_expectation(
                full_reference,
                reference,
                roi_box,
            )
            aligned_test, alignment_shift, alignment_score = self._micro_align(reference, raw_test)
            normalized_test = self._illumination_normalize(reference, aligned_test)
            difference, changed_mask = self._difference_map(reference, normalized_test)
            changed_coverage = float(np.mean(changed_mask > 0))
            appearance_similarity = self._appearance_similarity(reference, normalized_test, difference)
            appearance_loss = float(1.0 - appearance_similarity)

            (
                reference_edges,
                _,
                missing_edges,
                extra_edges,
                structure_loss,
                extra_structure,
                edge_similarity,
            ) = self._structural_metrics(reference, normalized_test)
            direct_similarity = float(
                np.clip(0.58 * appearance_similarity + 0.42 * edge_similarity, 0.0, 1.0)
            )

            expected_mask = np.full((height, width), 255, dtype=np.uint8)
            best_similarity = direct_similarity
            displacement_dx = 0.0
            displacement_dy = 0.0
            best_box = None
            background_exposure = 0.0
            presence_retention = float(np.clip(1.0 - changed_coverage, 0.0, 1.0))

            if expectation_mode == "structure":
                expected_mask = self._expected_structure_mask(
                    reference,
                    background_lab,
                    reference_edges,
                )
                expected_area = max(float(cv2.countNonZero(expected_mask)), 1.0)
                anomaly_mask = cv2.bitwise_and(changed_mask, expected_mask)
                anomaly_mask = cv2.bitwise_or(
                    anomaly_mask,
                    cv2.bitwise_and(
                        cv2.dilate(missing_edges, np.ones((3, 3), np.uint8)),
                        expected_mask,
                    ),
                )
                changed_coverage = float(cv2.countNonZero(anomaly_mask) / expected_area)
                presence_retention = float(np.clip(1.0 - changed_coverage, 0.0, 1.0))
                background_exposure = self._background_exposure(
                    reference,
                    normalized_test,
                    expected_mask,
                    background_lab,
                )
                best_similarity, displacement_dx, displacement_dy, best_box = (
                    self._search_expected_structure(
                        full_reference,
                        full_test,
                        roi_box,
                    )
                )
                displacement_pixels = float(np.hypot(displacement_dx, displacement_dy))
                displacement_pct = float(
                    displacement_pixels / max(float(np.hypot(width, height)), 1.0)
                )
                shift_strength = float(np.clip(displacement_pct / 0.22, 0.0, 1.0))

                absence_score = float(
                    np.clip(
                        0.44 * background_exposure
                        + 0.34 * changed_coverage
                        + 0.22 * structure_loss,
                        0.0,
                        1.0,
                    )
                )
                displacement_score = 0.0
                if best_similarity >= 0.43 and displacement_pixels >= max(2.0, min(width, height) * 0.055):
                    displacement_score = float(
                        np.clip(
                            0.46 * shift_strength
                            + 0.29 * (1.0 - direct_similarity)
                            + 0.25 * best_similarity,
                            0.0,
                            1.0,
                        )
                    )
                divergence_score = float(
                    np.clip(
                        0.38 * structure_loss
                        + 0.34 * appearance_loss
                        + 0.28 * changed_coverage,
                        0.0,
                        1.0,
                    )
                )
                score = max(absence_score, displacement_score, divergence_score)
                is_defect = bool(
                    score > self.TOLERANCE
                    and (
                        changed_coverage > 0.16
                        or structure_loss > 0.34
                        or displacement_score > self.TOLERANCE
                    )
                )

                if not is_defect:
                    classification = "ROI CONFORME"
                elif background_exposure > 0.48 and changed_coverage > 0.42:
                    classification = "COMPONENTE AUSENTE"
                elif displacement_score >= max(absence_score, divergence_score) and displacement_pixels >= 2.0:
                    classification = "COMPONENTE DESLOCADO"
                elif 0.22 < presence_retention < 0.78:
                    classification = "PRESENÇA PARCIAL"
                else:
                    classification = "ESTRUTURA/ORIENTAÇÃO DIVERGENTE"
            else:
                anomaly_mask = cv2.bitwise_or(
                    changed_mask,
                    cv2.dilate(extra_edges, np.ones((3, 3), np.uint8)),
                )
                changed_coverage = float(np.mean(anomaly_mask > 0))
                intrusion_strength = float(
                    np.clip(
                        0.42 * changed_coverage
                        + 0.30 * extra_structure
                        + 0.28 * appearance_loss,
                        0.0,
                        1.0,
                    )
                )
                score = intrusion_strength
                is_defect = bool(
                    score > self.BACKGROUND_TOLERANCE
                    and (changed_coverage > 0.12 or extra_structure > 0.24)
                )
                classification = "OCUPAÇÃO INDEVIDA" if is_defect else "ROI CONFORME"
                displacement_pixels = 0.0
                displacement_pct = 0.0

            matched_mask = cv2.bitwise_and(expected_mask, cv2.bitwise_not(anomaly_mask))
            reference_view, test_view, reconstruction_view = self._build_views(
                reference,
                raw_test,
                expectation_mode,
                expected_mask,
                matched_mask,
                anomaly_mask,
                (displacement_dx, displacement_dy),
                best_similarity,
            )
            bounding_box = self._bounding_box(anomaly_mask, offset) if is_defect else None

            expectation_label = "FUNDO LIVRE" if expectation_mode == "background" else "ESTRUTURA ESPERADA"
            if is_defect:
                reason = (
                    f"QUEBRA DA EXPECTATIVA DA ROI ({score:.0%}) • {classification}: "
                    f"esperado {expectation_label.lower()}, divergência {changed_coverage:.0%}, "
                    f"estrutura ausente {structure_loss:.0%}, estrutura extra {extra_structure:.0%}"
                )
                if displacement_pixels >= 2.0:
                    reason += (
                        f", deslocamento X:{displacement_dx:+.1f}px Y:{displacement_dy:+.1f}px"
                    )
            else:
                reason = (
                    f"ROI conforme ao gabarito: esperado {expectation_label.lower()}, "
                    f"similaridade direta {direct_similarity:.0%}"
                )

            return {
                "missing_active": True,
                "missing_comparison_mode": self.MODE,
                "missing_is_defect": is_defect,
                "missing_score": float(score),
                "missing_tolerance": (
                    self.BACKGROUND_TOLERANCE
                    if expectation_mode == "background"
                    else self.TOLERANCE
                ),
                "missing_expectation_mode": expectation_mode,
                "missing_classification": classification,
                "missing_structure_loss": float(structure_loss),
                "missing_extra_structure": float(extra_structure),
                "missing_coverage": float(changed_coverage),
                "missing_changed_coverage": float(changed_coverage),
                "missing_appearance_loss": float(appearance_loss),
                "missing_background_exposure": float(background_exposure),
                "missing_presence_retention": float(presence_retention),
                "missing_direct_similarity": float(direct_similarity),
                "missing_best_similarity": float(best_similarity),
                "missing_displacement_dx": float(displacement_dx),
                "missing_displacement_dy": float(displacement_dy),
                "missing_displacement_pixels": float(displacement_pixels),
                "missing_displacement_pct": float(displacement_pct),
                "missing_reference_distinctness": float(distinctness),
                "missing_alignment_score": float(alignment_score),
                "missing_alignment_shift": tuple(float(value) for value in alignment_shift),
                "missing_roi_box": roi_box,
                "missing_roi_width": int(width),
                "missing_roi_height": int(height),
                "component_expected_mask": expected_mask,
                "component_missing_mask": anomaly_mask,
                "component_matched_mask": matched_mask,
                "roi_anomaly_mask": anomaly_mask,
                "missing_reference_view": reference_view,
                "missing_test_view": test_view,
                "missing_reconstruction_view": reconstruction_view,
                "missing_best_match_box": best_box,
                "missing_bounding_box": bounding_box,
                "missing_reason": reason,
            }
        except Exception as exc:
            print(f"Erro no ROIExpectationExpert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")


__all__ = ["ROIExpectationExpert"]
