"""Motor dedicado à categoria FALTANDO.

Compara a mesma ROI do gabarito e do teste para medir quanto da estrutura,
aparência e ocupação esperadas do componente desapareceram.
"""

from __future__ import annotations

import cv2
import numpy as np


class MissingComponentExpert:
    MODE = "missing_component"
    CATEGORIES = frozenset({"FALTANDO", "MISSING"})
    TOLERANCE = 0.42

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
            "missing_structure_loss": 0.0,
            "missing_coverage": 0.0,
            "missing_appearance_loss": 0.0,
            "missing_background_exposure": 0.0,
            "missing_presence_retention": 1.0,
            "missing_alignment_score": 0.0,
            "missing_alignment_shift": (0.0, 0.0),
            "missing_roi_box": None,
            "missing_roi_width": 0,
            "missing_roi_height": 0,
            "component_expected_mask": None,
            "component_missing_mask": None,
            "component_matched_mask": None,
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
        return (
            reference[y1:y2, x1:x2].copy(),
            test[y1:y2, x1:x2].copy(),
            (x1, y1),
            (x1, y1, x2 - x1, y2 - y1),
        )

    @staticmethod
    def _align_test(reference: np.ndarray, test: np.ndarray):
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

        # O centro pode ter desaparecido. O alinhamento prioriza as bordas da ROI,
        # que representam o contexto estável em volta do componente.
        border_mask = np.full((height, width), 255, dtype=np.uint8)
        margin_x = max(3, int(width * 0.22))
        margin_y = max(3, int(height * 0.22))
        if width > margin_x * 2 and height > margin_y * 2:
            border_mask[margin_y : height - margin_y, margin_x : width - margin_x] = 0

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            70,
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
            if abs(dx) > width * 0.20 or abs(dy) > height * 0.20:
                return test.copy(), (0.0, 0.0), 0.0
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
    def _auto_edges(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(gray))
        lower = int(max(18, 0.62 * median))
        upper = int(min(235, max(lower + 20, 1.38 * median)))
        edges = cv2.Canny(gray, lower, upper)
        return cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

    @staticmethod
    def _central_prior(height: int, width: int) -> np.ndarray:
        yy, xx = np.indices((height, width), dtype=np.float32)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        nx = (xx - cx) / max(width * 0.46, 1.0)
        ny = (yy - cy) / max(height * 0.46, 1.0)
        distance = nx * nx + ny * ny
        return np.clip(1.0 - distance, 0.0, 1.0)

    @classmethod
    def _expected_component_mask(
        cls,
        reference: np.ndarray,
        reference_edges: np.ndarray,
    ) -> np.ndarray:
        height, width = reference.shape[:2]
        lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = max(2, int(round(min(height, width) * 0.10)))
        border_pixels = np.concatenate(
            [
                lab[:border, :, :].reshape(-1, 3),
                lab[-border:, :, :].reshape(-1, 3),
                lab[:, :border, :].reshape(-1, 3),
                lab[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(lab - background, axis=2) / 105.0
        prior = cls._central_prior(height, width)

        edge_body = cv2.dilate(
            reference_edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        candidate = (
            ((distance > 0.13) & (prior > 0.10))
            | ((edge_body > 0) & (prior > 0.05))
        ).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
        output = np.zeros_like(candidate)
        minimum_area = max(8, int(height * width * 0.008))
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            center_x, center_y = centroids[label]
            central = (
                abs(center_x - width / 2.0) <= width * 0.38
                and abs(center_y - height / 2.0) <= height * 0.38
            )
            if area >= minimum_area and central:
                output[labels == label] = 255

        if cv2.countNonZero(output) < minimum_area:
            fallback = (prior > 0.26).astype(np.uint8) * 255
            output = cv2.bitwise_and(candidate, fallback)
        return output

    @staticmethod
    def _appearance_loss(
        reference_gray: np.ndarray,
        test_gray: np.ndarray,
        expected_mask: np.ndarray,
    ) -> float:
        selected = expected_mask > 0
        if int(np.count_nonzero(selected)) < 8:
            return 0.0
        reference_values = reference_gray[selected].astype(np.float32)
        test_values = test_gray[selected].astype(np.float32)
        mean_difference = float(np.mean(np.abs(reference_values - test_values)) / 255.0)
        if np.std(reference_values) < 1e-5 or np.std(test_values) < 1e-5:
            correlation_loss = mean_difference
        else:
            correlation = float(np.corrcoef(reference_values, test_values)[0, 1])
            correlation_loss = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
        return float(np.clip(max(mean_difference * 1.7, correlation_loss), 0.0, 1.0))

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
        test: np.ndarray,
        expected_mask: np.ndarray,
        matched_mask: np.ndarray,
        missing_mask: np.ndarray,
    ):
        reference_view = cls._paint(
            reference,
            cv2.dilate(expected_mask, np.ones((3, 3), np.uint8)),
            (70, 200, 90),
            0.28,
        )
        test_view = cls._paint(
            test,
            cv2.dilate(matched_mask, np.ones((3, 3), np.uint8)),
            (255, 190, 30),
            0.55,
        )
        reconstruction = (test.astype(np.float32) * 0.70).astype(np.uint8)
        reconstruction = cls._paint(
            reconstruction,
            cv2.dilate(matched_mask, np.ones((3, 3), np.uint8)),
            (70, 190, 90),
            0.36,
        )
        reconstruction = cls._paint(
            reconstruction,
            cv2.dilate(missing_mask, np.ones((5, 5), np.uint8)),
            (35, 55, 255),
            0.92,
        )

        contours, _ = cv2.findContours(
            expected_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(reference_view, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.drawContours(reconstruction, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        missing_contours, _ = cv2.findContours(
            missing_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(reconstruction, missing_contours, -1, (20, 20, 255), 1, cv2.LINE_AA)
        return reference_view, test_view, reconstruction

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
            reference, test, offset, roi_box = self._extract_roi(
                full_reference,
                full_test,
                aoi_epicenters,
            )
            if reference.size == 0 or test.size == 0:
                return self._empty(True, "ROI inválida")
            height, width = reference.shape[:2]
            if min(height, width) < 9:
                return self._empty(True, "ROI pequena demais para presença do componente")

            aligned_test, alignment_shift, alignment_score = self._align_test(
                reference,
                test,
            )
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
            reference_edges = self._auto_edges(reference)
            test_edges = self._auto_edges(aligned_test)

            expected_mask = self._expected_component_mask(reference, reference_edges)
            expected_area = max(float(cv2.countNonZero(expected_mask)), 1.0)

            tolerance_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            test_edges_tolerant = cv2.dilate(test_edges, tolerance_kernel)
            reference_edges_in_component = cv2.bitwise_and(reference_edges, expected_mask)
            matched_edges = cv2.bitwise_and(
                reference_edges_in_component,
                test_edges_tolerant,
            )
            absent_edges = cv2.bitwise_and(
                reference_edges_in_component,
                cv2.bitwise_not(test_edges_tolerant),
            )
            reference_edge_count = max(
                float(cv2.countNonZero(reference_edges_in_component)),
                1.0,
            )
            structure_loss = float(cv2.countNonZero(absent_edges) / reference_edge_count)
            presence_retention = float(np.clip(1.0 - structure_loss, 0.0, 1.0))

            reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
            test_lab = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_difference = np.linalg.norm(test_lab - reference_lab, axis=2) / 115.0
            changed = (lab_difference > 0.12).astype(np.uint8) * 255
            missing_region = cv2.bitwise_and(changed, expected_mask)
            missing_region = cv2.bitwise_or(
                missing_region,
                cv2.dilate(absent_edges, tolerance_kernel),
            )
            missing_region = cv2.bitwise_and(missing_region, expected_mask)
            missing_region = cv2.morphologyEx(
                missing_region,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            missing_coverage = float(cv2.countNonZero(missing_region) / expected_area)

            matched_mask = cv2.bitwise_and(
                expected_mask,
                cv2.bitwise_not(missing_region),
            )
            appearance_loss = self._appearance_loss(
                reference_gray,
                test_gray,
                expected_mask,
            )

            border = max(2, int(round(min(height, width) * 0.10)))
            border_pixels = np.concatenate(
                [
                    test_lab[:border].reshape(-1, 3),
                    test_lab[-border:].reshape(-1, 3),
                    test_lab[:, :border].reshape(-1, 3),
                    test_lab[:, -border:].reshape(-1, 3),
                ],
                axis=0,
            )
            background = np.median(border_pixels, axis=0)
            reference_to_background = np.linalg.norm(reference_lab - background, axis=2)
            test_to_background = np.linalg.norm(test_lab - background, axis=2)
            exposed = (
                (test_to_background + 7.0 < reference_to_background)
                & (expected_mask > 0)
                & (lab_difference > 0.08)
            )
            background_exposure = float(np.count_nonzero(exposed) / expected_area)

            score = float(
                np.clip(
                    0.40 * structure_loss
                    + 0.30 * missing_coverage
                    + 0.20 * appearance_loss
                    + 0.10 * background_exposure,
                    0.0,
                    1.0,
                )
            )
            is_defect = bool(
                score > self.TOLERANCE
                and (missing_coverage > 0.13 or structure_loss > 0.46)
            )

            reference_view, test_view, reconstruction_view = self._build_views(
                reference,
                aligned_test,
                expected_mask,
                matched_mask,
                missing_region,
            )

            bounding_box = None
            contours, _ = cv2.findContours(
                missing_region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            significant = [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= max(4, height * width * 0.002)
            ]
            if significant:
                combined = np.vstack(significant)
                x, y, box_width, box_height = cv2.boundingRect(combined)
                offset_x, offset_y = offset
                margin = 4
                bounding_box = (
                    max(0, x + offset_x - margin),
                    max(0, y + offset_y - margin),
                    box_width + margin * 2,
                    box_height + margin * 2,
                )

            if is_defect:
                reason = (
                    f"COMPONENTE FALTANDO ({score:.0%}): estrutura ausente "
                    f"{structure_loss:.0%}, região perdida {missing_coverage:.0%}, "
                    f"aparência perdida {appearance_loss:.0%}"
                )
            elif score > 0.18:
                reason = (
                    f"Presença parcial dentro da tolerância: score {score:.0%}, "
                    f"retenção {presence_retention:.0%}"
                )
            else:
                reason = "Estrutura e aparência do componente preservadas"

            return {
                "missing_active": True,
                "missing_comparison_mode": self.MODE,
                "missing_is_defect": is_defect,
                "missing_score": score,
                "missing_tolerance": self.TOLERANCE,
                "missing_structure_loss": structure_loss,
                "missing_coverage": missing_coverage,
                "missing_appearance_loss": appearance_loss,
                "missing_background_exposure": background_exposure,
                "missing_presence_retention": presence_retention,
                "missing_alignment_score": float(alignment_score),
                "missing_alignment_shift": tuple(float(value) for value in alignment_shift),
                "missing_roi_box": roi_box,
                "missing_roi_width": int(width),
                "missing_roi_height": int(height),
                "component_expected_mask": expected_mask,
                "component_missing_mask": missing_region,
                "component_matched_mask": matched_mask,
                "missing_reference_view": reference_view,
                "missing_test_view": test_view,
                "missing_reconstruction_view": reconstruction_view,
                "missing_bounding_box": bounding_box if is_defect else None,
                "missing_reason": reason,
            }
        except Exception as exc:
            print(f"Erro no MissingComponentExpert: {exc}")
            return self._empty(True, f"Erro interno: {exc}")


__all__ = ["MissingComponentExpert"]
