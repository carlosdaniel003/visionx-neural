"""Compatibilidade de importação para o motor FALTANDO orientado à ROI."""

import cv2
import numpy as np

from src.core.experts.roi_expectation_expert import ROIExpectationExpert


class MissingComponentExpert(ROIExpectationExpert):
    """Nome histórico preservado para o orquestrador e testes existentes."""

    @staticmethod
    def _expected_structure_mask(
        reference: np.ndarray,
        background_lab: np.ndarray,
        reference_edges: np.ndarray,
    ) -> np.ndarray:
        """Implementação compatível com as versões atuais do OpenCV Python."""
        height, width = reference.shape[:2]
        lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        distance = np.linalg.norm(
            lab - background_lab.reshape(1, 1, 3),
            axis=2,
        )
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
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
        output = np.zeros_like(candidate)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
                output[labels == label] = 255
        if cv2.countNonZero(output) < minimum_area:
            output[:] = 255
        return output

    @staticmethod
    def _appearance_similarity(
        reference: np.ndarray,
        test: np.ndarray,
        difference,
    ) -> float:
        mean_similarity = float(
            np.clip(1.0 - np.mean(difference) * 1.35, 0.0, 1.0)
        )
        reference_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        test_gray = cv2.cvtColor(
            test,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        ref_std = float(np.std(reference_gray))
        test_std = float(np.std(test_gray))
        correlation = mean_similarity
        if ref_std > 2.0 and test_std > 2.0:
            candidate = float(
                np.corrcoef(
                    reference_gray.reshape(-1),
                    test_gray.reshape(-1),
                )[0, 1]
            )
            if np.isfinite(candidate):
                correlation = float(
                    np.clip((candidate + 1.0) / 2.0, 0.0, 1.0)
                )
        return float(
            np.clip(
                0.62 * mean_similarity + 0.38 * correlation,
                0.0,
                1.0,
            )
        )


__all__ = ["MissingComponentExpert"]
