"""Compatibilidade de importação para o motor FALTANDO orientado ao patch."""

import cv2
import numpy as np

from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert


class MissingComponentExpert(ROIPatchExpectationExpert):
    """Nome histórico preservado para o orquestrador e a interface."""

    @staticmethod
    def _palette_residual(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Mede se o teste ainda pertence à paleta cromática do patch.

        Em patches homogêneos, como um trecho do corpo preto do componente,
        pequenas mudanças de brilho não devem transformar toda a ROI em defeito.
        """
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = reference_lab.reshape(-1, 3)
        center = np.median(pixels, axis=0)
        mad = np.median(np.abs(pixels - center), axis=0) * 1.4826
        scale = np.maximum(mad * 2.6, np.asarray([14.0, 9.0, 9.0], dtype=np.float32))
        normalized = (test_lab - center.reshape(1, 1, 3)) / scale.reshape(1, 1, 3)
        distance = np.sqrt(np.mean(normalized * normalized, axis=2))
        return np.clip((distance - 0.75) / 3.2, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _residual_and_mask(cls, reference: np.ndarray, test: np.ndarray):
        color_residual = cls._local_color_residual(reference, test)
        edge_anomaly, missing_edges, extra_edges, edge_mismatch = cls._edge_metrics(
            reference,
            test,
        )
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        texture = float(np.std(reference_gray) / 64.0)
        edge_density = float(np.mean(cls._auto_edges(reference) > 0))
        homogeneous = texture < 0.48 and edge_density < 0.10

        if homogeneous:
            palette_residual = cls._palette_residual(reference, test)
            color_residual = np.minimum(color_residual, palette_residual)

        edge_layer = (edge_anomaly > 0).astype(np.float32)
        residual = np.clip(0.84 * color_residual + 0.16 * edge_layer, 0.0, 1.0)
        threshold = 0.29 if homogeneous else 0.31
        mask = (residual >= threshold).astype(np.uint8) * 255
        mask = cls._clean_mask(mask)
        return (
            residual,
            mask,
            missing_edges,
            extra_edges,
            edge_mismatch,
            texture,
            edge_density,
        )


__all__ = ["MissingComponentExpert"]
