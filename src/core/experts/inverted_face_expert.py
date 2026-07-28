"""Compatibilidade de importação do motor INVERTIDO orientado à marca testemunha."""

import numpy as np

from src.core.experts.inverted_witness_expert import InvertedWitnessExpert


class InvertedFaceExpert(InvertedWitnessExpert):
    """Nome histórico preservado para integração, UI e testes existentes."""

    @classmethod
    def _relocation_evidence(
        cls,
        full_reference,
        full_test,
        roi_box,
        reference_saliency,
        witness_mask,
        direct_similarity,
    ):
        similarity, _, dx, dy = super()._relocation_evidence(
            full_reference,
            full_test,
            roi_box,
            reference_saliency,
            witness_mask,
            direct_similarity,
        )
        # A classe-base converteu [-1, 1] para [0, 1]. Retornamos à
        # correlação positiva real para que correlação neutra não vire 50%.
        positive_similarity = float(np.clip(2.0 * similarity - 1.0, 0.0, 1.0))
        gain = float(np.clip(positive_similarity - direct_similarity, 0.0, 1.0))
        return positive_similarity, gain, dx, dy


__all__ = ["InvertedFaceExpert"]
