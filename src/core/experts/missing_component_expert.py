"""Compatibilidade de importação para o motor FALTANDO orientado à ROI."""

from src.core.experts.roi_expectation_expert import ROIExpectationExpert


class MissingComponentExpert(ROIExpectationExpert):
    """Nome histórico preservado para o orquestrador e testes existentes."""


__all__ = ["MissingComponentExpert"]
