"""Compatibilidade de importação para o motor FALTANDO orientado ao patch."""

from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert


class MissingComponentExpert(ROIPatchExpectationExpert):
    """Nome histórico preservado para o orquestrador e a interface."""


__all__ = ["MissingComponentExpert"]
