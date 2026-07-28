"""Compatibilidade de importação do motor INVERTIDO orientado à marca testemunha."""

from src.core.experts.inverted_witness_expert import InvertedWitnessExpert


class InvertedFaceExpert(InvertedWitnessExpert):
    """Nome histórico preservado para integração, UI e testes existentes."""


__all__ = ["InvertedFaceExpert"]
