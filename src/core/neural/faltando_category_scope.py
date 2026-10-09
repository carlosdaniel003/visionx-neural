"""Categorias da CNN FALTANDO v2, sem importar OpenCV/PyTorch.

Escopo idêntico aos aliases conhecidos pelo roteador KNN, mas sem depender
de módulos de análise de imagem: o gate de Produção deve ser leve.
"""
from __future__ import annotations

import re

_ABSENCE_ALIASES = frozenset({
    "FALTANDO", "MISSING", "MUSING", "MISSMG",
    "EMBORCADO", "EMBORCADA", "TOMBSTONE", "TOMBSTONED", "STANDING",
    "INVERTIDO", "INVERTED", "REVERSE", "UPSIDEDOWN",
    "DESLOCADO", "DDESLOCADO", "DESLOCAMENTO",
    "SHIFTED", "MISALIGNED", "OFFSET",
})


def uses_faltando_v2(category: str) -> bool:
    """Adesivo e categorias não previstas não podem entrar nesta CNN."""
    compact = re.sub(r"[^A-Z0-9]", "", str(category or "").upper())
    return compact in _ABSENCE_ALIASES


__all__ = ["uses_faltando_v2"]
