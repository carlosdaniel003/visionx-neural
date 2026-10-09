"""Escopo explícito da CNN FALTANDO v2 como detector de ausência visual."""
from src.core.strict_category_memory import canonical_memory_category

VISUAL_MISSING_CATEGORIES = frozenset({
    "FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO",
})

def uses_faltando_v2(category: str) -> bool:
    """Nenhum adesivo ou categoria desconhecida passa por este motor."""
    return canonical_memory_category(category) in VISUAL_MISSING_CATEGORIES
