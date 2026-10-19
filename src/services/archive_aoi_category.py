"""Resolver a categoria de capturas arquivadas com o contrato OCR atual.

O ScreenMonitor atual entrega apenas info["category"]; info["value"] é vazio.
O valor legado continua aceito somente como fallback de compatibilidade.
Nunca usar o nome do PNG para inventar uma categoria ausente no OCR.
"""
from __future__ import annotations

from src.core.strict_category_memory import canonical_memory_category
from src.utils.text_normalizer import CATEGORIES, normalize_aoi_text

KNOWN_CATEGORIES = frozenset(
    canonical_memory_category(name) for name in CATEGORIES
)


def resolve_archive_aoi_category(
    info: dict, category_hint: str,
) -> tuple[str, str]:
    """Devolve (categoria validada, value normalizado) ou falha fechada."""
    if not isinstance(info, dict):
        raise ValueError("Metadados AOI inválidos")

    expected = canonical_memory_category(category_hint)
    if expected not in KNOWN_CATEGORIES:
        raise ValueError(f"Categoria de arquivo não reconhecida: {expected!r}")

    raw_category = str(info.get("category", "") or "").strip()
    raw_value = str(info.get("value", "") or "").strip()

    current, _ = normalize_aoi_text(raw_category)
    legacy, normalized_value = normalize_aoi_text(raw_value)
    current = canonical_memory_category(current)
    legacy = canonical_memory_category(legacy)

    current_ok = current in KNOWN_CATEGORIES
    legacy_ok = legacy in KNOWN_CATEGORIES
    if current_ok and legacy_ok and current != legacy:
        raise ValueError(
            f"Conflito no OCR: category={current!r}, value={legacy!r}"
        )

    # A leitura atual em ScreenMonitor é exclusivamente de category.
    # Aceitar o campo value apenas em capturas legadas sem categoria válida.
    detected = current if current_ok else legacy if legacy_ok else ""
    if not detected:
        raise ValueError(
            "Categoria AOI ausente ou ilegível no OCR "
            f"(category={raw_category!r}, value={raw_value!r})"
        )
    if detected != expected:
        raise ValueError(
            f"Categoria AOI {detected!r} não confere com nome {expected!r}"
        )
    return detected, normalized_value


__all__ = ["resolve_archive_aoi_category"]
