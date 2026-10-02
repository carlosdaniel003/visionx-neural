"""Utilitários compartilhados pelos arquivos visuais OK/NG."""

from __future__ import annotations

from datetime import datetime
import re
import unicodedata


def safe_archive_category(value: str) -> str:
    """Converte a categoria para um trecho seguro de nome de arquivo."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_text.upper()).strip("_")
    return compact or "SEM_CATEGORIA"


def build_archive_filename(
    category: str,
    timestamp: datetime | None = None,
) -> str:
    """Mantém um único formato de nome para evidências OK e NG."""
    moment = timestamp or datetime.now()
    return (
        f"{moment:%Y-%m-%d}_{moment:%H%M}_"
        f"{safe_archive_category(category)}.png"
    )


__all__ = ["build_archive_filename", "safe_archive_category"]
