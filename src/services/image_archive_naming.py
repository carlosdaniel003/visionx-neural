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
    """Novo acervo neutro: nome identifica a LUZ, não a categoria AOI.

    O carregador continua aceitando nomes históricos FALTANDO, ADESIVO,
    EMBORCADO etc. O parâmetro category é mantido na API para preservar
    compatibilidade com os produtores antigos, mas seu conteúdo só define
    se a iluminação é TOP, MID ou SIDE.
    """
    moment = timestamp or datetime.now()
    normalized = safe_archive_category(category)
    light = next((
        mode for mode in ("SIDE", "TOP", "MID")
        if normalized == mode or normalized.endswith("_" + mode)
    ), "SIDE")
    return f"{moment:%Y-%m-%d}_{moment:%H%M}_AOI_{light}.png"


__all__ = ["build_archive_filename", "safe_archive_category"]
