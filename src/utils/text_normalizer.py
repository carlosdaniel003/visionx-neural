# src/utils/text_normalizer.py
"""Normalização das categorias oficiais recebidas da AOI."""

from __future__ import annotations

import difflib
import re
import unicodedata


# Categorias canônicas exibidas e persistidas pelo VisionX.
CATEGORIES = (
    "INVERTIDO",
    "FALTANDO",
    "MUITO ADESIVO",
)

# Entradas antigas, traduções e erros recorrentes de OCR permanecem aceitos,
# mas sempre são convertidos para uma das três categorias canônicas.
ALIASES = {
    "INVERTIDO": "INVERTIDO",
    "INVERTED": "INVERTIDO",
    "REVERSE": "INVERTIDO",
    "UP SIDE DOWN": "INVERTIDO",
    "UPSIDE DOWN": "INVERTIDO",
    "UP SIDE DOAN": "INVERTIDO",
    "UP SIDE DOWM": "INVERTIDO",
    "FALTANDO": "FALTANDO",
    "MISSING": "FALTANDO",
    "MUSING": "FALTANDO",
    "MISSMG": "FALTANDO",
    "MUITO ADESIVO": "MUITO ADESIVO",
    "MUCH ADHESIVE": "MUITO ADESIVO",
    "MUCH ADHES1VE": "MUITO ADESIVO",
    "EXCESS ADHESIVE": "MUITO ADESIVO",
    "ADESIVO EM EXCESSO": "MUITO ADESIVO",
}


def _ascii_upper(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", without_accents)
    return " ".join(cleaned.upper().split())


def _valid_words(normalized_text: str) -> list[str]:
    words = []
    for token in normalized_text.split():
        alpha_count = sum(character.isalpha() for character in token)
        if alpha_count >= 3:
            words.append(token)
    return words


def _find_category(normalized_text: str) -> str:
    # Expressões maiores precisam ser avaliadas primeiro para evitar que uma
    # palavra isolada antecipe uma categoria composta.
    for alias in sorted(ALIASES, key=len, reverse=True):
        if alias in normalized_text:
            return ALIASES[alias]

    words = _valid_words(normalized_text)
    if not words:
        return "Unknown"

    alias_names = list(ALIASES)
    max_group = min(4, len(words))
    for group_size in range(max_group, 0, -1):
        for index in range(len(words) - group_size + 1):
            candidate = " ".join(words[index : index + group_size])
            matches = difflib.get_close_matches(
                candidate,
                alias_names,
                n=1,
                cutoff=0.70 if group_size > 1 else 0.74,
            )
            if matches:
                return ALIASES[matches[0]]
    return "Unknown"


def normalize_aoi_text(ocr_text: str):
    """Retorna ``(categoria_canônica, valor_normalizado)``.

    A categoria nunca inclui informações de depuração como
    ``Unknown - Testou: [...]``. Textos antigos em inglês continuam aceitos.
    """
    original = str(ocr_text or "").strip()
    if not original or original == "-":
        return "Unknown", original

    category = _find_category(_ascii_upper(original))
    if category == "Unknown":
        return "Unknown", original

    # Quando a entrada já veio contaminada pela mensagem antiga de debug,
    # não propagamos esse texto para o campo Valor/OCR.
    normalized_value = (
        category
        if "UNKNOWN" in original.upper() and "TESTOU" in original.upper()
        else original
    )
    return category, normalized_value
