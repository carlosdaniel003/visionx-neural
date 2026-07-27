"""Regras puras de iconografia e sanitização dos textos visuais."""

from __future__ import annotations

import re


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2300-\u23FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)

ACTION_ICON_NAMES = {
    "capture",
    "discard",
    "approve",
    "defect",
    "light-left",
    "light-down",
    "light-right",
    "database-delete",
}

STATUS_ICON_NAMES = {
    "network",
    "processor",
    "history",
    "idle",
    "warning",
    "database",
    "approve",
    "defect",
    "light-right",
}

ALL_ICON_NAMES = ACTION_ICON_NAMES | STATUS_ICON_NAMES


def sanitize_visual_text(value: object) -> str:
    """Remove pictogramas Unicode e preserva o conteúdo textual legível."""
    text = str(value or "")
    text = text.replace("\ufe0f", "").replace("\u200d", "").replace("\u20e3", "")
    text = EMOJI_PATTERN.sub(" ", text)

    cleaned_lines = []
    for line in text.splitlines() or [text]:
        cleaned_lines.append(" ".join(line.split()))
    return "\n".join(cleaned_lines).strip()


def status_icon_name(slot: str, message: object, active: bool = False) -> str:
    """Escolhe um SVG conforme o papel do status e o conteúdo da mensagem."""
    raw = str(message or "")
    clean = sanitize_visual_text(raw)
    normalized = clean.casefold()

    error_tokens = (
        "erro",
        "falha",
        "defeito",
        "desconect",
        "recusad",
        "inválid",
        "invalido",
    )
    warning_tokens = (
        "alerta",
        "atenção",
        "atencao",
        "aguardando",
        "parcial",
    )

    contains_error_symbol = "\u274c" in raw or "\u26d4" in raw
    contains_warning_symbol = "\u26a0" in raw or "\u26a1" in raw

    if contains_error_symbol or any(token in normalized for token in error_tokens):
        return "defect"
    if contains_warning_symbol or any(token in normalized for token in warning_tokens):
        return "warning"

    slot_name = str(slot or "").casefold()
    if slot_name == "network":
        return "network"

    if slot_name == "history":
        if "dataset" in normalized or "memória" in normalized or "memoria" in normalized:
            return "database"
        if "ng" in normalized:
            return "defect"
        if "ok" in normalized or "salv" in normalized or "conclu" in normalized:
            return "approve"
        return "history"

    if "ilumina" in normalized or "luz" in normalized:
        return "light-right"
    if "dataset" in normalized or "memória" in normalized or "memoria" in normalized:
        return "database"
    if active or any(
        token in normalized
        for token in ("process", "captur", "recebend", "analis", "calcul")
    ):
        return "processor"
    if any(token in normalized for token in ("conclu", "sucesso", "pronto")):
        return "approve"
    return "idle"
