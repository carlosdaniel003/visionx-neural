"""Semântica dos status visuais; não altera controle, inferência ou XP."""

from __future__ import annotations


STATUS_COLORS = {
    "neutral": "#a6a6a6",
    "waiting": "#f5c518",
    "active": "#69a9ff",
    "connected": "#4ade80",
    "error": "#ff7777",
    "ok": "#4ade80",
    "ng": "#ff7777",
}


def network_tone(message: str) -> str:
    """Escutar porta não prova conexão com a AOI nem captura recebida."""
    value = str(message or "").casefold()
    if any(word in value for word in (
        "erro", "falha", "desconect", "não conectado", "nao conectado",
        "timeout", "conexão perdida", "conexao perdida", "rejeitad",
        "❌",
    )):
        return "error"
    if any(word in value for word in (
        "aguard", "ouvindo", "esperando", "escutando", "porta",
        "sem conexão", "sem conexao", "alerta", "preparando", "⚡",
    )):
        return "waiting"
    if any(word in value for word in (
        "recebid", "conectado", "conectada", "conexão estabelecida",
        "conexao estabelecida", "transferindo",
    )):
        return "connected"
    return "neutral"


def inspection_tone(message: str, *, is_active: bool) -> str:
    value = str(message or "").casefold()
    if any(word in value for word in (
        "erro", "falha ao", "inválid", "invalid", "bloqueado",
    )):
        return "error"
    if any(word in value for word in (
        "aguard", "ocioso", "pausad", "pendente", "esperando",
    )):
        return "waiting"
    return "active" if is_active else "neutral"


def last_decision_status(label: str, source: str) -> tuple[str, str]:
    """Diferencia rótulo operacional automático de confirmação humana."""
    verdict = str(label or "").strip()
    origin = str(source or "").strip().casefold()
    if origin in {"auto", "production_auto"}:
        suffix = "automática, sem confirmação física"
    elif origin == "button":
        suffix = "operador ODIN"
    elif origin == "xp_keyboard":
        suffix = "operador XP"
    else:
        suffix = "origem não verificada"
    if any(word in verdict.casefold() for word in (
        "descart", "concordou",
    )):
        return f"Última ação: {verdict} ({suffix})", "neutral"
    if verdict == "OK":
        return f"Última decisão: OK ({suffix})", "ok"
    if verdict == "NG":
        return f"Última decisão: NG ({suffix})", "ng"
    return f"Último registro: {verdict or 'indisponível'} ({suffix})", "neutral"


__all__ = [
    "STATUS_COLORS", "network_tone", "inspection_tone",
    "last_decision_status",
]
