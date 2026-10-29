"""Exceção experimental, auditável, para o padrão OK TOP/MID já julgado.

Não conclui ausência/presença física e não reclassifica a sonda. Apenas
identifica uma contraprova de qualidade limitada, com discordância entre
as duas escalas e aparagem mínima de TODOS os pares. A decisão de XP
continua condicionada à memória visual histórica validada e ao controlador
de Produção, que verifica novamente a política antes de enviar.
"""
from __future__ import annotations

import math

from src.core.physical_missing_veto import POLICY as PHYSICAL_PROBE_POLICY

LIGHTS = ("SIDE", "TOP", "MID")
FRACTIONS = (0.58, 0.76)
# Limitação explícita da exceção experimental: a região que disparou a
# suspeita deve ter baixa distinção estrutural da referência. Não é uma
# nova classificação física nem um limite geral de qualidade industrial.
MAX_SUSPECT_REFERENCE_DISTINCTNESS = 0.35


def _real_unit(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def _minimal_trim(geometry: object) -> bool:
    if not isinstance(geometry, dict):
        return False
    if (
        geometry.get("method") != "center_trim_no_rescale"
        or geometry.get("requires_review") is not True
    ):
        return False
    ref = geometry.get("reference_hw")
    test = geometry.get("test_hw")
    comp = geometry.get("comparison_hw")
    if (
        not isinstance(ref, list) or not isinstance(test, list)
        or not isinstance(comp, list)
        or any(len(pair) != 2 for pair in (ref, test, comp))
        or any(type(n) is not int or n < 32 for pair in (ref, test, comp) for n in pair)
    ):
        return False
    if comp != [min(ref[0], test[0]), min(ref[1], test[1])]:
        return False
    diff = [abs(a-b) for a, b in zip(ref, test)]
    return (
        any(diff)
        and all(d <= 3 and d <= min(r, t) * 0.02 for d, r, t in zip(diff, ref, test))
    )


def _verified_scales(light_data: object) -> list[dict] | None:
    if not isinstance(light_data, dict):
        return None
    rows = light_data.get("scales")
    if not isinstance(rows, list) or len(rows) != 2:
        return None
    normalized = []
    for row, fraction in zip(rows, FRACTIONS):
        if not isinstance(row, dict):
            return None
        if _real_unit(row.get("focus_fraction")) != fraction:
            return None
        if type(row.get("suspected_absence")) is not bool:
            return None
        if not isinstance(row.get("classification"), str):
            return None
        # Campos vindos da medição física independente: um status solto,
        # inserido no relatório sem ROI/métricas, nunca habilita 0/OK.
        fields = (
            "score", "coverage", "background", "distinctness",
            "direct_similarity", "nearby_similarity",
        )
        if any(_real_unit(row.get(name)) is None for name in fields):
            return None
        normalized.append(row)
    return normalized


def historical_ok_mid_probe_disagreement(audit: object) -> bool:
    """Admite APENAS a discordância conhecida, não uma ausência confirmada.

    A função é parte de uma política experimental de Produção. A auditoria
    permanece SUSPECT_ABSENCE e deve continuar exposta ao operador.
    """
    if not isinstance(audit, dict):
        return False
    if (
        audit.get("policy") != PHYSICAL_PROBE_POLICY
        or audit.get("diagnostic_only") is not True
        or audit.get("review_required") is not True
        or audit.get("authorizes_ok") is not False
        or audit.get("authorizes_ng") is not False
        or audit.get("status") != "SUSPECT_ABSENCE"
        or audit.get("suspect_lights") != ["MID"]
        or audit.get("geometry_review_lights") != ["SIDE", "TOP"]
        or audit.get("unavailable_lights") != []
    ):
        return False
    lights = audit.get("lights")
    if not isinstance(lights, dict) or set(lights) != set(LIGHTS):
        return False
    scales = {}
    for light in LIGHTS:
        entry = lights[light]
        expected = "SUSPECT_ABSENCE" if light == "MID" else "GEOMETRY_REVIEW"
        if not isinstance(entry, dict) or entry.get("status") != expected:
            return False
        if not _minimal_trim(entry.get("geometry")):
            return False
        rows = _verified_scales(entry)
        if rows is None:
            return False
        scales[light] = rows
    if any(row["suspected_absence"] for light in ("SIDE", "TOP") for row in scales[light]):
        return False
    mid = scales["MID"]
    if not (mid[0]["suspected_absence"] is True
            and mid[1]["suspected_absence"] is False):
        return False
    if mid[0]["classification"] != "CONTEÚDO ESPERADO AUSENTE":
        return False
    if mid[0]["distinctness"] > MAX_SUSPECT_REFERENCE_DISTINCTNESS:
        return False
    # A área mais ampla deve reter informação visual suficiente; se
    # ambas forem fracas, a divergência não é justificativa de exceção.
    return mid[1]["direct_similarity"] >= 0.60
