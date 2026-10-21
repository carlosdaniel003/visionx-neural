"""Autorização por sessão da estratégia C para comando 0/1 supervisionado.

Exige evidência da fusão das três iluminações; consulta a política industrial
ANTES do envio e emite apenas proposta para o controlador já existente.
Sem armamento explícito, nada muda. A pausa por espaço desarma a sessão.
"""
from __future__ import annotations

SCHEMA = "visionx.knn_c_supervised_candidate.v1"
LIGHTS = frozenset({"SIDE", "TOP", "MID"})


def supervised_knn_c_decision(analysis: dict | None) -> str | None:
    if not isinstance(analysis, dict):
        return None
    if not (
        analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and analysis.get("production_review_required") is True
    ):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    if not (
        detail.get("knn_c_supervised_armed") is True
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("knn_c_multilight_review_gate") is True
        and detail.get("operator_review_required") is True
    ):
        return None
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict) or not (
        trace.get("knn_c_multilight_review_gate") is True
        and trace.get("operator_review_required") is True
    ):
        return None
    candidate = detail.get("knn_c_supervised_candidate")
    if not isinstance(candidate, dict) or not (
        candidate.get("schema") == SCHEMA
        and candidate.get("all_selected") is True
        and candidate.get("independent_reviews") is False
        and candidate.get("xp_auto_requires_session_arm") is True
        and candidate.get("lights") == ["SIDE", "TOP", "MID"]
    ):
        return None
    label = candidate.get("label")
    if label not in {"OK", "NG"}:
        return None
    if (
        analysis.get("verdict") != (
            "FALHA FALSA" if label == "OK" else "DEFEITO REAL"
        )
        or analysis.get("is_defect") is not (label == "NG")
    ):
        return None
    diagnostic = detail.get("knn_c_light_diagnostics")
    if not isinstance(diagnostic, dict) or set(diagnostic) != LIGHTS:
        return None
    if any(
        not isinstance(diagnostic[light], dict)
        or diagnostic[light].get("status") != "SUGERE_" + label
        or diagnostic[light].get("requires_human_confirmation") is not True
        or diagnostic[light].get("conflicts_with_existing_verdict") is True
        for light in LIGHTS
    ):
        return None
    return label


__all__ = ["supervised_knn_c_decision"]
