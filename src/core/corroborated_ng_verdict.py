"""Contrato: classificação NG corroborada != aprovação automática.

A fusão CNN/KNN/física produz um julgamento NG quando três evidências
auditáveis concordam sobre ausência. A confirmação humana é outra dimensão
e pode continuar pendente sem reescrever o julgamento para REVISÃO.
"""
from __future__ import annotations

from typing import Any

LIGHTS = frozenset({"SIDE", "TOP", "MID"})
FUSION_RULE = "knn_visual_cnn_physical_corroboration_v1"


def corroborated_ng_verdict(analysis: Any) -> bool:
    if not isinstance(analysis, dict):
        return False
    if (
        analysis.get("multilight_final") is not True
        or analysis.get("is_defect") is not True
        or str(analysis.get("verdict", "")).strip().upper() != "DEFEITO REAL"
    ):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        trace = {}
    if (
        detail.get("multilight_category") != "FALTANDO"
        or detail.get("recognition_route") != "NEW_CNN"
        or detail.get("cnn_v2_consensus") != "OK"
        or (
            detail.get("fusion_rule") != FUSION_RULE
            and trace.get("fusion_rule") != FUSION_RULE
        )
    ):
        return False
    visual = detail.get("knn_visual_corroboration")
    physical = detail.get("physical_missing_veto")
    if not isinstance(visual, dict) or not isinstance(physical, dict):
        return False
    strong = visual.get("strong_ng_lights")
    if (
        visual.get("status") != "NG_CORROBORATED"
        or not isinstance(strong, list)
        or len(set(strong)) < 2
        or any(light not in LIGHTS for light in strong)
        or physical.get("status") != "SUSPECT_ABSENCE"
    ):
        return False
    suspected = physical.get("suspect_lights")
    return (
        isinstance(suspected, list)
        and bool(suspected)
        and all(light in LIGHTS for light in suspected)
    )


__all__ = ["corroborated_ng_verdict", "FUSION_RULE"]
