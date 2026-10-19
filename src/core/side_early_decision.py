"""Encerramento antecipado da captura multilight após SIDE confiável.

A confiança exibida é um score do motor, não uma probabilidade calibrada.
Esta regra otimiza a aquisição, sem autorizar comandos automáticos no XP.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite


SIDE_EARLY_EXIT_THRESHOLD = 0.97
SIDE_EARLY_EXIT_RULE = "side_confidence_early_exit_v1"


def build_side_early_decision(analysis: dict | None, category: str) -> dict | None:
    """Cria um resultado final auditável quando SIDE já permite encerrar a coleta.

    Uma classificação local com revisão, conflito ou falha técnica não permite
    pular as testemunhas TOP/MID, independentemente do score reportado.
    """
    if not isinstance(analysis, dict):
        return None

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict not in {"DEFEITO REAL", "FALHA FALSA"}:
        return None
    if type(analysis.get("is_defect")) is not bool:
        return None
    if analysis["is_defect"] != (verdict == "DEFEITO REAL"):
        return None

    try:
        confidence = float(analysis.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(confidence) or not SIDE_EARLY_EXIT_THRESHOLD <= confidence <= 1.0:
        return None

    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    trace = detail.get("decision_trace", {})
    if not isinstance(trace, dict):
        return None
    if (
        analysis.get("production_review_required") is True
        or detail.get("operator_review_required") is True
        or trace.get("operator_review_required") is True
        or detail.get("memory_conflict") is True
        or trace.get("memory_conflict") is True
    ):
        return None

    route = str(detail.get("recognition_route", "") or "").strip().upper()
    if route == "MEMORY_CONFLICT":
        return None
    if route == "KNOWN_KNN" and not (
        detail.get("recognition_memory_verified") is True
        and detail.get("recognition_match") == "EXACT_PAIR"
        and detail.get("recognition_known_label") == (
            "NG" if analysis["is_defect"] else "OK"
        )
    ):
        return None

    if route == "NEW_CNN" or detail.get("cnn_v2_active") is True:
        if not (
            detail.get("cnn_v2_active") is True
            and detail.get("cnn_v2_status") == "INFERENCE_OK"
            and detail.get("cnn_v2_checkpoint_verified") is True
            and detail.get("cnn_v2_trial_manual_only") is not True
        ):
            return None

    final = deepcopy(analysis)
    final_detail = final["detail"]
    final["reason"] = (
        f"SIDE: encerramento antecipado com confiança reportada de "
        f"{confidence:.1%}; TOP e MID dispensadas. "
        + str(analysis.get("reason", "") or "")
    )
    final["multilight_final"] = True
    final["multilight_category"] = str(category or "").strip().upper()
    final["eligible_for_final_decision"] = True
    # Uma imagem não fornece os três votos da política de Produção.
    # Preservar o veredito visual, mas não autorizar 0/1 automaticamente.
    final["production_review_required"] = True

    final_detail["multilight_final"] = True
    final_detail["multilight_category"] = final["multilight_category"]
    final_detail["eligible_for_final_decision"] = True
    final_detail["operator_review_required"] = True
    final_detail["side_early_exit"] = {
        "rule": SIDE_EARLY_EXIT_RULE,
        "threshold": SIDE_EARLY_EXIT_THRESHOLD,
        "reported_confidence": confidence,
        "verdict": verdict,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": False,
    }
    return final


__all__ = [
    "SIDE_EARLY_EXIT_THRESHOLD",
    "SIDE_EARLY_EXIT_RULE",
    "build_side_early_decision",
]
