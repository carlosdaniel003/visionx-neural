"""Encerramento antecipado da captura multilight após SIDE confiável.

A confiança exibida é um score do motor, não uma probabilidade calibrada.
A política de Produção pode emitir OK/NG após a apresentação, se o gate\nindustrial revalidar o resultado. A confiança não é acurácia certificada.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256


SIDE_EARLY_EXIT_THRESHOLD = 0.97
SIDE_EARLY_EXIT_RULE = "side_confidence_early_exit_v1"


def _eligible_side_analysis(analysis: dict | None) -> bool:
    """Revalida a classificação SIDE sem depender de flags de UI ou Produção."""
    if not isinstance(analysis, dict):
        return False

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict not in {"DEFEITO REAL", "FALHA FALSA"}:
        return False
    if type(analysis.get("is_defect")) is not bool:
        return False
    if analysis["is_defect"] != (verdict == "DEFEITO REAL"):
        return False

    try:
        confidence = float(analysis.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not isfinite(confidence) or not SIDE_EARLY_EXIT_THRESHOLD <= confidence <= 1.0:
        return False

    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace", {})
    if not isinstance(trace, dict):
        return False
    if (
        analysis.get("production_review_required") is True
        or detail.get("operator_review_required") is True
        or trace.get("operator_review_required") is True
        or detail.get("memory_conflict") is True
        or trace.get("memory_conflict") is True
    ):
        return False

    route = str(detail.get("recognition_route", "") or "").strip().upper()
    if route == "MEMORY_CONFLICT":
        return False
    if route == "KNOWN_KNN" and not (
        detail.get("recognition_memory_verified") is True
        and detail.get("recognition_match") == "EXACT_PAIR"
        and detail.get("recognition_known_label") == (
            "NG" if analysis["is_defect"] else "OK"
        )
    ):
        return False

    if route == "NEW_CNN" or detail.get("cnn_v2_active") is True:
        if not (
            detail.get("cnn_v2_active") is True
            and detail.get("cnn_v2_status") == "INFERENCE_OK"
            and detail.get("cnn_v2_checkpoint_verified") is True
            and detail.get("cnn_v2_trial_manual_only") is not True
        ):
            return False


    # Nunca promover o checkpoint experimental explicitamente bloqueado.
    if detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256:
        return False
    return True


def build_side_early_decision(analysis: dict | None, category: str) -> dict | None:
    """Finaliza a peça pela SIDE sem capturar TOP/MID quando o resultado é válido."""
    if not _eligible_side_analysis(analysis):
        return None

    verdict = str(analysis["verdict"]).strip().upper()
    confidence = float(analysis["confidence"])
    final = deepcopy(analysis)
    final_detail = final["detail"]
    final["reason"] = (
        f"SIDE: encerramento antecipado com confiança reportada de "
        f"{confidence:.1%}; TOP e MID dispensadas. "
        + str(analysis.get("reason", "") or "")
    )
    final["lighting_mode"] = "SIDE"
    final["multilight_final"] = True
    final["multilight_category"] = str(category or "").strip().upper()
    final["eligible_for_final_decision"] = True
    # A autorização de comando será recalculada pela política de Produção.
    # Revisão, conflito e checkpoints bloqueados já foram excluídos.
    final["production_review_required"] = False

    final_detail["multilight_final"] = True
    final_detail["multilight_category"] = final["multilight_category"]
    final_detail["eligible_for_final_decision"] = True
    final_detail["operator_review_required"] = False
    final_detail["side_early_exit"] = {
        "rule": SIDE_EARLY_EXIT_RULE,
        "threshold": SIDE_EARLY_EXIT_THRESHOLD,
        "reported_confidence": confidence,
        "verdict": verdict,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": True,
    }
    return final


def side_automatic_decision(analysis: dict | None) -> str | None:
    """Retorna OK/NG somente para a SIDE final válida, nunca para um flag isolado.

    A política de Produção ainda executa os vetos e verifica o equipamento.
    """
    if not _eligible_side_analysis(analysis):
        return None

    detail = analysis["detail"]
    marker = detail.get("side_early_exit")
    if not isinstance(marker, dict):
        return None
    try:
        reported = float(marker.get("reported_confidence"))
    except (TypeError, ValueError, OverflowError):
        return None

    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and analysis.get("production_review_required") is False
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("operator_review_required") is False
        and bool(analysis.get("multilight_category"))
        and analysis["multilight_category"] == detail.get("multilight_category")
        and marker.get("rule") == SIDE_EARLY_EXIT_RULE
        and marker.get("threshold") == SIDE_EARLY_EXIT_THRESHOLD
        and isfinite(reported)
        and reported == float(analysis["confidence"])
        and marker.get("verdict") == str(analysis["verdict"]).strip().upper()
        and marker.get("analyzed_lights") == ["SIDE"]
        and marker.get("skipped_lights") == ["TOP", "MID"]
        and marker.get("production_auto_allowed") is True
    ):
        return None

    return "NG" if analysis["is_defect"] else "OK"


__all__ = [
    "SIDE_EARLY_EXIT_THRESHOLD",
    "SIDE_EARLY_EXIT_RULE",
    "build_side_early_decision",
    "side_automatic_decision",
]
