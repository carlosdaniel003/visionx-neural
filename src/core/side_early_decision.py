"""Encerramento antecipado da captura multilight após SIDE confiável.

A confiança exibida é um score do motor, não uma probabilidade calibrada.
A política de Produção pode emitir OK/NG após a apresentação, se o gate
industrial revalidar o resultado. A confiança não é acurácia certificada.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256


SIDE_EARLY_EXIT_THRESHOLD = 0.97
SIDE_EARLY_EXIT_RULE = "side_confidence_early_exit_v1"
SIDE_TRIAL_NG_REVIEW_RULE = "side_trial_ng_human_review_v1"


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


def _eligible_trial_ng_for_review(
    analysis: dict | None, category: str,
) -> bool:
    """Dispensa outras luzes para mostrar NG, sem liberar comando industrial.

    O checkpoint experimental teve falsos NG históricos: apenas NG local
    inequívoco pode antecipar a *apresentação*, sempre com revisão humana.
    OK, conflito, falha técnica e demais checkpoints mantêm o fluxo anterior.
    """
    if not isinstance(analysis, dict) or str(category).strip().upper() != "FALTANDO":
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("verdict") == "DEFEITO REAL"
        and analysis.get("is_defect") is True
        and analysis.get("production_review_required") is True
        and detail.get("operator_review_required") is True
        and trace.get("operator_review_required") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_active") is True
        and detail.get("cnn_v2_experimental") is True
        and detail.get("cnn_v2_status") == "INFERENCE_OK"
        and detail.get("cnn_v2_checkpoint_verified") is True
        and detail.get("cnn_v2_trial_manual_only") is True
        and detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256
        and detail.get("cnn_v2_aoi_category") == "FALTANDO"
        and detail.get("cnn_v2_lighting_mode") == "SIDE"
        and trace.get("fusion_rule") == "cnn_v2_dualscale_direct"
        and detail.get("memory_conflict") is not True
        and trace.get("memory_conflict") is not True
    ):
        return False
    # Não encerrar antecipadamente quando a própria estratégia C discorda.
    candidate = detail.get("knn_c")
    if isinstance(candidate, dict) and candidate.get("status") == "SUGERE_OK":
        return False
    try:
        confidence = float(analysis["confidence"])
        ng_score = float(detail["cnn_v2_ng_score_uncalibrated"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return False
    return bool(
        isfinite(confidence)
        and isfinite(ng_score)
        and SIDE_EARLY_EXIT_THRESHOLD <= confidence <= 1.0
        and SIDE_EARLY_EXIT_THRESHOLD <= ng_score <= 1.0
        and abs(confidence - ng_score) <= 0.000001
    )


def build_side_early_decision(analysis: dict | None, category: str) -> dict | None:
    """Finaliza a inspeção pela SIDE; separa classificação de comando XP."""
    automatic = _eligible_side_analysis(analysis)
    manual_trial_ng = (
        not automatic and _eligible_trial_ng_for_review(analysis, category)
    )
    if not automatic and not manual_trial_ng:
        return None

    verdict = str(analysis["verdict"]).strip().upper()
    confidence = float(analysis["confidence"])
    final = deepcopy(analysis)
    final_detail = final["detail"]
    final["reason"] = (
        f"SIDE: encerramento antecipado com confiança reportada de "
        f"{confidence:.1%}; TOP e MID dispensadas. "
        + ("Checkpoint experimental: confirmar NG no operador. " if manual_trial_ng else "")
        + str(analysis.get("reason", "") or "")
    )
    final["lighting_mode"] = "SIDE"
    final["multilight_final"] = True
    final["multilight_category"] = str(category or "").strip().upper()
    final["eligible_for_final_decision"] = True
    # A apresentação é final, mas o modelo experimental NÃO autoriza XP.
    final["production_review_required"] = manual_trial_ng

    final_detail["multilight_final"] = True
    final_detail["multilight_category"] = final["multilight_category"]
    final_detail["eligible_for_final_decision"] = True
    final_detail["operator_review_required"] = manual_trial_ng
    final_detail["side_early_exit"] = {
        "rule": SIDE_TRIAL_NG_REVIEW_RULE if manual_trial_ng else SIDE_EARLY_EXIT_RULE,
        "threshold": SIDE_EARLY_EXIT_THRESHOLD,
        "reported_confidence": confidence,
        "verdict": verdict,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": automatic,
        "operator_review_required": manual_trial_ng,
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
    "SIDE_TRIAL_NG_REVIEW_RULE",
    "build_side_early_decision",
    "side_automatic_decision",
]
