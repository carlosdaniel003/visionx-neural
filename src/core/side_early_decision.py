"""Encerramento antecipado da captura multilight após SIDE confiável.

A confiança exibida é um score do motor, não uma probabilidade calibrada.
A política de Produção pode emitir OK/NG após a apresentação, se o gate
industrial revalidar o resultado. A confiança não é acurácia certificada.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category


SIDE_EARLY_EXIT_THRESHOLD = 0.97
SIDE_EARLY_EXIT_RULE = "side_confidence_early_exit_v1"
SIDE_TRIAL_NG_REVIEW_RULE = "side_trial_ng_human_review_v1"
SIDE_TRIAL_OK_REVIEW_RULE = "side_trial_ok_human_review_v1"


def _uncorroborated_missing_ng_vs_two_knn_ok(analysis: dict, category: str) -> bool:
    """Do not finish trial SIDE when both KNN votes contradict CNN NG.

    KNN A/B review votes are weak diagnostic signals, NOT proof of OK.
    In this narrow FALTANDO situation, acquire TOP/MID before finalizing.
    Never changes the CNN verdict, label, production policy or XP command.
    """
    if (canonical_memory_category(category) != "FALTANDO"
            or analysis.get("is_defect") is not True
            or analysis.get("hard_missing_evidence") is True):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    if detail.get("hard_missing_evidence") is True:
        return False
    knn = detail.get("knn_c")
    return bool(
        isinstance(knn, dict)
        and knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("bank_status") == "READY"
        and knn.get("status") == "REVISAO"
        and knn.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
    )



def _ng_with_stronger_nearest_ok(analysis: dict, category: str) -> bool:
    """Acquire TOP/MID when independent nearest OK neighbors contradict SIDE NG.

    This only prevents a premature 1/NG: similarity never authorizes 0/OK.
    It also preserves the immediate NG path for physical missing evidence.
    """
    if (canonical_memory_category(category) != "FALTANDO"
            or analysis.get("is_defect") is not True
            or analysis.get("hard_missing_evidence") is True):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("hard_missing_evidence") is True:
        return False
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") == "SUSPECT_ABSENCE":
        return False
    knn = detail.get("knn_c")
    if not (isinstance(knn, dict)
            and knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and knn.get("bank_status") == "READY"
            and knn.get("diagnostic_only") is True
            and knn.get("auto_decision_authorized") is False):
        return False
    support = knn.get("reference_support")
    if not (isinstance(support, dict)
            and support.get("OK", 0) > 0 and support.get("NG", 0) > 0):
        return False
    for key, minimum_ok in (("neighbors_A", .90), ("neighbors_B", .85)):
        rows = knn.get(key)
        if (not isinstance(rows, list) or len(rows) < 5
                or any(not isinstance(row, dict)
                       or row.get("label") not in {"OK", "NG"}
                       for row in rows[:5])):
            return False
        try:
            first = rows[0]
            if first.get("label") != "OK":
                return False
            ok_score = float(first["similarity"])
            ng_scores = [
                float(row["similarity"]) for row in rows[1:5]
                if row.get("label") == "NG"
            ]
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            return False
        if (not isfinite(ok_score) or ok_score < minimum_ok
                or not ng_scores or any(not isfinite(v) for v in ng_scores)
                or ok_score - max(ng_scores) < .04):
            return False
    return True


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


def _eligible_trial_side_for_review(
    analysis: dict | None, category: str,
) -> bool:
    """Verifica o resultado binário da SIDE experimental antes do encerramento.

    A exigência herdada de revisão é preservada para auditoria. Autorizar
    envio ao XP é responsabilidade exclusiva do gate de Produção.
    """
    # Categoria AOI indica o roteamento, não o tipo real do defeito.
    # O checkpoint é compartilhado por FALTANDO/INVERTIDO/EMBORCADO/
    # DESLOCADO; nunca limitar o encerramento à palavra "FALTANDO".
    if not isinstance(analysis, dict) or not uses_faltando_v2(category):
        return False
    category_scope = canonical_memory_category(category)
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("verdict") in {"DEFEITO REAL", "FALHA FALSA"}
        and analysis.get("is_defect") is (analysis.get("verdict") == "DEFEITO REAL")
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
        and canonical_memory_category(detail.get("cnn_v2_aoi_category")) == category_scope
        and detail.get("cnn_v2_lighting_mode") == "SIDE"
        and trace.get("fusion_rule") == "cnn_v2_dualscale_direct"
        and detail.get("memory_conflict") is not True
        and trace.get("memory_conflict") is not True
    ):
        return False
    # A revisão herdada do checkpoint é tolerada; contraprovas independentes
    # e discordâncias permanecem impeditivas para 0/OK e 1/NG.
    defect = analysis["is_defect"]
    opposite_c = "SUGERE_OK" if defect else "SUGERE_NG"
    candidate = detail.get("knn_c")
    if isinstance(candidate, dict) and (
        candidate.get("status") == opposite_c
        or candidate.get("conflicts_with_existing_verdict") is True
    ):
        return False
    neighbors = detail.get("knn_visual_corroboration")
    if (not defect and isinstance(neighbors, dict)
            and neighbors.get("status") in {"NG_CORROBORATED", "CONFLICT"}):
        return False
    veto = detail.get("physical_missing_veto")
    if (not defect and isinstance(veto, dict)
            and veto.get("status") in {
                "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
            }):
        return False
    if detail.get("multilight_physical_disagreement") is True:
        return False
    if detail.get("multilight_missing_verified_presence"):
        return False
    # Provenance: 26/10/2026 false NG (SIDE FALTANDO, CNN .99919,
    # KNN A/B vote OK but both still REVISAO). High, uncalibrated CNN
    # confidence cannot by itself skip the two independent lights.
    if _uncorroborated_missing_ng_vs_two_knn_ok(analysis, category):
        return False
    # Mesmo com maioria KNN NG, dois vizinhos OK visualmente mais próximos
    # exigem observar TOP/MID antes de uma rejeição XP irreversível.
    if _ng_with_stronger_nearest_ok(analysis, category):
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
        and 0.0 <= ng_score <= 1.0
        and (
            ng_score >= SIDE_EARLY_EXIT_THRESHOLD if defect
            else ng_score <= 1.0 - SIDE_EARLY_EXIT_THRESHOLD
        )
        and abs(confidence - (ng_score if defect else 1.0 - ng_score)) <= 0.000001
    )


def build_side_early_decision(analysis: dict | None, category: str) -> dict | None:
    """Finaliza a inspeção pela SIDE; separa classificação de comando XP."""
    automatic = _eligible_side_analysis(analysis)
    trial_review = (
        not automatic and _eligible_trial_side_for_review(analysis, category)
    )
    if not automatic and not trial_review:
        return None

    verdict = str(analysis["verdict"]).strip().upper()
    confidence = float(analysis["confidence"])
    final = deepcopy(analysis)
    final_detail = final["detail"]
    final["reason"] = (
        f"SIDE: encerramento antecipado com confiança reportada de "
        f"{confidence:.1%}; TOP e MID dispensadas. "
        + ("Checkpoint experimental: autorização XP depende do gate de Produção. "
           if trial_review else "")
        + str(analysis.get("reason", "") or "")
    )
    final["lighting_mode"] = "SIDE"
    final["multilight_final"] = True
    final["multilight_category"] = str(category or "").strip().upper()
    final["eligible_for_final_decision"] = True
    # Preservar a revisão herdada como evidência do checkpoint experimental.
    # A autorização real 0/1 continua exclusiva da política de Produção.
    final["production_review_required"] = trial_review

    final_detail["multilight_final"] = True
    final_detail["multilight_category"] = final["multilight_category"]
    final_detail["eligible_for_final_decision"] = True
    final_detail["operator_review_required"] = trial_review
    final_detail["side_early_exit"] = {
        "rule": (
            (SIDE_TRIAL_NG_REVIEW_RULE if final["is_defect"]
             else SIDE_TRIAL_OK_REVIEW_RULE) if trial_review
            else SIDE_EARLY_EXIT_RULE
        ),
        "threshold": SIDE_EARLY_EXIT_THRESHOLD,
        "reported_confidence": confidence,
        "verdict": verdict,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": automatic,
        "operator_review_required": trial_review,
    }
    return final


def side_trial_automatic_decision(analysis: dict | None) -> str | None:
    """Permite 0/OK ou 1/NG somente para a SIDE trial final íntegra.

    O checkpoint mantém seu histórico de erro; os resultados da CNN são
    escores não calibrados e não comprovam acerto em todas as peças.
    """
    if not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    marker = detail.get("side_early_exit")
    if not isinstance(marker, dict):
        return None
    category = analysis.get("multilight_category")
    if not _eligible_trial_side_for_review(analysis, category):
        return None
    try:
        reported = float(marker.get("reported_confidence"))
    except (ValueError, TypeError, OverflowError):
        return None
    expected_rule = (
        SIDE_TRIAL_NG_REVIEW_RULE if analysis["is_defect"]
        else SIDE_TRIAL_OK_REVIEW_RULE
    )
    if not (
        analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("multilight_category") == category
        and uses_faltando_v2(category)
        and canonical_memory_category(detail.get("cnn_v2_aoi_category"))
            == canonical_memory_category(category)
        and marker.get("rule") == expected_rule
        and marker.get("threshold") == SIDE_EARLY_EXIT_THRESHOLD
        and marker.get("verdict") == analysis.get("verdict")
        and marker.get("analyzed_lights") == ["SIDE"]
        and marker.get("skipped_lights") == ["TOP", "MID"]
        and marker.get("production_auto_allowed") is False
        and marker.get("operator_review_required") is True
        and isfinite(reported)
        and reported == float(analysis.get("confidence"))
    ):
        return None
    return "NG" if analysis["is_defect"] else "OK"


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
    "SIDE_TRIAL_OK_REVIEW_RULE",
    "build_side_early_decision",
    "side_automatic_decision",
    "side_trial_automatic_decision",
]
