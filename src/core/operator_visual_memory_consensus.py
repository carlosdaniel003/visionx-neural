"""Prioridade de memória humana multilight sobre falso NG da CNN experimental.

A memória visual é uma contraprova supervisionada, não identidade RGB exata
nem autorização de saída 0/XP. A mesma função é usada em Teste, Produção e
Sombra, antes de a política do modo decidir sobre o XP.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category

LIGHTS = ("SIDE", "TOP", "MID")
SCHEMA = "visionx.operator_visual_memory_consensus.v1"
RULE = "operator_visual_memory_multilight_ok_v1"
MIN_HUMAN = {"SIDE": 0.98, "TOP": 0.98, "MID": 0.965}
MIN_HUMAN_MARGIN = 0.07
MIN_VISUAL_OK = 0.96
MIN_VISUAL_MARGIN = 0.07


def _score(value):
    if type(value) not in (float, int):
        return None
    number = float(value)
    return number if isfinite(number) and 0 <= number <= 1 else None


def _no_hard_ng(fused: dict) -> bool:
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if any(
        row.get(key) is True
        for row in (fused, detail, trace)
        for key in ("hard_missing_evidence", "raw_hard_missing_evidence")
    ):
        return False
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and (
        physical.get("status") in {"SUSPECT_ABSENCE", "NG_CORROBORATED", "UNAVAILABLE"}
        or physical.get("suspect_lights")
        or physical.get("unavailable_lights")
    ):
        return False
    return _score(detail.get("physical_score")) == 0.0


def memory_ok_counterevidence(analyses: dict, fused: dict) -> dict | None:
    """Verifica as três iluminações e duas fontes visuais independentes.

    Não decide por quantidade de imagens na pasta nem por semelhança única.
    NG físico ou memória NG competitiva impede qualquer promoção.
    """
    if not isinstance(fused, dict) or not isinstance(analyses, dict):
        return None
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return None
    category = canonical_memory_category(fused.get("multilight_category", ""))
    if not (
        uses_faltando_v2(category)
        and fused.get("multilight_final") is True
        and fused.get("verdict") == "DEFEITO REAL"
        and fused.get("is_defect") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "REVIEW"
        and detail.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        and set(detail.get("multilight_strong_positive_modes") or ()) == set(LIGHTS)
        and _no_hard_ng(fused)
    ):
        return None
    report = detail.get("knn_visual_corroboration")
    if not isinstance(report, dict) or not (
        report.get("schema") == "visionx.knn_visual_corroboration.v1"
        and report.get("category") == category
        and report.get("status") == "NO_CONCLUSION"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return None
    rows = report.get("lights")
    if not isinstance(rows, dict) or set(rows) != set(LIGHTS):
        return None

    # A KNN C oferece uma terceira contraprova da SIDE. Nas outras luzes
    # seu voto pode estar em revisão, mas a busca independente deve
    # demonstrar margem OK consistente em cada luz.
    side = analyses.get("SIDE")
    side_detail = side.get("detail") if isinstance(side, dict) else None
    knn = side_detail.get("knn_c") if isinstance(side_detail, dict) else None
    if not isinstance(knn, dict) or not (
        knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("bank_status") == "READY"
        and knn.get("status") == "SUGERE_OK"
        and knn.get("reason") == "A_E_B_CONCORDAM"
        and knn.get("vote_A") == knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == knn.get("suggestion_B") == "SUGERE_OK"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
    ):
        return None
    for key, minimum in (("neighbors_A", 0.96), ("neighbors_B", 0.90)):
        neighbors = knn.get(key)
        if not isinstance(neighbors, list) or len(neighbors) < 5 or any(
            not isinstance(item, dict) or item.get("label") != "OK"
            for item in neighbors[:5]
        ):
            return None
        if (_score(neighbors[0].get("similarity")) or 0) < minimum:
            return None

    observations = {}
    model_sha = None
    for light in LIGHTS:
        item = analyses.get(light)
        local = item.get("detail") if isinstance(item, dict) else None
        if not isinstance(local, dict) or not (
            item.get("lighting_mode") == light
            and local.get("cnn_v2_lighting_mode") == light
            and local.get("cnn_v2_checkpoint_verified") is True
            and local.get("cnn_v2_status") == "INFERENCE_OK"
            and local.get("recognition_route") == "NEW_CNN"
            and canonical_memory_category(local.get("cnn_v2_aoi_category")) == category
        ):
            return None
        digest = local.get("cnn_v2_checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return None
        if model_sha is not None and digest != model_sha:
            return None
        model_sha = digest

        human = local.get("confirmed_visual_recall")
        if not isinstance(human, dict) or not (
            human.get("schema") == "visionx.human_visual_recall.v1"
            and human.get("source") == "PERSISTENT_HUMAN_LABELS"
            and human.get("diagnostic_only") is True
            and human.get("auto_ok_authorized") is False
            and human.get("category") == category
            and human.get("status") in {"HUMAN_OK_VISUAL_CANDIDATE", "NO_MATCH"}
            and human.get("reason") == "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY"
        ):
            return None
        # O índice histórico usa >=.97 como candidato; MID .967 pode
        # continuar NO_MATCH sem perder a contraprova multilight.
        if light != "MID" and human["status"] != "HUMAN_OK_VISUAL_CANDIDATE":
            return None
        source = human.get("best_ok_source")
        if not isinstance(source, dict) or not (
            source.get("source") == "OPERATOR_CONFIRMED"
            and source.get("label") == "OK"
            and source.get("lighting_mode") == light
            and isinstance(source.get("record_id"), str)
            and len(source["record_id"]) == 64
        ):
            return None
        score_ok = _score(human.get("best_ok_similarity"))
        score_ng = _score(human.get("best_ng_similarity"))
        if (
            score_ok is None or score_ng is None
            or score_ok < MIN_HUMAN[light]
            or score_ok - score_ng < MIN_HUMAN_MARGIN
        ):
            return None

        visual = rows[light]
        if not isinstance(visual, dict) or not (
            visual.get("status") == "NO_CONCLUSION"
            and visual.get("cross_category_ng") is False
            and visual.get("ng_support") in (0, 1)
        ):
            return None
        best_ok = _score(visual.get("ok", {}).get("best_similarity"))
        best_ng = _score(visual.get("ng", {}).get("best_similarity"))
        margin = visual.get("ng_margin_over_ok")
        if (
            best_ok is None or best_ng is None
            or type(margin) not in (float, int) or not isfinite(margin)
            or best_ok < MIN_VISUAL_OK
            or best_ok - best_ng < MIN_VISUAL_MARGIN
            or abs(float(margin) - (best_ng - best_ok)) > 0.00001
        ):
            return None
        observations[light] = {
            "human_ok": score_ok, "human_ng": score_ng,
            "human_record_id": source["record_id"],
            "archive_ok": best_ok, "archive_ng": best_ng,
        }

    return {
        "schema": SCHEMA,
        "label": "OK",
        "category": category,
        "model_sha256": model_sha,
        "lights": observations,
        "source": "OPERATOR_CONFIRMED_PLUS_VISUAL_KNN",
        "rgb_exact_pair": False,
        "automatic_xp_authorized": False,
        "reason": "THREE_LIGHT_HUMAN_OK_WITH_KNN_COUNTEREVIDENCE",
    }


def apply_operator_visual_memory_consensus(
    fused: dict | None, analyses: dict | None,
) -> dict | None:
    """Prioridade do conhecimento supervisionado no veredito, sem liberar XP."""
    if not isinstance(fused, dict):
        return fused
    proof = memory_ok_counterevidence(analyses, fused)
    if proof is None:
        return fused
    result = deepcopy(fused)
    detail = result["detail"]
    detail["operator_visual_memory_consensus"] = proof
    detail["cnn_raw_verdict_before_memory"] = fused["verdict"]
    result["verdict"] = "FALHA FALSA"
    result["is_defect"] = False
    result["confidence"] = 0.5
    result["production_review_required"] = True
    result["eligible_for_final_decision"] = False
    result["reason"] = (
        "MEMÓRIA HUMANA MULTILIGHT: OK anteriormente confirmado com "
        "contraprova visual favorável em SIDE/TOP/MID e KNN C SIDE OK; "
        "CNN experimental contradiz as referências. Sem par RGB exato: "
        "decisão de XP requer nova confirmação humana."
    )
    detail["fusion_rule"] = RULE
    detail["operator_review_required"] = True
    detail["eligible_for_final_decision"] = False
    detail["cnn_v2_supervised_auto_eligible"] = False
    detail["final_score"] = 0.5
    trace = detail.get("decision_trace")
    if isinstance(trace, dict):
        trace.update(
            verdict="FALHA FALSA", physical_defect=False,
            operator_review_required=True,
            fusion_rule=RULE, final_score=0.5,
        )
    return result


__all__ = [
    "SCHEMA", "RULE", "memory_ok_counterevidence",
    "apply_operator_visual_memory_consensus",
]
