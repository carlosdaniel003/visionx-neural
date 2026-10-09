"""Consenso conservador da CNN de ausência v2 em SIDE/TOP/MID.

Somente três inferências válidas, com o mesmo modelo verificado, podem ser
consideradas para decisão supervisionada automática. Concordância histórica
no arquivo não é prova de generalização em NG novos.
"""
from __future__ import annotations

import math
import re

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category

LIGHTS = ("SIDE", "TOP", "MID")
SCORE_OK_MAX = 0.10
SCORE_NG_MIN = 0.90


def summarize_cnn_multilight(analyses: dict, category: str) -> dict:
    result = {
        "cnn_v2_supervised_auto_eligible": False,
        "cnn_v2_consensus": "REVIEW",
        "cnn_v2_light_votes": {light: "REVIEW" for light in LIGHTS},
        "cnn_v2_verified_lights": [],
        "cnn_v2_consensus_reason": "UNSUPPORTED_OR_INCOMPLETE",
        "cnn_v2_model_sha256": None,
    }
    if not uses_faltando_v2(category) or not isinstance(analyses, dict):
        return result
    expected_category = canonical_memory_category(category)
    votes = {}
    digests = []
    for light in LIGHTS:
        local = analyses.get(light)
        if not isinstance(local, dict):
            result["cnn_v2_consensus_reason"] = "MISSING_LIGHT_RESULT"
            return result
        detail = local.get("detail", {})
        if not isinstance(detail, dict):
            return result
        trace = detail.get("decision_trace", {})
        if not isinstance(trace, dict):
            trace = {}
        if (detail.get("recognition_route") != "NEW_CNN"
            or canonical_memory_category(detail.get("cnn_v2_aoi_category"))
               != expected_category
            or detail.get("cnn_v2_active") is not True
            or detail.get("cnn_v2_checkpoint_verified") is not True
            or detail.get("cnn_v2_experimental") is not True
            or detail.get("cnn_v2_status") != "INFERENCE_OK"
            or detail.get("cnn_v2_lighting_mode") != light
            or str(local.get("lighting_mode", "")).upper() != light
            or local.get("production_review_required", False)
            or detail.get("operator_review_required", False)
            or trace.get("operator_review_required", False)
            or trace.get("fusion_rule") != "cnn_v2_dualscale_direct"):
            result["cnn_v2_consensus_reason"] = "INVALID_LIGHT_EVIDENCE"
            return result
        digest = str(detail.get("cnn_v2_checkpoint_sha256") or "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            result["cnn_v2_consensus_reason"] = "UNKNOWN_MODEL_HASH"
            return result
        digests.append(digest.lower())
        try:
            score = float(detail.get("cnn_v2_ng_score_uncalibrated"))
        except (TypeError, ValueError):
            result["cnn_v2_consensus_reason"] = "INVALID_SCORE"
            return result
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            result["cnn_v2_consensus_reason"] = "INVALID_SCORE"
            return result
        if score <= SCORE_OK_MAX:
            vote = "OK"
            expected_verdict = "FALHA FALSA"
            defect = False
        elif score >= SCORE_NG_MIN:
            vote = "NG"
            expected_verdict = "DEFEITO REAL"
            defect = True
        else:
            result["cnn_v2_consensus_reason"] = "INDETERMINATE_SCORE"
            return result
        if (str(local.get("verdict", "")).upper() != expected_verdict
                or local.get("is_defect") is not defect):
            result["cnn_v2_consensus_reason"] = "INCONSISTENT_VERDICT"
            return result
        votes[light] = vote
        result["cnn_v2_light_votes"][light] = vote
        result["cnn_v2_verified_lights"].append(light)

    if len(set(digests)) != 1:
        result["cnn_v2_consensus_reason"] = "CHECKPOINT_CHANGED_DURING_CYCLE"
        return result
    result["cnn_v2_model_sha256"] = digests[0]
    if len(set(votes.values())) != 1:
        result["cnn_v2_consensus_reason"] = "DISAGREEMENT_BETWEEN_LIGHTS"
        return result
    result["cnn_v2_consensus"] = votes["SIDE"]
    result["cnn_v2_supervised_auto_eligible"] = True
    result["cnn_v2_consensus_reason"] = "ALL_THREE_LIGHTS_CONCLUSIVE"
    return result


__all__ = ["summarize_cnn_multilight", "LIGHTS", "SCORE_OK_MAX", "SCORE_NG_MIN"]
