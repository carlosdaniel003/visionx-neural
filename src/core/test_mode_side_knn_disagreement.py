"""Arbitragem não decisória: em Modo Teste investigar SIDE CNN-NG × KNN A/B-OK.

Não muda pesos, rotas, classificação ou autorização 0/1; coleta TOP/MID
quando a CNN encerraria na SIDE, mas as duas KNNs só conseguiram votos OK
não seletivos (REVISAO). Falha fechada em campos incompletos.
"""
from __future__ import annotations

from math import isfinite

SCHEMA = "visionx.test_side_knn_uncertainty.v1"


def test_mode_side_knn_disagreement(analysis: dict | None, mode: str) -> dict | None:
    if mode != "Modo Teste" or not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    cnn = detail.get("cnn_v2_ng_score_uncalibrated")
    try:
        score = float(cnn)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(score) or score < 0.97 or score > 1.0:
        return None
    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("verdict") == "DEFEITO REAL"
        and analysis.get("is_defect") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_active") is True
        and detail.get("cnn_v2_status") == "INFERENCE_OK"
        and detail.get("cnn_v2_checkpoint_verified") is True
    ):
        return None
    knn = detail.get("knn_c")
    if not isinstance(knn, dict):
        return None
    if not (
        knn.get("bank_status") == "READY"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("status") == "REVISAO"
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
    ):
        return None
    return {
        "schema": SCHEMA,
        "status": "DEFER_TO_TOP_MID",
        "mode": "Modo Teste",
        "reason": "CNN_NG_KNN_A_B_OK_SELETIVAS_EM_REVISAO",
        "cnn_verdict_unchanged": "DEFEITO REAL",
        "knn_votes": {"A": "OK", "B": "OK"},
        "knn_suggestions": {"A": "REVISAO", "B": "REVISAO"},
        "side_cnn_ng_score_uncalibrated": score,
        "extra_lights": ["TOP", "MID"],
        "diagnostic_only": True,
        "xp_command_authorized": False,
        "auto_ok_authorized": False,
    }


__all__ = ["SCHEMA", "test_mode_side_knn_disagreement"]
