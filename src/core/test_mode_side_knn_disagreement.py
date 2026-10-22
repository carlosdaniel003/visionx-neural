"""Investigação SIDE CNN-NG no Modo Teste: KNN A/B-OK incerta ou LOADING.

Não muda pesos, veredito nem autorização 0/1. Quando a CNN encerraria a
SIDE e a memória visual ainda carrega ou as KNN A/B votam OK sem sugestão
seletiva, requisita TOP/MID. Nunca inventa voto quando banco está LOADING.
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
        knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("status") == "REVISAO"
    ):
        return None

    # Distinção crucial: LOADING não é voto OK e não equivale a falha
    # de correspondência. Teste deve coletar mais evidências da mesma peça,
    # sem esperar pelo índice e sem modificar decisões de Produção/Sombra.
    loading = (
        knn.get("bank_status") == "LOADING"
        and knn.get("reason") == "MEMORIA_VISUAL_EM_PREPARACAO"
        and knn.get("vote_A") is None
        and knn.get("vote_B") is None
    )
    disagreement = (
        knn.get("bank_status") == "READY"
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
    )
    if not (loading or disagreement):
        return None
    return {
        "schema": SCHEMA,
        "status": "DEFER_TO_TOP_MID",
        "mode": "Modo Teste",
        "reason": (
            "CNN_NG_KNN_C_LOADING_SEM_VOTO"
            if loading else "CNN_NG_KNN_A_B_OK_SELETIVAS_EM_REVISAO"
        ),
        "knn_bank_status": str(knn.get("bank_status")),
        "cnn_verdict_unchanged": "DEFEITO REAL",
        "knn_votes": {"A": None if loading else "OK",
                      "B": None if loading else "OK"},
        "knn_suggestions": {
            "A": None if loading else "REVISAO",
            "B": None if loading else "REVISAO",
        },
        "side_cnn_ng_score_uncalibrated": score,
        "extra_lights": ["TOP", "MID"],
        "diagnostic_only": True,
        "xp_command_authorized": False,
        "auto_ok_authorized": False,
    }


__all__ = ["SCHEMA", "test_mode_side_knn_disagreement"]
