"""Candidato C: concordância OFFLINE da KNN A visual e B geométrica.

Aceita sugestões seletivas A ou B SOMENTE quando os dois votos KNN
independentes de rótulo consultado concordam e nenhuma sugestão diverge.
Não modifica a KNN de produção, CNN, memória, XP ou qualquer treinamento.
"""
from __future__ import annotations

SCHEMA = "visionx.knn_archive_visual_geometry_concordance.v1"
SUGGESTIONS = {"SUGERE_OK", "SUGERE_NG"}


def select_concordance_candidate(
    visual: dict | None, geometry: dict | None, *,
    visual_prediction: str | None, geometry_prediction: str | None,
) -> dict:
    result = {
        "schema": SCHEMA,
        "status": "REVISAO",
        "reason": "SEM_SUGESTAO_CONFIAVEL",
        "source": None,
        "diagnostic_only": True,
        "auto_decision_authorized": False,
        "production_approved": False,
        "xp_command": None,
        "uses_expected_label": False,
    }
    visual_status = (visual or {}).get("status", "REVISAO")
    geo_status = (geometry or {}).get("status", "REVISAO")
    # O voto total de ambas as KNNs deve concordar. Divergência = revisão
    # mesmo quando uma delas possui sugestão com alta similaridade.
    if (visual_prediction not in {"OK", "NG"}
            or geometry_prediction not in {"OK", "NG"}):
        result["reason"] = "SEM_DOIS_VOTOS_KNN"
        return result
    if visual_prediction != geometry_prediction:
        result["reason"] = "DISCORDANCIA_KNN_A_B"
        return result
    selected = {status for status in (visual_status, geo_status)
                if status in SUGGESTIONS}
    if len(selected) > 1:
        result["reason"] = "SUGESTOES_A_B_CONFLITANTES"
        return result
    if not selected:
        return result
    chosen = selected.pop()
    if chosen != "SUGERE_" + visual_prediction:
        result["reason"] = "SUGESTAO_DIVERGE_DO_VOTO_KNN"
        return result
    result["status"] = chosen
    if visual_status == geo_status == chosen:
        result["source"] = "A_E_B"
        result["reason"] = "A_E_B_CONCORDAM"
    elif visual_status == chosen:
        result["source"] = "A_COM_VOTO_B"
        result["reason"] = "A_SELETIVA_E_VOTOS_A_B_CONCORDAM"
    else:
        result["source"] = "B_COM_VOTO_A"
        result["reason"] = "B_SELETIVA_E_VOTOS_A_B_CONCORDAM"
    return result


__all__ = ["SCHEMA", "select_concordance_candidate"]
