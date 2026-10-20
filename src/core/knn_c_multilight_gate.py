"""Não permitir que a fusão SIDE/TOP/MID perca a revisão da KNN C.

É uma trava industrial de apresentação/XP; não recalcula o veredito
físico, não transforma um NG em OK e não cria memória/treinamento.
"""
from __future__ import annotations

from copy import deepcopy

LIGHTS = ("SIDE", "TOP", "MID")
C_SELECTED = frozenset({"SUGERE_OK", "SUGERE_NG"})


def gate_knn_c_multilight(fused: dict | None, analyses: dict) -> dict | None:
    if not isinstance(fused, dict):
        return fused
    current = {}
    for light in LIGHTS:
        raw = analyses.get(light)
        if not isinstance(raw, dict):
            continue
        detail = raw.get("detail")
        cand = detail.get("knn_c") if isinstance(detail, dict) else None
        if (isinstance(cand, dict)
                and cand.get("status") in C_SELECTED
                and cand.get("requires_human_confirmation") is True):
            current[light] = {
                "status": cand["status"],
                "source": cand.get("source"),
                "conflicts_with_existing_verdict": bool(
                    cand.get("conflicts_with_existing_verdict")
                ),
                "requires_human_confirmation": True,
            }
    if not current:
        return fused
    detail = fused.setdefault("detail", {})
    if not isinstance(detail, dict):
        detail = {}
        fused["detail"] = detail
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        trace = {}
        detail["decision_trace"] = trace

    fused["production_review_required"] = True
    detail["operator_review_required"] = True
    trace["operator_review_required"] = True
    trace["knn_c_multilight_review_gate"] = True
    detail["knn_c_light_diagnostics"] = deepcopy(current)
    detail["knn_c_multilight_review_gate"] = True

    # A fusão pode ter declarado todas as luzes OK; um NG da C exige
    # revisão, nunca deve produzir 0/OK automático.
    if (str(fused.get("verdict", "")).upper() == "FALHA FALSA"
            and any(c["status"] == "SUGERE_NG" for c in current.values())):
        fused["verdict"] = "REVISÃO OBRIGATÓRIA"
        fused["confidence"] = .5
        trace["verdict"] = "REVISÃO OBRIGATÓRIA"
    fused["reason"] = (
        str(fused.get("reason", "")) +
        " | KNN C por iluminação: " +
        ", ".join(f"{light}:{current[light]['status']}" for light in LIGHTS
                  if light in current) +
        "; confirmação humana obrigatória (XP bloqueado)."
    )
    return fused


__all__ = ["gate_knn_c_multilight"]
