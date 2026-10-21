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

    # Liberação SUPERVISIONADA somente quando AS TRÊS LUZES tiveram
    # uma recomendação seletiva C idêntica, votos A/B de acordo e nenhum
    # especialista de origem exigia revisão. Divergência = veto, nunca XP.
    suggestions = {item["status"] for item in current.values()}
    agreed = next(iter(suggestions)) if len(suggestions) == 1 else None
    label = {"SUGERE_OK": "OK", "SUGERE_NG": "NG"}.get(agreed)
    routes = detail.get("recognition_light_routes")
    sources_valid = bool(
        label
        and len(current) == 3
        and isinstance(routes, dict)
        and set(routes) == set(LIGHTS)
        and all(routes[light] in {"NEW_CNN", "NEW_EXPERTS"} for light in LIGHTS)
    )
    if sources_valid:
        for light in LIGHTS:
            raw = analyses.get(light)
            raw_detail = raw.get("detail") if isinstance(raw, dict) else None
            cand = raw_detail.get("knn_c") if isinstance(raw_detail, dict) else None
            if not (isinstance(cand, dict)
                    and cand.get("schema") == "visionx.knn_c_live_review_gated.v1"
                    and cand.get("bank_status") == "READY"
                    and cand.get("status") == "SUGERE_" + label
                    and cand.get("vote_A") == label
                    and cand.get("vote_B") == label
                    and cand.get("suggestion_A") in {"REVISAO", "SUGERE_" + label}
                    and cand.get("suggestion_B") in {"REVISAO", "SUGERE_" + label}
                    and (cand.get("suggestion_A") == "SUGERE_" + label
                         or cand.get("suggestion_B") == "SUGERE_" + label)
                    and cand.get("pre_c_review_required") is False
                    and cand.get("pre_c_verdict") == (
                        "FALHA FALSA" if label == "OK" else "DEFEITO REAL"
                    )
                    and cand.get("pre_c_is_defect") is (label == "NG")
                    and cand.get("conflicts_with_existing_verdict") is False
                    and raw.get("is_defect") is (label == "NG")):
                sources_valid = False
                break
    if sources_valid:
        # Um resultado fisicamente inconsistente ou veredito de fusão
        # independente não pode ser reescrito pela estratégia C.
        if (detail.get("multilight_physical_disagreement") is True
                or detail.get("multilight_missing_verified_presence")
                or detail.get("multilight_strong_positive_modes") and label == "OK"
                or fused.get("is_defect") is not (label == "NG")
                or (
                    label == "OK"
                    and detail.get("fusion_rule") != "multilight_local_review"
                )
                or (
                    label == "NG"
                    and fused.get("verdict") != "DEFEITO REAL"
                )):
            sources_valid = False
    if sources_valid:
        detail["knn_c_supervised_candidate"] = {
            "schema": "visionx.knn_c_supervised_candidate.v1",
            "label": label,
            "lights": list(LIGHTS),
            "all_selected": True,
            "independent_reviews": False,
            "xp_auto_requires_session_arm": True,
        }
        # Exibir classificação coerente com as três luzes, ainda com
        # operator_review_required=True até armamento explícito no painel.
        if label == "OK":
            fused["verdict"] = "FALHA FALSA"
            fused["confidence"] = min(
                float(analyses[light].get("confidence", .5))
                for light in LIGHTS
            )
            trace["verdict"] = "FALHA FALSA"
        fused["reason"] = (
            str(fused.get("reason", "")) +
            f" | KNN C unânime {label} nas 3 luzes: disponível apenas "
            "com autorização supervisionada desta sessão."
        )

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
