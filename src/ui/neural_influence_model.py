"""Telemetria da influência CNN/KNN sem inventar fusão ponderada.

Retorna linhas compatíveis com DecisionInfluenceWidget. Em multilight as
três linhas são votos independentes, e não parcelas aritméticas do score.
"""
from __future__ import annotations

from src.ui.neural_telemetry_model import neural_summary, _finite_score, _map, LIGHTS


def neural_influence_rows(analysis: dict | None) -> list[dict]:
    m = neural_summary(analysis)
    if not m["active"]:
        return []

    def row(key: str, label: str, score: float | None, *, text: str,
            weight: float, selected: bool, value_format: str = "") -> dict:
        s = score if score is not None else 0.0
        return {
            "id": key, "label": label, "active": True, "triggered": s >= .5,
            "raw_score": s, "effective_score": s, "threshold": .5,
            "evidence_score": s, "evidence_threshold": .5,
            "selected": selected, "participates": True,
            "fusion_weight": weight, "score_contribution": 0.0,
            "effect_vs_physical": 0.0, "final_influence": 0.0,
            "multilight_local_origin": False, "summary": text,
            "display_text": value_format or text, "telemetry_row": True,
        }

    rows = []
    if m["cnn_per_light"]:
        for light in LIGHTS:
            it = _map(m["cnn_per_light"].get(light))
            if it.get("cnn_active"):
                score = _finite_score(it.get("ng_score_uncalibrated"))
                vote = str(it.get("verdict", "") or "-")
                # Visibilidade do OK: score de evidência é complementar
                # ao NG bruto, mas não vira probabilidade calibrada.
                strength = (
                    (1 - score) if score is not None and vote == "FALHA FALSA"
                    else score
                )
                rows.append(row(
                    f"cnn_{light.lower()}", f"CNN FALTANDO • {light}",
                    strength, text=vote,
                    weight=0.0, selected=True,
                    value_format=f"{vote} • score NG {score*100:.4f}%" if score is not None
                    else f"{vote} • score indisponível",
                ))
            elif it.get("route") == "KNOWN_KNN":
                label = it.get("human_memory_label") or "-"
                rows.append(row(
                    f"knn_{light.lower()}", f"KNN EXATO • {light}", 1.0,
                    text=f"Rótulo humano {label}", weight=0.0,
                    selected=True, value_format=f"Rótulo humano {label} • par exato",
                ))
        if rows:
            return rows

    if m["cnn_active"]:
        score = m["cnn_ng_score"]
        verdict = m["verdict"]
        strength = (
            1 - score if score is not None and verdict == "FALHA FALSA"
            else score
        )
        return [row(
            "cnn_faltando_v2", "CNN FALTANDO v2", strength,
            text=verdict, weight=1.0, selected=True,
            value_format=f"{verdict} • NG {score*100:.4f}% • única CNN"
                if score is not None else "CNN sem score disponível",
        )]

    if m["memory_route"] == "KNOWN_KNN":
        return [row(
            "knn_verified", "KNN • PAR EXATO", 1.0,
            text=f"Rótulo humano {m['memory_label'] or '-'}",
            weight=1.0, selected=True,
            value_format=f"Rótulo {m['memory_label'] or '-'} • memória humana 100%",
        )]
    return []
