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
            "ng_score_uncalibrated": None,
            "known_memory_label": "",
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
    detail = (analysis or {}).get("detail", {}) if isinstance(analysis, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    neighbor_report = detail.get("knn_visual_corroboration") or {}
    neighbor_report = neighbor_report if isinstance(neighbor_report, dict) else {}
    physical = detail.get("physical_missing_veto") or {}
    physical = physical if isinstance(physical, dict) else {}
    contradictory = bool(
        m["verdict"] in {"DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}
        and (neighbor_report.get("status") == "NG_CORROBORATED"
             or physical.get("status") == "SUSPECT_ABSENCE")
    )
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
                entry = row(
                    f"cnn_{light.lower()}", f"CNN FALTANDO • {light}",
                    strength, text=vote,
                    weight=0.0, selected=True,
                    value_format=f"{vote} • score NG {score*100:.4f}%" if score is not None
                    else f"{vote} • score indisponível",
                )
                entry["ng_score_uncalibrated"] = score
                entry["contradicted"] = bool(contradictory and vote == "FALHA FALSA")
                rows.append(entry)
            elif it.get("route") == "KNOWN_KNN":
                label = it.get("human_memory_label") or "-"
                entry = row(
                    f"knn_{light.lower()}", f"KNN EXATO • {light}", 1.0,
                    text=f"Rótulo humano {label}", weight=0.0,
                    selected=True, value_format=f"Rótulo humano {label} • par exato",
                )
                entry["known_memory_label"] = label
                rows.append(entry)
        # Motor KNN aproximado: mostrar sua evidência separadamente.
        # O número exibido é similaridade de assinatura, não score CNN.
        by_light = neighbor_report.get("lights")
        by_light = by_light if isinstance(by_light, dict) else {}
        strong = []
        observed = []
        for light in LIGHTS:
            info = _map(by_light.get(light))
            candidates = _map(info.get("ng"))
            nearest = _finite_score(candidates.get("best_similarity"))
            if nearest is not None:
                observed.append((light, nearest))
            if info.get("status") == "NG_SIMILAR":
                strong.append(light)
        # Uma linha única mantém legibilidade no painel de quatro cartões;
        # o detalhamento completo SIDE/TOP/MID fica no debug do resultado.
        if observed:
            strongest = max(score for _, score in observed)
            status = "NG POR SEMELHANÇA" if len(strong) >= 2 else "VIZINHOS SEM CONSENSO"
            display = (
                f"{len(strong)}/3 luzes NG fortes • "
                f"melhor similaridade {strongest * 100:.2f}% (não probabilística)"
            )
            item = row(
                "knn_visual_summary", "KNN • VIZINHOS",
                strongest, text=status,
                weight=0.0, selected=False,
                value_format=display,
            )
            item["knn_approximate"] = True
            item["knn_visual_label"] = "NG" if len(strong) >= 2 else ""
            item["known_memory_label"] = ""
            item["ng_score_uncalibrated"] = None
            item["similarity_score"] = strongest
            rows.append(item)
        if rows:
            return rows

    if m["cnn_active"]:
        score = m["cnn_ng_score"]
        verdict = m["verdict"]
        strength = (
            1 - score if score is not None and verdict == "FALHA FALSA"
            else score
        )
        entry = row(
            "cnn_faltando_v2", "CNN FALTANDO v2", strength,
            text=verdict, weight=1.0, selected=True,
            value_format=f"{verdict} • NG {score*100:.4f}% • única CNN"
                if score is not None else "CNN sem score disponível",
        )
        entry["ng_score_uncalibrated"] = score
        return [entry]

    if m["memory_route"] == "KNOWN_KNN":
        entry = row(
            "knn_verified", "KNN • PAR EXATO", 1.0,
            text=f"Rótulo humano {m['memory_label'] or '-'}",
            weight=1.0, selected=True,
            value_format=f"Rótulo {m['memory_label'] or '-'} • histórico confirmado",
        )
        entry["known_memory_label"] = m["memory_label"]
        return [entry]
    return []
