"""Formatação pura dos dados usados pelo painel de decisão."""

from __future__ import annotations


ENGINE_LABELS = {
    "adhesive": "Fluxo de adesivo",
    "missing": "Presença do componente",
    "structural": "Comparador estrutural",
    "semantic": "Debug semântico",
    "texture": "Laboratório de textura",
    "knn": "Memória local KNN",
}


def decision_trace_from_analysis(analysis: dict | None) -> dict:
    if not analysis:
        return {}
    detail = analysis.get("detail", {})
    trace = detail.get("decision_trace")
    return trace if isinstance(trace, dict) else {}


def decision_summary(trace: dict) -> str:
    if not trace:
        return "Aguardando rastreamento da decisão."
    final_score = float(trace.get("final_score", 0.0))
    cutoff = float(trace.get("cutoff", 0.45))
    confidence = float(trace.get("confidence", 0.5))
    dominant = ENGINE_LABELS.get(
        str(trace.get("dominant_engine", "none")),
        "Nenhum motor dominante",
    )
    return (
        f"Score final {final_score:.0%} • corte {cutoff:.0%} • "
        f"confiança {confidence:.0%} • dominante: {dominant}"
    )


def fusion_summary(trace: dict) -> str:
    if not trace:
        return "Sem regra de fusão disponível."
    rule_labels = {
        "physical_only": "Somente motores físicos",
        "memory_veto": "Memória substituiu o julgamento físico",
        "memory_priority": "Memória recebeu prioridade de 80%",
        "weighted_physical": "Fusão: físico 70% e KNN 30%",
        "memory_override": "Memória decidiu por alta similaridade",
        "balanced_fusion": "Fusão equilibrada 50/50",
        "weighted_low_similarity": "Memória auxiliar de 30%",
    }
    rule = str(trace.get("fusion_rule", "physical_only"))
    weights = trace.get("weights", {})
    return (
        f"{rule_labels.get(rule, rule)} • "
        f"peso físico {float(weights.get('physical', 1.0)):.0%} • "
        f"peso KNN {float(weights.get('knn', 0.0)):.0%}"
    )


def memory_summary(trace: dict) -> tuple[str, str]:
    memory = trace.get("memory", {}) if trace else {}
    if not memory.get("has_memory", False):
        return "Dataset sem memória compatível.", "SEM MEMÓRIA"

    vote = float(memory.get("vote_defect", 0.5))
    similarity = float(memory.get("best_similarity", 0.0))
    neighbors = int(memory.get("n_neighbors", 0))
    role = str(memory.get("role", "MEMÓRIA AUXILIAR"))
    primary = (
        f"Voto {vote:.0%} NG • similaridade {similarity:.0%} • "
        f"{neighbors} vizinho(s)"
    )
    return primary, role


def _physical_source_id(trace: dict, engines: list[dict]) -> str:
    """Identifica qual motor físico forneceu o score usado na fusão por máximo."""
    physical = [
        engine
        for engine in engines
        if str(engine.get("id", "")) != "knn" and bool(engine.get("active", False))
    ]
    if not physical:
        return ""
    return str(
        max(
            physical,
            key=lambda engine: float(engine.get("effective_score", 0.0)),
        ).get("id", "")
    )


def influence_rows(trace: dict) -> list[dict]:
    """Expõe apenas os motores presentes no rastreamento desta categoria."""
    if not trace:
        return []

    engines = [item for item in trace.get("engines", []) if isinstance(item, dict)]
    weights = trace.get("weights", {}) if isinstance(trace.get("weights"), dict) else {}
    physical_weight = float(weights.get("physical", 1.0))
    knn_weight = float(weights.get("knn", 0.0))
    physical_score = float(trace.get("physical_score", 0.0))
    dominant_id = str(trace.get("dominant_engine", "none"))
    physical_source_id = _physical_source_id(trace, engines)

    rows = []
    for engine in engines:
        engine_id = str(engine.get("id", "unknown"))
        raw_score = float(engine.get("raw_score", 0.0))
        effective_score = float(engine.get("effective_score", 0.0))

        if engine_id == "knn":
            fusion_weight = knn_weight if bool(engine.get("active", False)) else 0.0
            score_contribution = raw_score * fusion_weight
            effect_vs_physical = (raw_score - physical_score) * fusion_weight
        elif engine_id == physical_source_id:
            fusion_weight = physical_weight
            score_contribution = effective_score * fusion_weight
            effect_vs_physical = 0.0
        else:
            fusion_weight = 0.0
            score_contribution = 0.0
            effect_vs_physical = 0.0

        rows.append(
            {
                "id": engine_id,
                "label": str(
                    engine.get(
                        "label",
                        ENGINE_LABELS.get(engine_id, "Motor"),
                    )
                ),
                "active": bool(engine.get("active", False)),
                "triggered": bool(engine.get("triggered", False)),
                "raw_score": raw_score,
                "effective_score": effective_score,
                "threshold": float(engine.get("threshold", 0.45)),
                "selected": engine_id == dominant_id,
                "participates": fusion_weight > 0.0,
                "fusion_weight": max(0.0, min(1.0, fusion_weight)),
                "score_contribution": score_contribution,
                "effect_vs_physical": effect_vs_physical,
                "final_influence": score_contribution,
                "summary": str(engine.get("summary", "")),
            }
        )
    return rows
