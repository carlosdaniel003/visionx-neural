"""Formatação pura dos dados usados pelo painel de decisão."""

from __future__ import annotations


ENGINE_LABELS = {
    "adhesive": "Fluxo de adesivo",
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


def influence_rows(trace: dict) -> list[dict]:
    rows = []
    for engine in trace.get("engines", []) if trace else []:
        rows.append(
            {
                "id": str(engine.get("id", "unknown")),
                "label": str(
                    engine.get(
                        "label",
                        ENGINE_LABELS.get(str(engine.get("id", "")), "Motor"),
                    )
                ),
                "active": bool(engine.get("active", False)),
                "triggered": bool(engine.get("triggered", False)),
                "raw_score": float(engine.get("raw_score", 0.0)),
                "effective_score": float(engine.get("effective_score", 0.0)),
                "threshold": float(engine.get("threshold", 0.45)),
                "selected": bool(engine.get("selected", False)),
                "final_influence": float(engine.get("final_influence", 0.0)),
                "summary": str(engine.get("summary", "")),
            }
        )
    return rows
