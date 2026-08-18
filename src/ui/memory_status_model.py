"""Modelo de apresentação da memória visual.

Somente traduz os campos já calculados pelos motores em valores adequados à UI.
Não participa de similaridade, fusão, classificação ou persistência.
"""

from __future__ import annotations

from typing import Any


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return max(0.0, min(1.0, result))


def _int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _pick(mapping: dict, key: str, fallback: Any = None) -> Any:
    return mapping[key] if key in mapping else fallback


def memory_status_from_detail(detail: dict | None) -> dict:
    payload = _dict(detail)
    trace = _dict(payload.get("decision_trace"))
    memory = _dict(trace.get("memory"))
    prototype_stats = _dict(payload.get("memory_prototype_stats"))
    breakdown = _dict(payload.get("similarity_breakdown"))

    conflict = bool(
        _pick(memory, "memory_conflict", payload.get("memory_conflict", False))
    )
    review_required = bool(
        _pick(
            memory,
            "operator_review_required",
            payload.get(
                "operator_review_required",
                trace.get("operator_review_required", False),
            ),
        )
    )
    has_memory = bool(
        _pick(memory, "has_memory", payload.get("has_memory", False))
    )
    memory_available = bool(
        _pick(
            memory,
            "memory_available",
            payload.get("memory_available", has_memory or conflict),
        )
    )

    best_ok = _float(
        _pick(memory, "best_ok_similarity", payload.get("best_ok_similarity", 0.0))
    )
    best_ng = _float(
        _pick(memory, "best_ng_similarity", payload.get("best_ng_similarity", 0.0))
    )
    combined = _float(
        _pick(memory, "best_similarity", payload.get("best_similarity", 0.0))
    )

    margin_raw = _pick(
        memory,
        "hypothesis_margin",
        payload.get("hypothesis_margin"),
    )
    margin = None if margin_raw is None else _float(margin_raw)

    leading = str(
        _pick(
            memory,
            "best_match_label",
            payload.get("leading_hypothesis", payload.get("best_match_label", "")),
        )
        or ""
    ).strip().upper()

    epicenter = breakdown.get("epicenter_similarity")
    context = breakdown.get("context_similarity")
    if epicenter is None and breakdown.get("policy") == "legacy_epicenter_only":
        epicenter = combined
    epicenter_similarity = None if epicenter is None else _float(epicenter)
    context_similarity = None if context is None else _float(context)

    dual_scale = bool(
        breakdown.get("dual_scale", False)
        or context_similarity is not None
    )
    scale_weights = _dict(breakdown.get("scale_weights"))
    epicenter_weight = _float(scale_weights.get("epicenter", 0.70), 0.70)
    context_weight = _float(scale_weights.get("component_context", 0.30), 0.30)
    if not dual_scale:
        epicenter_weight, context_weight = 1.0, 0.0

    memory_score = _float(
        _pick(
            memory,
            "memory_score",
            payload.get("memory_score", payload.get("vote_defect", 0.5)),
        ),
        0.5,
    )
    quantity_influence = bool(
        _pick(
            memory,
            "quantity_influence",
            payload.get("memory_quantity_influence", payload.get("quantity_influence", False)),
        )
    )

    return {
        "active": bool(payload),
        "has_memory": has_memory,
        "memory_available": memory_available,
        "conflict": conflict,
        "review_required": review_required,
        "role": str(memory.get("role", payload.get("memory_reason", "MEMÓRIA"))),
        "policy": str(memory.get("policy", payload.get("memory_policy", ""))),
        "scope": str(memory.get("memory_scope", payload.get("memory_scope", "none"))),
        "category": str(payload.get("memory_category", "")).upper(),
        "leading_hypothesis": leading,
        "best_ok_similarity": best_ok,
        "best_ng_similarity": best_ng,
        "hypothesis_margin": margin,
        "conflict_margin_threshold": _float(
            payload.get(
                "conflict_margin_threshold",
                memory.get("conflict_margin_threshold", 0.01),
            ),
            0.01,
        ),
        "combined_similarity": combined,
        "epicenter_similarity": epicenter_similarity,
        "context_similarity": context_similarity,
        "dual_scale": dual_scale,
        "epicenter_weight": epicenter_weight,
        "context_weight": context_weight,
        "memory_score": memory_score,
        "quantity_influence": quantity_influence,
        "n_neighbors": _int(payload.get("n_neighbors", memory.get("n_neighbors", 0))),
        "ok_prototypes": _int(prototype_stats.get("ok_prototypes", 0)),
        "ok_observations": _int(prototype_stats.get("ok_observations", 0)),
        "raw_ok_jsons": _int(prototype_stats.get("raw_ok_jsons", 0)),
        "protected_ng": _int(prototype_stats.get("protected_ng_prototypes", 0)),
        "raw_ng_jsons": _int(prototype_stats.get("raw_ng_jsons", 0)),
        "prototype_stats_available": bool(prototype_stats),
    }


def _pct(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.{decimals}f}%"


def memory_summary_text(detail: dict | None) -> str:
    model = memory_status_from_detail(detail)
    if not model["active"]:
        return "Sem dados no momento."

    if model["conflict"]:
        return (
            "CONFLITO DE MEMÓRIA • "
            f"NG {_pct(model['best_ng_similarity'])} × "
            f"OK {_pct(model['best_ok_similarity'])} • "
            f"margem {_pct(model['hypothesis_margin'])} • revisão obrigatória"
        )

    if model["has_memory"]:
        leader = model["leading_hypothesis"] or "-"
        parts = [
            f"Hipótese {leader}",
            f"NG {_pct(model['best_ng_similarity'])} × OK {_pct(model['best_ok_similarity'])}",
        ]
        if model["hypothesis_margin"] is not None:
            parts.append(f"margem {_pct(model['hypothesis_margin'])}")
        if model["dual_scale"]:
            parts.append(
                f"epicentro {_pct(model['epicenter_similarity'])} / "
                f"contexto {_pct(model['context_similarity'])}"
            )
        else:
            parts.append(f"similaridade {_pct(model['combined_similarity'])}")
        return " • ".join(parts)

    if model["memory_available"]:
        return "Memória disponível, mas sem correspondência visual confiável."
    return "Sem memória compatível para esta inspeção."


__all__ = ["memory_status_from_detail", "memory_summary_text"]
