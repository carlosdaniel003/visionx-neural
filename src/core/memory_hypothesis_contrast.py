"""Contraste explícito entre a melhor hipótese NG e a melhor hipótese OK.

Esta camada não altera assinaturas, protótipos, categorias ou pesos dual-scale.
Ela consulta separadamente as duas classes e mede a margem entre as melhores
correspondências. Quando OK e NG são simultaneamente confiáveis e quase
empatados, a memória fica inconclusiva e exige revisão humana.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.core.anomaly_signature import valid_anomaly_signature
from src.core.best_match_memory import (
    BEST_MATCH_MIN_SIMILARITY,
    BEST_MATCH_STRONG_SIMILARITY,
)


HYPOTHESIS_POLICY = "best_ok_vs_best_ng"
HYPOTHESIS_CONFLICT_MARGIN = 0.01
HYPOTHESIS_REVIEW_CONFIDENCE = 0.50


def _label_score(label: str) -> float:
    normalized = str(label or "").strip().upper()
    if normalized == "NG":
        return 1.0
    if normalized == "OK":
        return 0.0
    return 0.5


def _match_strength(similarity: float) -> str:
    if similarity >= BEST_MATCH_STRONG_SIMILARITY:
        return "strong"
    if similarity >= BEST_MATCH_MIN_SIMILARITY:
        return "intermediate"
    return "weak"


def _similarity(item: tuple | None) -> float:
    if item is None:
        return 0.0
    return float(np.clip(1.0 - float(item[0]), 0.0, 1.0))


def _best_for_label(distances: list[tuple], label: str) -> tuple | None:
    normalized = str(label).upper()
    candidates = [
        item
        for item in distances
        if str(item[1]).strip().upper() == normalized
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item[0]))


def _hypothesis_payload(item: tuple | None, label: str) -> dict[str, Any]:
    if item is None:
        return {
            "label": label,
            "available": False,
            "similarity": 0.0,
            "path": "",
            "similarity_breakdown": {},
        }
    return {
        "label": label,
        "available": True,
        "similarity": _similarity(item),
        "path": str(item[2]),
        "similarity_breakdown": item[3] if len(item) > 3 else {},
    }


def build_hypothesis_result(
    *,
    distances: list[tuple],
    top_k: int,
    mode: str,
    scope: str,
    query_anomaly_signature: dict | None = None,
) -> dict:
    """Resolve a memória contrastando explicitamente melhor OK e melhor NG."""
    if not distances:
        return {
            "has_memory": False,
            "memory_available": False,
            "match_reliable": False,
            "vote_defect": 0.5,
            "memory_score": 0.5,
            "best_similarity": 0.0,
            "best_match_label": "",
            "leading_hypothesis": "",
            "best_ok_similarity": 0.0,
            "best_ng_similarity": 0.0,
            "best_ok_path": "",
            "best_ng_path": "",
            "ok_memory_available": False,
            "ng_memory_available": False,
            "hypothesis_margin": None,
            "best_match_margin": 0.0,
            "memory_conflict": False,
            "conflicting_tie": False,
            "operator_review_required": False,
            "memory_policy": HYPOTHESIS_POLICY,
            "quantity_influence": False,
            "memory_mode": "none",
            "memory_scope": "none",
            "match_strength": "none",
            "n_neighbors": 0,
            "neighbor_details": [],
            "hypotheses": {
                "OK": _hypothesis_payload(None, "OK"),
                "NG": _hypothesis_payload(None, "NG"),
            },
            "memory_reason": "Sem memória compatível",
            "query_anomaly_signature": query_anomaly_signature or {},
        }

    ordered = sorted(distances, key=lambda item: float(item[0]))
    best_ok = _best_for_label(ordered, "OK")
    best_ng = _best_for_label(ordered, "NG")
    ok_similarity = _similarity(best_ok)
    ng_similarity = _similarity(best_ng)
    ok_available = best_ok is not None
    ng_available = best_ng is not None

    if ok_available and ng_available:
        margin = abs(ng_similarity - ok_similarity)
        leading_label = "NG" if ng_similarity > ok_similarity else "OK"
        leading_item = best_ng if leading_label == "NG" else best_ok
        opposing_similarity = ok_similarity if leading_label == "NG" else ng_similarity
    elif ng_available:
        margin = None
        leading_label = "NG"
        leading_item = best_ng
        opposing_similarity = 0.0
    elif ok_available:
        margin = None
        leading_label = "OK"
        leading_item = best_ok
        opposing_similarity = 0.0
    else:
        margin = None
        leading_label = ""
        leading_item = None
        opposing_similarity = 0.0

    leading_similarity = _similarity(leading_item)
    both_reliable = bool(
        ok_available
        and ng_available
        and ok_similarity >= BEST_MATCH_MIN_SIMILARITY
        and ng_similarity >= BEST_MATCH_MIN_SIMILARITY
    )
    conflict = bool(
        both_reliable
        and margin is not None
        and margin <= HYPOTHESIS_CONFLICT_MARGIN
    )
    reliable = bool(
        leading_label in {"OK", "NG"}
        and leading_similarity >= BEST_MATCH_MIN_SIMILARITY
        and not conflict
    )

    memory_score = _label_score(leading_label) if reliable else 0.5
    strength = _match_strength(leading_similarity)
    audit_neighbors = ordered[: max(1, int(top_k))]
    details = []
    for rank, item in enumerate(audit_neighbors, start=1):
        label = str(item[1]).strip().upper()
        details.append(
            {
                "rank": rank,
                "label": label,
                "similarity": _similarity(item),
                "distance": float(item[0]),
                "path": str(item[2]),
                "used_for_decision": bool(
                    reliable
                    and label == leading_label
                    and str(item[2]) == str(leading_item[2])
                ),
                "similarity_breakdown": item[3] if len(item) > 3 else {},
            }
        )

    if conflict:
        reason = (
            "CONFLITO DE MEMÓRIA: melhor NG "
            f"{ng_similarity:.1%} x melhor OK {ok_similarity:.1%}; "
            f"margem {float(margin):.1%} <= {HYPOTHESIS_CONFLICT_MARGIN:.1%}. "
            "Memória inconclusiva; operador obrigatório"
        )
    elif not reliable:
        reason = (
            f"Melhor hipótese {leading_label or '-'} com "
            f"{leading_similarity:.1%}, abaixo do mínimo confiável de "
            f"{BEST_MATCH_MIN_SIMILARITY:.0%}"
        )
    else:
        other = (
            f"; hipótese oposta {opposing_similarity:.1%}"
            if ok_available and ng_available
            else "; classe oposta sem memória disponível"
        )
        reason = (
            f"Hipótese {leading_label} venceu com {leading_similarity:.1%}{other}; "
            "quantidade de exemplos ignorada"
        )

    leading_path = str(leading_item[2]) if leading_item is not None else ""
    leading_breakdown = (
        leading_item[3]
        if leading_item is not None and len(leading_item) > 3
        else {}
    )

    return {
        "has_memory": reliable,
        "memory_available": True,
        "match_reliable": reliable,
        "vote_defect": float(memory_score),
        "memory_score": float(memory_score),
        "best_match_score": float(_label_score(leading_label)),
        "best_similarity": float(leading_similarity),
        "best_match_label": leading_label,
        "leading_hypothesis": leading_label,
        "best_match_path": leading_path,
        "best_match_rank": 1,
        "best_ok_similarity": float(ok_similarity),
        "best_ng_similarity": float(ng_similarity),
        "best_ok_path": str(best_ok[2]) if best_ok is not None else "",
        "best_ng_path": str(best_ng[2]) if best_ng is not None else "",
        "ok_memory_available": ok_available,
        "ng_memory_available": ng_available,
        "hypothesis_margin": float(margin) if margin is not None else None,
        "best_match_margin": (
            float(margin) if margin is not None else float(leading_similarity)
        ),
        "opposing_best_similarity": float(opposing_similarity),
        "memory_conflict": conflict,
        "conflicting_tie": conflict,
        "operator_review_required": conflict,
        "conflict_margin_threshold": HYPOTHESIS_CONFLICT_MARGIN,
        "n_neighbors": int(len(audit_neighbors)),
        "memory_candidate_count_compared": int(len(ordered)),
        "neighbor_details": details,
        "memory_label_counts": {
            "OK": int(sum(item["label"] == "OK" for item in details)),
            "NG": int(sum(item["label"] == "NG" for item in details)),
        },
        "memory_mode": mode,
        "memory_scope": scope,
        "memory_policy": HYPOTHESIS_POLICY,
        "quantity_influence": False,
        "match_strength": strength,
        "memory_reason": reason,
        "similarity_breakdown": leading_breakdown,
        "hypotheses": {
            "OK": _hypothesis_payload(best_ok, "OK"),
            "NG": _hypothesis_payload(best_ng, "NG"),
        },
        "query_anomaly_signature": query_anomaly_signature or {},
    }


def analyze_anomaly_hypotheses(
    query_signature: dict,
    valid_ok: list[dict],
    valid_ng: list[dict],
    top_k: int,
    scope: str,
    comparator: Callable[[dict, dict], tuple[float, dict]],
) -> dict:
    distances: list[tuple[float, str, str, dict]] = []
    for label, records in (("OK", valid_ok), ("NG", valid_ng)):
        for item in records:
            stored = item.get("anomaly_signature")
            if not valid_anomaly_signature(stored):
                continue
            try:
                similarity, breakdown = comparator(query_signature, stored)
                similarity = float(np.clip(float(similarity), 0.0, 1.0))
            except Exception:
                continue
            distances.append(
                (
                    1.0 - similarity,
                    label,
                    str(item.get("path", item.get("json_path", ""))),
                    breakdown if isinstance(breakdown, dict) else {},
                )
            )

    result = build_hypothesis_result(
        distances=distances,
        top_k=top_k,
        mode="anomaly",
        scope=scope,
        query_anomaly_signature=query_signature,
    )
    result["memory_schema"] = str(query_signature.get("schema", ""))
    return result


def _fusion_wrapper_factory(original_dynamic_fusion):
    def dynamic_fusion(
        orchestrator,
        detail: dict,
        category: str,
        missing_result: dict | None,
        knn: dict | None,
    ):
        result = original_dynamic_fusion(
            orchestrator,
            detail,
            category,
            missing_result,
            knn,
        )
        final_score, is_defect, confidence, reason, trace = result
        memory = knn if isinstance(knn, dict) else {}
        if not bool(memory.get("memory_conflict", False)):
            return result

        ok_similarity = float(memory.get("best_ok_similarity", 0.0) or 0.0)
        ng_similarity = float(memory.get("best_ng_similarity", 0.0) or 0.0)
        margin = memory.get("hypothesis_margin")
        margin_value = float(margin) if margin is not None else 0.0

        memory_trace = trace.setdefault("memory", {})
        memory_trace.update(
            {
                "has_memory": False,
                "memory_available": True,
                "memory_conflict": True,
                "operator_review_required": True,
                "role": "CONFLITO DE MEMÓRIA",
                "policy": HYPOTHESIS_POLICY,
                "best_ok_similarity": ok_similarity,
                "best_ng_similarity": ng_similarity,
                "hypothesis_margin": margin_value,
                "conflict_margin_threshold": HYPOTHESIS_CONFLICT_MARGIN,
                "quantity_influence": False,
                "memory_score": 0.5,
                "vote_defect": 0.5,
            }
        )
        trace["operator_review_required"] = True
        trace["operator_review_reason"] = "memory_hypothesis_conflict"
        trace["fusion_rule"] = "memory_conflict_operator_review"
        trace["confidence"] = min(float(confidence), HYPOTHESIS_REVIEW_CONFIDENCE)

        for engine in trace.get("engines", []):
            if engine.get("id") == "knn":
                engine.update(
                    {
                        "active": True,
                        "triggered": False,
                        "raw_score": 0.5,
                        "effective_score": 0.5,
                        "selected": False,
                        "final_influence": 0.0,
                        "summary": (
                            f"CONFLITO: NG {ng_similarity:.1%} x OK "
                            f"{ok_similarity:.1%}; margem {margin_value:.1%}; "
                            "operador obrigatório"
                        ),
                    }
                )

        conflict_reason = (
            f"CONFLITO DE MEMÓRIA: NG {ng_similarity:.1%} x OK "
            f"{ok_similarity:.1%}; margem {margin_value:.1%}. "
            "Decisão automática bloqueada; operador deve usar 0=OK ou 1=NG"
        )
        reason = f"{reason} || {conflict_reason}" if reason else conflict_reason
        return (
            final_score,
            is_defect,
            min(float(confidence), HYPOTHESIS_REVIEW_CONFIDENCE),
            reason,
            trace,
        )

    return dynamic_fusion


def install_memory_hypothesis_contrast(
    knn_expert_cls,
    anomaly_memory_module,
    best_match_module,
) -> None:
    """Instala o contraste depois de dual-scale, best-match e protótipos."""
    if getattr(knn_expert_cls, "_memory_hypothesis_contrast_installed", False):
        return

    def analyze_hypotheses(
        self,
        query_signature: dict,
        valid_ok: list[dict],
        valid_ng: list[dict],
        top_k: int,
        scope: str,
    ) -> dict:
        del self
        return analyze_anomaly_hypotheses(
            query_signature,
            valid_ok,
            valid_ng,
            top_k,
            scope,
            best_match_module.compare_anomaly_signatures,
        )

    knn_expert_cls._analyze_anomaly_memory = analyze_hypotheses
    original_fusion = anomaly_memory_module._dynamic_fusion
    anomaly_memory_module._dynamic_fusion = _fusion_wrapper_factory(original_fusion)

    knn_expert_cls._memory_hypothesis_contrast_installed = True
    anomaly_memory_module._memory_hypothesis_contrast_installed = True


__all__ = [
    "HYPOTHESIS_CONFLICT_MARGIN",
    "HYPOTHESIS_POLICY",
    "HYPOTHESIS_REVIEW_CONFIDENCE",
    "analyze_anomaly_hypotheses",
    "build_hypothesis_result",
    "install_memory_hypothesis_contrast",
]
