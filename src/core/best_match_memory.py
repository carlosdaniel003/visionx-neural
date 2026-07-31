"""Memória visual por melhor correspondência, sem votação por quantidade.

A assinatura da anomalia continua sendo comparada com todos os registros da
mesma categoria. Somente o registro visualmente mais compatível fornece o
rótulo da memória. Os vizinhos seguintes são mantidos apenas para auditoria.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.anomaly_signature import (
    compare_anomaly_signatures,
    valid_anomaly_signature,
)


BEST_MATCH_MIN_SIMILARITY = 0.75
BEST_MATCH_STRONG_SIMILARITY = 0.90
BEST_MATCH_CONFLICT_MARGIN = 0.0025
BEST_MATCH_POLICY = "best_match_visual"


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


def _confidence(final_score: float, cutoff: float) -> float:
    is_defect = final_score > cutoff
    distance_max = 1.0 - cutoff if is_defect else cutoff
    current_distance = (
        final_score - cutoff if is_defect else cutoff - final_score
    )
    return float(
        max(
            0.50,
            min(
                0.99,
                0.50
                + 0.49 * (current_distance / max(distance_max, 1e-6)),
            ),
        )
    )


def _build_result(
    *,
    distances: list[tuple],
    top_k: int,
    mode: str,
    scope: str,
    query_anomaly_signature: dict | None = None,
    query_embedding: np.ndarray | None = None,
) -> dict:
    """Converte comparações ordenadas em uma decisão por melhor ocorrência."""
    if not distances:
        return {
            "has_memory": False,
            "memory_available": False,
            "match_reliable": False,
            "vote_defect": 0.5,
            "memory_score": 0.5,
            "best_similarity": 0.0,
            "n_neighbors": 0,
            "best_match_label": "",
            "neighbor_details": [],
            "memory_mode": "none",
            "memory_scope": "none",
            "memory_policy": BEST_MATCH_POLICY,
            "quantity_influence": False,
            "match_strength": "none",
            "query_anomaly_signature": query_anomaly_signature or {},
            "query_embedding": (
                query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else []
            ),
        }

    ordered = sorted(distances, key=lambda item: float(item[0]))
    audit_neighbors = ordered[: max(1, int(top_k))]
    best = ordered[0]
    best_distance = float(best[0])
    best_label = str(best[1]).strip().upper()
    best_path = str(best[2])
    best_breakdown = best[3] if len(best) > 3 else {}
    best_similarity = float(np.clip(1.0 - best_distance, 0.0, 1.0))

    opposing_similarities = [
        float(np.clip(1.0 - float(item[0]), 0.0, 1.0))
        for item in ordered
        if str(item[1]).strip().upper() in {"OK", "NG"}
        and str(item[1]).strip().upper() != best_label
    ]
    opposing_best = max(opposing_similarities) if opposing_similarities else 0.0
    decision_margin = best_similarity - opposing_best
    conflicting_tie = bool(
        opposing_similarities
        and decision_margin < BEST_MATCH_CONFLICT_MARGIN
    )

    strength = _match_strength(best_similarity)
    reliable = bool(
        best_label in {"OK", "NG"}
        and best_similarity >= BEST_MATCH_MIN_SIMILARITY
        and not conflicting_tie
    )
    best_score = _label_score(best_label)
    memory_score = best_score if reliable else 0.5

    details = []
    for rank, item in enumerate(audit_neighbors, start=1):
        distance = float(item[0])
        detail = {
            "rank": rank,
            "label": str(item[1]).strip().upper(),
            "similarity": float(np.clip(1.0 - distance, 0.0, 1.0)),
            "distance": distance,
            "path": str(item[2]),
            "used_for_decision": rank == 1 and reliable,
        }
        if len(item) > 3:
            detail["similarity_breakdown"] = item[3]
        details.append(detail)

    same_label_count = sum(
        item["label"] == best_label for item in details
    )
    neighbor_consistency = (
        same_label_count / len(details) if details else 0.0
    )

    if conflicting_tie:
        reason = (
            "Melhores correspondências OK e NG praticamente empatadas; "
            "memória mantida inconclusiva"
        )
    elif not reliable:
        reason = (
            f"Melhor correspondência {best_label or '-'} com "
            f"{best_similarity:.1%}, abaixo do mínimo de "
            f"{BEST_MATCH_MIN_SIMILARITY:.0%}"
        )
    else:
        reason = (
            f"Melhor correspondência individual {best_label} com "
            f"{best_similarity:.1%}; quantidade de exemplos ignorada"
        )

    return {
        "has_memory": reliable,
        "memory_available": True,
        "match_reliable": reliable,
        "vote_defect": float(memory_score),
        "memory_score": float(memory_score),
        "best_match_score": float(best_score),
        "best_similarity": best_similarity,
        "n_neighbors": int(len(audit_neighbors)),
        "memory_candidate_count_compared": int(len(ordered)),
        "best_match_path": best_path,
        "best_match_label": best_label,
        "best_match_rank": 1,
        "neighbor_details": details,
        "memory_label_counts": {
            "OK": int(sum(item["label"] == "OK" for item in details)),
            "NG": int(sum(item["label"] == "NG" for item in details)),
        },
        "neighbor_consistency": float(neighbor_consistency),
        "opposing_best_similarity": float(opposing_best),
        "best_match_margin": float(decision_margin),
        "conflicting_tie": conflicting_tie,
        "memory_mode": mode,
        "memory_scope": scope,
        "memory_policy": BEST_MATCH_POLICY,
        "quantity_influence": False,
        "match_strength": strength,
        "memory_reason": reason,
        "similarity_breakdown": best_breakdown,
        "query_anomaly_signature": query_anomaly_signature or {},
        "query_embedding": (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else []
        ),
    }


def _analyze_anomaly_memory_best_match(
    self,
    query_signature: dict,
    valid_ok: list[dict],
    valid_ng: list[dict],
    top_k: int,
    scope: str,
) -> dict:
    distances: list[tuple[float, str, str, dict]] = []
    for label, records in (("OK", valid_ok), ("NG", valid_ng)):
        for item in records:
            stored = item.get("anomaly_signature")
            if not valid_anomaly_signature(stored):
                continue
            similarity, breakdown = compare_anomaly_signatures(
                query_signature,
                stored,
            )
            distances.append(
                (
                    1.0 - float(similarity),
                    label,
                    str(item.get("path", item.get("json_path", ""))),
                    breakdown,
                )
            )

    result = _build_result(
        distances=distances,
        top_k=top_k,
        mode="anomaly",
        scope=scope,
        query_anomaly_signature=query_signature,
    )
    result["memory_schema"] = str(query_signature.get("schema", ""))
    return result


def _analyze_legacy_memory_best_match(
    self,
    query_image: np.ndarray,
    valid_ok: list[dict],
    valid_ng: list[dict],
    top_k: int,
    scope: str,
    query_anomaly_signature: dict | None,
) -> dict:
    query_signature = self._compute_embedding(query_image)
    if query_signature is None:
        return _build_result(
            distances=[],
            top_k=top_k,
            mode="legacy_image",
            scope=scope,
            query_anomaly_signature=query_anomaly_signature,
        )

    distances: list[tuple[float, str, str]] = []
    for label, records in (("OK", valid_ok), ("NG", valid_ng)):
        for item in records:
            stored = item.get("sig")
            if not isinstance(stored, np.ndarray):
                continue
            similarity = self._cosine_similarity(query_signature, stored)
            if similarity is None:
                continue
            distances.append(
                (
                    1.0 - float(similarity),
                    label,
                    str(item.get("path", item.get("json_path", ""))),
                )
            )

    return _build_result(
        distances=distances,
        top_k=top_k,
        mode="legacy_image",
        scope=scope,
        query_anomaly_signature=query_anomaly_signature,
        query_embedding=query_signature,
    )


def _best_match_dynamic_fusion_factory(original_dynamic_fusion):
    def dynamic_fusion(
        orchestrator,
        detail: dict,
        category: str,
        missing_result: dict | None,
        knn: dict | None,
    ):
        # Primeiro calcula exclusivamente os motores físicos. Isso evita que a
        # antiga votação KNN participe antes da aplicação da nova política.
        physical_knn = dict(knn or {})
        physical_knn["has_memory"] = False
        physical_knn["vote_defect"] = 0.5
        (
            base_score,
            base_is_defect,
            base_confidence,
            base_reason,
            trace,
        ) = original_dynamic_fusion(
            orchestrator,
            detail,
            category,
            missing_result,
            physical_knn,
        )

        memory = knn if isinstance(knn, dict) else {}
        best_label = str(memory.get("best_match_label", "")).strip().upper()
        similarity = float(memory.get("best_similarity", 0.0) or 0.0)
        reliable = bool(
            memory.get("has_memory", False)
            and memory.get("match_reliable", True)
            and best_label in {"OK", "NG"}
            and similarity >= BEST_MATCH_MIN_SIMILARITY
        )

        memory_trace = trace.setdefault("memory", {})
        memory_trace.update(
            {
                "has_memory": reliable,
                "memory_available": bool(memory.get("memory_available", False)),
                "best_match_label": best_label,
                "best_similarity": similarity,
                "memory_score": float(_label_score(best_label)) if reliable else 0.5,
                "vote_defect": float(_label_score(best_label)) if reliable else 0.5,
                "n_neighbors": int(memory.get("n_neighbors", 0) or 0),
                "role": "SEM CORRESPONDÊNCIA CONFIÁVEL",
                "memory_mode": str(memory.get("memory_mode", "none")),
                "memory_scope": str(memory.get("memory_scope", "none")),
                "policy": BEST_MATCH_POLICY,
                "quantity_influence": False,
                "match_strength": str(memory.get("match_strength", "none")),
                "conflicting_tie": bool(memory.get("conflicting_tie", False)),
            }
        )

        knn_engine = next(
            (engine for engine in trace.get("engines", []) if engine.get("id") == "knn"),
            None,
        )

        if not reliable:
            if knn_engine is not None:
                knn_engine.update(
                    {
                        "active": False,
                        "triggered": False,
                        "raw_score": 0.5,
                        "effective_score": 0.5,
                        "selected": False,
                        "final_influence": 0.0,
                        "summary": str(
                            memory.get(
                                "memory_reason",
                                "Sem correspondência visual confiável",
                            )
                        ),
                    }
                )
            if memory.get("memory_available", False):
                base_reason += " || " + str(
                    memory.get(
                        "memory_reason",
                        "Memória sem correspondência visual confiável",
                    )
                )
            return (
                base_score,
                base_is_defect,
                base_confidence,
                base_reason,
                trace,
            )

        memory_score = _label_score(best_label)
        physical_score = float(trace.get("physical_score", base_score))

        if similarity >= BEST_MATCH_STRONG_SIMILARITY:
            physical_weight = 0.0
            memory_weight = 1.0
            fusion_rule = "best_match_strong"
            memory_role = "MELHOR CORRESPONDÊNCIA DECISIVA"
        else:
            progress = float(
                np.clip(
                    (similarity - BEST_MATCH_MIN_SIMILARITY)
                    / (
                        BEST_MATCH_STRONG_SIMILARITY
                        - BEST_MATCH_MIN_SIMILARITY
                    ),
                    0.0,
                    1.0,
                )
            )
            memory_weight = 0.55 + 0.30 * progress
            physical_weight = 1.0 - memory_weight
            fusion_rule = "best_match_intermediate"
            memory_role = "MELHOR CORRESPONDÊNCIA INTERMEDIÁRIA"

        final_score = float(
            np.clip(
                physical_score * physical_weight
                + memory_score * memory_weight,
                0.0,
                1.0,
            )
        )
        cutoff = float(orchestrator.DECISION_CUTOFF)
        is_defect = bool(final_score > cutoff)
        confidence = _confidence(final_score, cutoff)

        for engine in trace.get("engines", []):
            if engine.get("id") == "knn":
                engine.update(
                    {
                        "active": True,
                        "triggered": best_label == "NG",
                        "raw_score": float(memory_score),
                        "effective_score": float(memory_score),
                        "threshold": cutoff,
                        "selected": True,
                        "final_influence": float(memory_score * memory_weight),
                        "summary": (
                            f"{memory_role}; melhor correspondência {best_label} "
                            f"{similarity:.1%}; quantidade ignorada"
                        ),
                    }
                )
            elif engine.get("selected", False):
                engine["final_influence"] = float(
                    float(engine.get("effective_score", 0.0))
                    * physical_weight
                )

        dominant_engine = (
            "knn"
            if memory_weight >= physical_weight
            else str(trace.get("dominant_engine", "none"))
        )
        verdict = "DEFEITO REAL" if is_defect else "FALHA FALSA"

        trace.update(
            {
                "final_score": final_score,
                "confidence": confidence,
                "verdict": verdict,
                "dominant_engine": dominant_engine,
                "fusion_rule": fusion_rule,
                "weights": {
                    "physical": float(physical_weight),
                    "knn": float(memory_weight),
                },
            }
        )
        memory_trace.update(
            {
                "has_memory": True,
                "vote_defect": float(memory_score),
                "memory_score": float(memory_score),
                "best_similarity": similarity,
                "best_match_label": best_label,
                "role": memory_role,
                "policy": BEST_MATCH_POLICY,
                "quantity_influence": False,
            }
        )

        reason = (
            f"{base_reason} || Memória por melhor correspondência: "
            f"{best_label} com {similarity:.1%}; quantidade de exemplos ignorada"
        )
        return final_score, is_defect, confidence, reason, trace

    return dynamic_fusion


def install_best_match_memory(knn_expert_cls, anomaly_memory_module) -> None:
    """Instala a política como última camada da memória e da fusão."""
    if not getattr(knn_expert_cls, "_best_match_memory_installed", False):
        knn_expert_cls._analyze_anomaly_memory = _analyze_anomaly_memory_best_match
        knn_expert_cls._analyze_legacy_memory = _analyze_legacy_memory_best_match
        knn_expert_cls._best_match_memory_installed = True

    if not getattr(anomaly_memory_module, "_best_match_memory_installed", False):
        original = anomaly_memory_module._dynamic_fusion
        anomaly_memory_module._dynamic_fusion = (
            _best_match_dynamic_fusion_factory(original)
        )
        anomaly_memory_module._best_match_memory_installed = True


__all__ = [
    "BEST_MATCH_CONFLICT_MARGIN",
    "BEST_MATCH_MIN_SIMILARITY",
    "BEST_MATCH_POLICY",
    "BEST_MATCH_STRONG_SIMILARITY",
    "install_best_match_memory",
]
