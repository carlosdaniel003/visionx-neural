"""Integra o especialista exclusivo de INVERTIDO ao fluxo já existente."""

from __future__ import annotations

import numpy as np

from src.core.anomaly_memory_integration import (
    _dynamic_fusion,
    _focus_box,
    _normalize_knn_memory_categories,
    canonical_category_key,
)
from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.inverted_face_expert import InvertedFaceExpert


INVERTED_KEYS = frozenset({"INVERTIDO"})


def is_inverted_category(category: str) -> bool:
    return canonical_category_key(category) in INVERTED_KEYS


def _combine_signature_mask(detail: dict, mask):
    signature_detail = dict(detail)
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return signature_detail
    current = signature_detail.get("diff_mask")
    if isinstance(current, np.ndarray) and current.shape == mask.shape:
        signature_detail["diff_mask"] = np.maximum(current, mask)
    else:
        signature_detail["diff_mask"] = mask
    signature_detail["inverted_anomaly_mask"] = mask
    signature_detail["inverted_score"] = float(detail.get("inverted_score", 0.0))
    return signature_detail


def _memory_weights(physical_defect, memory_vote, memory_similarity, has_memory):
    physical_weight = 1.0
    memory_weight = 0.0
    fusion_rule = "physical_only"
    memory_role = "SEM MEMÓRIA"
    if not has_memory:
        return physical_weight, memory_weight, fusion_rule, memory_role

    if physical_defect:
        if memory_similarity >= 0.85:
            return 0.0, 1.0, "memory_veto", "VETO DA MEMÓRIA"
        if memory_vote < 0.30 and memory_similarity >= 0.75:
            return 0.20, 0.80, "memory_priority", "MEMÓRIA PRIORITÁRIA"
        return 0.70, 0.30, "weighted_physical", "FUSÃO 70/30"

    if memory_similarity >= 0.85:
        return 0.0, 1.0, "memory_override", "MEMÓRIA DECISIVA"
    if memory_similarity >= 0.75:
        return 0.50, 0.50, "balanced_fusion", "FUSÃO EQUILIBRADA"
    return 0.70, 0.30, "weighted_low_similarity", "MEMÓRIA AUXILIAR"


def _fusion_with_inverted(orchestrator, detail: dict, inverted: dict, knn: dict):
    no_memory = {
        "has_memory": False,
        "vote_defect": 0.5,
        "best_similarity": 0.0,
        "n_neighbors": 0,
    }
    _, _, _, _, base_trace = _dynamic_fusion(
        orchestrator,
        detail,
        "INVERTIDO",
        None,
        no_memory,
    )
    engines = [
        dict(engine)
        for engine in base_trace.get("engines", [])
        if str(engine.get("id", "")) != "knn"
    ]

    active = bool(inverted.get("inverted_active", False))
    raw = float(inverted.get("inverted_score", 0.0))
    threshold = float(inverted.get("inverted_tolerance", 0.43))
    triggered = bool(
        active
        and inverted.get("inverted_is_defect", raw > threshold)
    )
    effective = max(0.90, min(1.0, raw)) if triggered else 0.0
    inverted_entry = orchestrator._engine_entry(
        "inverted",
        "Assinatura da face",
        active,
        triggered,
        raw,
        effective,
        threshold,
        str(inverted.get("inverted_reason", "")),
    )
    engines.insert(0, inverted_entry)

    for engine in engines:
        engine["selected"] = False
        engine["final_influence"] = 0.0

    physical_candidates = [engine for engine in engines if engine.get("active", False)]
    physical_dominant = (
        max(physical_candidates, key=lambda item: float(item.get("effective_score", 0.0)))
        if physical_candidates
        else None
    )
    physical_score = (
        float(physical_dominant.get("effective_score", 0.0))
        if physical_dominant
        else 0.0
    )
    physical_defect = any(bool(engine.get("triggered", False)) for engine in engines)
    if physical_dominant:
        physical_dominant["selected"] = True

    has_memory = bool(knn and knn.get("has_memory", False))
    memory_vote = float(knn.get("vote_defect", 0.5)) if knn else 0.5
    memory_similarity = float(knn.get("best_similarity", 0.0)) if knn else 0.0
    neighbors = int(knn.get("n_neighbors", 0)) if knn else 0
    physical_weight, memory_weight, fusion_rule, memory_role = _memory_weights(
        physical_defect,
        memory_vote,
        memory_similarity,
        has_memory,
    )

    final_score = float(
        np.clip(
            physical_score * physical_weight + memory_vote * memory_weight,
            0.0,
            1.0,
        )
    )
    is_defect = bool(final_score > orchestrator.DECISION_CUTOFF)
    distance_max = (
        1.0 - orchestrator.DECISION_CUTOFF
        if is_defect
        else orchestrator.DECISION_CUTOFF
    )
    current_distance = (
        final_score - orchestrator.DECISION_CUTOFF
        if is_defect
        else orchestrator.DECISION_CUTOFF - final_score
    )
    confidence = float(
        max(
            0.50,
            min(
                0.99,
                0.50 + 0.49 * (current_distance / max(distance_max, 1e-6)),
            ),
        )
    )

    for engine in engines:
        if engine.get("selected", False):
            engine["final_influence"] = float(
                engine.get("effective_score", 0.0) * physical_weight
            )

    memory_entry = orchestrator._engine_entry(
        "knn",
        "Memória local KNN",
        has_memory,
        has_memory and memory_vote > orchestrator.DECISION_CUTOFF,
        memory_vote,
        memory_vote,
        orchestrator.DECISION_CUTOFF,
        (
            f"{memory_role}; similaridade {memory_similarity:.0%}; "
            f"{neighbors} vizinho(s)"
            if has_memory
            else "Dataset sem anomalias compatíveis"
        ),
    )
    memory_entry["selected"] = bool(memory_weight > 0.0)
    memory_entry["final_influence"] = float(memory_vote * memory_weight)
    engines.append(memory_entry)

    if memory_weight >= physical_weight and memory_weight > 0.0:
        dominant_engine = "knn"
    elif physical_dominant:
        dominant_engine = str(physical_dominant.get("id", "none"))
    else:
        dominant_engine = "none"

    reasons = [
        str(engine.get("summary", ""))
        for engine in engines
        if engine.get("triggered", False)
        and str(engine.get("id", "")) != "knn"
        and str(engine.get("summary", ""))
    ]
    if not reasons:
        reasons.append("Sem anomalias físicas significativas")
    reason = " | ".join(reasons)
    if has_memory:
        reason += (
            f" || KNN {memory_vote:.0%} NG; similaridade "
            f"{memory_similarity:.0%}; {memory_role}"
        )

    trace = {
        "schema": orchestrator.DECISION_SCHEMA,
        "cutoff": orchestrator.DECISION_CUTOFF,
        "final_score": final_score,
        "confidence": confidence,
        "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
        "physical_score": physical_score,
        "physical_defect": bool(physical_defect),
        "dominant_engine": dominant_engine,
        "fusion_rule": fusion_rule,
        "weights": {
            "physical": float(physical_weight),
            "knn": float(memory_weight),
        },
        "memory": {
            "has_memory": has_memory,
            "vote_defect": memory_vote,
            "best_similarity": memory_similarity,
            "n_neighbors": neighbors,
            "role": memory_role,
            "best_match_label": str((knn or {}).get("best_match_label", "")),
            "memory_mode": str((knn or {}).get("memory_mode", "none")),
            "memory_scope": str((knn or {}).get("memory_scope", "none")),
        },
        "engines": engines,
    }
    return final_score, is_defect, confidence, reason, trace


def install_inverted_face_integration(orchestrator_cls) -> None:
    if getattr(orchestrator_cls, "_inverted_face_installed", False):
        return

    previous_inspect = orchestrator_cls.inspect

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        analysis = previous_inspect(
            self,
            full_gab,
            full_test,
            raw_anomalies,
            aoi_info,
            global_box_info,
            aoi_epicenters,
        )
        category = str((aoi_info or {}).get("category", "Unknown"))
        if not is_inverted_category(category):
            return analysis

        if "inverted" not in self.experts:
            self.experts["inverted"] = InvertedFaceExpert()
        inverted = self.experts["inverted"].analyze(
            full_gab,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )

        detail = analysis.setdefault("detail", {})
        active_engines = analysis.setdefault("active_engines", [])
        detail.update(inverted)
        if inverted.get("inverted_active", False):
            if "inverted_expert.py" not in active_engines:
                active_engines.insert(0, "inverted_expert.py")
            if inverted.get("inverted_bounding_box"):
                analysis["bounding_box"] = inverted["inverted_bounding_box"]
                analysis.setdefault("all_boxes", {})["inverted"] = inverted[
                    "inverted_bounding_box"
                ]

        focus = _focus_box(aoi_epicenters, analysis, detail)
        signature_detail = _combine_signature_mask(
            detail,
            inverted.get("inverted_anomaly_mask"),
        )
        signature = build_anomaly_signature(
            full_gab,
            full_test,
            signature_detail,
            aoi_info,
            focus,
        )
        knn_expert = self.experts["knn"]
        _normalize_knn_memory_categories(knn_expert)
        knn_result = knn_expert.analyze(
            full_gab,
            full_test,
            None,
            None,
            aoi_info,
            anomaly_signature=signature,
        )

        final_score, is_defect, confidence, reason, trace = _fusion_with_inverted(
            self,
            detail,
            inverted,
            knn_result,
        )
        analysis["is_defect"] = is_defect
        analysis["confidence"] = confidence
        analysis["verdict"] = "DEFEITO REAL" if is_defect else "FALHA FALSA"
        analysis["reason"] = reason
        detail.update(knn_result)
        detail.update(
            {
                "anomaly_signature": signature,
                "query_anomaly_signature": signature,
                "final_score": final_score,
                "physical_score": trace["physical_score"],
                "decision_cutoff": trace["cutoff"],
                "dominant_engine": trace["dominant_engine"],
                "fusion_rule": trace["fusion_rule"],
                "decision_trace": trace,
            }
        )
        return analysis

    orchestrator_cls.inspect = inspect
    orchestrator_cls._inverted_face_installed = True


__all__ = [
    "install_inverted_face_integration",
    "is_inverted_category",
]
