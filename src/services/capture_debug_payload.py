"""Payload comum de observabilidade para capturas XP e MSS.

Não importa motores de visão, PyQt ou serviços de captura. Serve apenas para
serializar a decisão já calculada e resumos de imagens.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def image_summary(image: Any) -> dict:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {"valid": False}

    array = np.asarray(image)
    summary = {
        "valid": True,
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
        "min": round(float(np.min(array)), 3),
        "max": round(float(np.max(array)), 3),
        "mean": round(float(np.mean(array)), 3),
    }
    if array.ndim == 3 and array.shape[2] >= 3:
        summary["mean_bgr"] = [
            round(float(np.mean(array[:, :, index])), 3)
            for index in range(3)
        ]
    return summary


def decision_record(analysis: Any, aoi_info: dict | None) -> dict:
    if not isinstance(analysis, dict):
        return {}

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}
    analysis_time_seconds = detail.get("analysis_time_seconds")
    analysis_time_start_source = detail.get("analysis_time_start_source")
    analysis_time_contract = detail.get("analysis_time_contract")
    memory = trace.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}

    missing_fields = (
        "missing_active",
        "missing_is_defect",
        "missing_score",
        "missing_tolerance",
        "missing_classification",
        "missing_changed_coverage",
        "missing_residual_mean",
        "missing_structure_loss",
        "missing_background_exposure",
        "missing_best_similarity",
        "missing_direct_similarity",
        "missing_appearance_loss",
        "missing_edge_mismatch",
        "missing_residual_p90",
        "missing_body_presence_active",
        "missing_component_body_present",
        "missing_body_presence_veto",
        "missing_body_presence_source",
        "missing_body_presence_box",
        "missing_body_coarse_similarity",
        "missing_body_silhouette_dice",
        "missing_body_area_ratio",
        "missing_body_centroid_shift",
        "missing_body_box_width_ratio",
        "missing_body_box_height_ratio",
        "missing_body_presence_policy",
        "missing_body_presence_reason",
        "missing_invariant_occupancy_support",
        "missing_invariant_occupancy_veto",
        "missing_invariant_occupancy_reason",
        "missing_global_envelope_active",
        "missing_global_envelope_support",
        "missing_global_envelope_veto",
        "missing_global_envelope_box",
        "missing_global_envelope_row_profile",
        "missing_global_envelope_col_profile",
        "missing_global_envelope_coarse_similarity",
        "missing_global_envelope_background_exposure",
        "missing_global_envelope_dark_threshold",
        "missing_global_envelope_reference_dark_fraction",
        "missing_global_envelope_test_dark_fraction",
        "missing_global_envelope_dark_retention",
        "missing_global_envelope_invariant_row_profile",
        "missing_global_envelope_invariant_col_profile",
        "missing_global_envelope_invariant_support",
        "missing_global_envelope_reason",
        "missing_hard_absence",
        "missing_hard_absence_reason",
        "missing_cross_category_guard",
        "missing_guard_policy",
        "missing_guard_source_category",
        "missing_guard_physical_support",
        "missing_dual_scale_policy",
        "missing_dual_scale_active",
        "missing_dual_scale_triggered",
        "missing_scale_disagreement",
        "missing_local_global_area_ratio",
        "missing_context_box",
        "missing_context_area_ratio",
        "missing_context_score",
        "missing_context_coverage",
        "missing_context_residual_mean",
        "missing_context_residual_p90",
        "missing_context_structure_loss",
        "missing_context_edge_mismatch",
        "missing_context_direct_similarity",
        "missing_context_appearance_loss",
        "missing_context_best_similarity",
        "missing_context_hard_absence",
        "missing_context_hard_reason",
        "missing_context_physical_support",
    )
    missing = {
        key: json_safe(detail.get(key))
        for key in missing_fields
        if key in detail
    }

    inverted_fields = (
        "inverted_active",
        "inverted_is_defect",
        "inverted_score",
        "inverted_tolerance",
        "inverted_classification",
        "inverted_signature_strength",
        "inverted_test_signature_strength",
        "inverted_direct_similarity",
        "inverted_witness_retention",
        "inverted_witness_loss",
        "inverted_feature_loss",
        "inverted_extra_structure",
        "inverted_topology_mismatch",
        "inverted_orientation_mismatch",
        "inverted_alternate_face_signal",
        "inverted_transform_gain",
        "inverted_best_transform",
        "inverted_best_transform_similarity",
        "inverted_relocation_similarity",
        "inverted_relocation_gain",
        "inverted_relocation_dx",
        "inverted_relocation_dy",
        "inverted_changed_coverage",
        "inverted_witness_coverage",
        "inverted_local_global_area_ratio",
        "inverted_small_witness_roi",
        "inverted_high_authority",
        "inverted_corroborated",
        "inverted_corroboration_reason",
        "inverted_reason",
    )
    inverted = {
        key: json_safe(detail.get(key))
        for key in inverted_fields
        if key in detail
    }

    raw_memory_conflict = bool(
        memory.get(
            "memory_conflict",
            detail.get("memory_conflict", False),
        )
    )
    raw_memory_review = bool(
        memory.get(
            "operator_review_required",
            detail.get("operator_review_required", False),
        )
    )
    raw_hard_missing = bool(
        trace.get(
            "raw_hard_missing_evidence",
            detail.get("missing_hard_absence", False),
        )
        or detail.get("missing_hard_absence", False)
    )
    hard_missing_contradicted_by_exact_ok = bool(
        trace.get("hard_missing_contradicted_by_exact_ok", False)
        or memory.get("hard_missing_contradicted_by_exact_ok", False)
    )
    hard_missing_contradicted_by_invariant_ok = bool(
        trace.get("hard_missing_contradicted_by_invariant_ok", False)
        or memory.get("hard_missing_contradicted_by_invariant_ok", False)
    )
    hard_missing = bool(
        not hard_missing_contradicted_by_exact_ok
        and not hard_missing_contradicted_by_invariant_ok
        and (
            trace.get("hard_missing_evidence", False)
            or memory.get("suppressed_by_hard_missing", False)
            or detail.get("missing_hard_absence", False)
            or str(trace.get("fusion_rule", "")) == "missing_hard_absence"
        )
    )
    effective_memory_conflict = bool(
        raw_memory_conflict and not hard_missing
    )
    effective_memory_review = bool(
        raw_memory_review and not hard_missing
    )

    return {
        "category": str((aoi_info or {}).get("category", "") or ""),
        "is_defect": bool(analysis.get("is_defect", False)),
        "verdict": str(analysis.get("verdict", "") or ""),
        "confidence": json_safe(analysis.get("confidence")),
        "reason": str(analysis.get("reason", "") or ""),
        "analysis_time_seconds": json_safe(analysis_time_seconds),
        "analysis_time_start_source": str(
            analysis_time_start_source or ""
        ),
        "analysis_time_contract": str(analysis_time_contract or ""),
        "final_score": json_safe(detail.get("final_score")),
        "physical_score": json_safe(detail.get("physical_score")),
        "fusion_rule": str(detail.get("fusion_rule", "") or ""),
        "dominant_engine": str(detail.get("dominant_engine", "") or ""),
        "operator_review_required": bool(
            analysis.get("production_review_required", False)
            or trace.get("operator_review_required", False)
            or effective_memory_review
        ),
        "hard_missing_evidence": hard_missing,
        "raw_hard_missing_evidence": raw_hard_missing,
        "hard_missing_contradicted_by_exact_ok": (
            hard_missing_contradicted_by_exact_ok
        ),
        "hard_missing_contradicted_by_invariant_ok": (
            hard_missing_contradicted_by_invariant_ok
        ),
        "missing": missing,
        "inverted": inverted,
        "memory": {
            "has_memory": bool(
                memory.get("has_memory", detail.get("has_memory", False))
            ),
            "memory_available": bool(
                memory.get(
                    "memory_available",
                    detail.get("memory_available", False),
                )
            ),
            "best_match_label": str(
                memory.get(
                    "best_match_label",
                    detail.get("best_match_label", ""),
                )
                or ""
            ),
            "best_similarity": json_safe(
                memory.get(
                    "best_similarity",
                    detail.get("best_similarity"),
                )
            ),
            "best_ok_similarity": json_safe(
                memory.get(
                    "best_ok_similarity",
                    detail.get("best_ok_similarity"),
                )
            ),
            "best_ng_similarity": json_safe(
                memory.get(
                    "best_ng_similarity",
                    detail.get("best_ng_similarity"),
                )
            ),
            "memory_conflict": effective_memory_conflict,
            "raw_memory_conflict": raw_memory_conflict,
            "operator_review_required": effective_memory_review,
            "raw_operator_review_required": raw_memory_review,
            "role": str(
                memory.get("role", detail.get("memory_reason", "")) or ""
            ),
            "suppressed_by_hard_missing": hard_missing,
        },
    }


__all__ = [
    "decision_record",
    "image_summary",
    "json_safe",
]
