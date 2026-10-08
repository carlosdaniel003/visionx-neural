"""Telemetria somente leitura do replay SIDE sem KNN.

Extrai uma visão JSON-segura da mesma análise que decidiu o veredito,
sem recalcular escores, modificar limiares, consultar memória ou escrever PNGs.
Campos indisponíveis permanecem ausentes/None, nunca são estimados.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np


SCHEMA = "visionx.side_replay_telemetry.v1"
# Apenas medições físicas: a assinatura KNN e arrays de máscaras/imagens
# não devem vazar para o relatório nem provocar saída JSON gigantesca.
DETAIL_FIELDS = (
    "adhesive_score", "adhesive_tolerance", "adhesive_is_defect",
    "adhesive_reason", "shift_active", "shift_score", "shift_tolerance",
    "missing_active", "missing_score", "missing_tolerance",
    "missing_is_defect", "missing_hard_absence",
    "missing_cross_category_guard", "missing_reason",
    "missing_component_body_present", "missing_coverage",
    "missing_pct", "missing_geometry_only", "missing_coarse_correlation",
    "missing_coarse_similarity", "missing_context_dice",
    "missing_global_envelope_support", "missing_global_background_exposure",
    "inverted_active", "inverted_score", "inverted_tolerance",
    "inverted_is_defect", "inverted_high_authority", "inverted_reason",
    "silk_error_pct", "semantic_loss", "semantic_reason",
    "local_score", "ctx_score", "ssim", "pct_changed",
    "decision_threshold",
)
MAX_BOXES = 40
MAX_TEXT = 900


def _scalar(value: Any) -> bool | int | float | str | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value[:MAX_TEXT]
    return None


def _number(value: Any) -> float | None:
    converted = _scalar(value)
    if isinstance(converted, bool):
        return None
    if isinstance(converted, (float, int)):
        return float(converted)
    return None


def _box(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 4:
        return None
    try:
        coords = [int(v) for v in value[:4]]
    except (ValueError, TypeError, OverflowError):
        return None
    if coords[2] <= 0 or coords[3] <= 0:
        return None
    return coords


def _boxes(items: Any) -> list[list[int]]:
    if not isinstance(items, (list, tuple)):
        return []
    return [v for item in items[:MAX_BOXES] if (v := _box(item)) is not None]


def _global_box(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {key: _scalar(value.get(key)) for key in ("x", "y", "w", "h", "detected")}
    return {key: val for key, val in result.items() if val is not None}


def geometry_snapshot(context: dict, analysis: dict, reference, test) -> dict:
    """Caixas no referencial da área AOI extraída, sem recortar mais a imagem."""
    context = context if isinstance(context, dict) else {}
    analysis = analysis if isinstance(analysis, dict) else {}
    raw = _boxes(context.get("raw_anomalies"))
    old = _boxes(context.get("old_epicenters"))
    epicenters = _boxes(context.get("real_epicenters"))
    detail = analysis.get("detail", {}) or {}
    source_boxes = analysis.get("all_boxes") or {}
    specialist_boxes = {
        str(key): val
        for key, item in source_boxes.items()
        if (val := _box(item)) is not None
    } if isinstance(source_boxes, dict) else {}
    for key in (
        "missing_bounding_box", "inverted_bounding_box",
        "semantic_bounding_box", "silk_bounding_box",
    ):
        box = _box(detail.get(key))
        if box is not None:
            specialist_boxes.setdefault(key, box)

    ref_h, ref_w = reference.shape[:2]
    test_h, test_w = test.shape[:2]
    return {
        "coordinate_system": "AOI_EXTRACTED_IMAGE_XYWH",
        "reference_width": int(ref_w),
        "reference_height": int(ref_h),
        "test_width": int(test_w),
        "test_height": int(test_h),
        "input_dimensions_equal": bool((ref_w, ref_h) == (test_w, test_h)),
        "global_box": _global_box(context.get("global_box_info")),
        "raw_anomaly_count": len(context.get("raw_anomalies") or []),
        "raw_anomaly_boxes": raw,
        "aoi_epicenter_boxes": old,
        "selected_epicenter_boxes": epicenters,
        "selected_epicenter_source": (
            "EPICENTER_EXTRACTOR" if epicenters else "NONE"
        ),
        "final_bounding_box": _box(analysis.get("bounding_box")),
        "specialist_boxes": specialist_boxes,
        "box_limit": MAX_BOXES,
        "boxes_truncated": any(
            len(context.get(k) or []) > MAX_BOXES
            for k in ("raw_anomalies", "old_epicenters", "real_epicenters")
        ),
    }


def decision_snapshot(analysis: dict) -> dict:
    """Copia literalmente os especialistas da fusão, sem inventar scores."""
    detail = analysis.get("detail", {}) or {}
    trace = detail.get("decision_trace", {}) or {}
    if not isinstance(trace, dict):
        trace = {}
    engine_rows = []
    for source in trace.get("engines", []) or []:
        if not isinstance(source, dict):
            continue
        engine_id = str(source.get("id", "") or "")
        # KNN fica fora do quadro de *evidências físicas*, embora seu
        # peso zero continue rastreável no contrato separado sem memória.
        if engine_id.lower() == "knn":
            continue
        engine_rows.append({
            "id": engine_id,
            "label": str(source.get("label", "") or "")[:MAX_TEXT],
            "active": bool(source.get("active", False)),
            "triggered": bool(source.get("triggered", False)),
            "selected": bool(source.get("selected", False)),
            "raw_score": _number(source.get("raw_score")),
            "effective_score": _number(source.get("effective_score")),
            "threshold": _number(source.get("threshold")),
            "final_influence": _number(source.get("final_influence")),
            "summary": str(source.get("summary", "") or "")[:MAX_TEXT],
        })

    readings = {
        field: value
        for field in DETAIL_FIELDS
        if (value := _scalar(detail.get(field))) is not None
    }
    return {
        "schema": SCHEMA,
        "reason": str(analysis.get("reason", "") or "")[:MAX_TEXT],
        "final_score": _number(trace.get("final_score")),
        "physical_score": _number(trace.get("physical_score")),
        "cutoff": _number(trace.get("cutoff")),
        "dominant_engine": str(trace.get("dominant_engine", "") or ""),
        "fusion_rule": str(trace.get("fusion_rule", "") or ""),
        "physical_defect": bool(trace.get("physical_defect", False)),
        "hard_missing_evidence": bool(trace.get("hard_missing_evidence", False)),
        "raw_hard_missing_evidence": bool(
            trace.get("raw_hard_missing_evidence", False)
        ),
        "operator_review_required": bool(
            trace.get("operator_review_required", False)
        ),
        "weights": {
            "physical": _number((trace.get("weights") or {}).get("physical")),
            "knn": _number((trace.get("weights") or {}).get("knn")),
        },
        "engines": engine_rows,
        "physical_readings": readings,
        "memory_consulted": False,
        "knn_enabled": False,
    }


def aggregate_replay_cases(cases: list[dict]) -> dict:
    """Distribuição de erros por categoria, rótulo, regra e motor dominante."""
    groups: dict[tuple[str, str], dict] = {}
    by_rule = Counter()
    by_engine = Counter()
    triggered_regressions = Counter()
    for item in cases:
        label = str(item.get("expected_label", "") or "")
        category = str(item.get("category", item.get("category_hint", "UNKNOWN")) or "")
        status = str(item.get("status", "") or "")
        key = (label, category)
        if key not in groups:
            groups[key] = {
                "label": label, "category": category,
                "total": 0, "passed": 0, "regressions": 0, "invalid": 0,
            }
        group = groups[key]
        group["total"] += 1
        if status == "PASSOU":
            group["passed"] += 1
        elif status == "REGRESSAO":
            group["regressions"] += 1
        else:
            group["invalid"] += 1

        telemetry = item.get("telemetry") or {}
        if status != "INVALIDO" and telemetry:
            rule = telemetry.get("fusion_rule") or "UNAVAILABLE"
            dominant = telemetry.get("dominant_engine") or "UNAVAILABLE"
            by_rule[f"{label}|{status}|{rule}"] += 1
            by_engine[f"{label}|{status}|{dominant}"] += 1
            if status == "REGRESSAO":
                for engine in telemetry.get("engines", []) or []:
                    if engine.get("triggered"):
                        triggered_regressions[
                            f"{category}|{engine.get('id', 'UNKNOWN')}"
                        ] += 1

    return {
        "by_category_and_label": [
            groups[k] for k in sorted(groups)
        ],
        "by_label_status_fusion_rule": dict(sorted(by_rule.items())),
        "by_label_status_dominant_engine": dict(sorted(by_engine.items())),
        "triggered_engines_in_regressions_by_category": dict(
            sorted(triggered_regressions.items())
        ),
        "notes": (
            "Os motores acionados podem coexistir no mesmo caso; "
            "contagens por motor não são mutuamente exclusivas."
        ),
    }


__all__ = [
    "SCHEMA", "decision_snapshot", "geometry_snapshot",
    "aggregate_replay_cases",
]
