"""Assinatura compacta da divergência visual para memória KNN."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "visionx.anomaly.v1"
SEMANTIC_SIZE = 128
SPATIAL_SIZE = 16
MAP_SIDE = 8
MAP_SIZE = MAP_SIDE * MAP_SIDE
PHYSICS_SIZE = 16
VECTOR_SIZE = SEMANTIC_SIZE + SPATIAL_SIZE + MAP_SIZE + PHYSICS_SIZE

RANGES = {
    "semantic_delta": (0, SEMANTIC_SIZE),
    "spatial_grid": (SEMANTIC_SIZE, SEMANTIC_SIZE + SPATIAL_SIZE),
    "anomaly_map": (
        SEMANTIC_SIZE + SPATIAL_SIZE,
        SEMANTIC_SIZE + SPATIAL_SIZE + MAP_SIZE,
    ),
    "physics": (
        SEMANTIC_SIZE + SPATIAL_SIZE + MAP_SIZE,
        VECTOR_SIZE,
    ),
}

GROUP_WEIGHTS = {
    "semantic_delta": 0.25,
    "spatial_grid": 0.25,
    "anomaly_map": 0.35,
    "physics": 0.15,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _fixed_vector(value: Any, size: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        array = np.empty(0, dtype=np.float32)
    array = array[np.isfinite(array)]
    output = np.zeros(size, dtype=np.float32)
    if array.size:
        output[: min(size, array.size)] = array[:size]
    return np.clip(output, 0.0, 1.0)


def _normalize_map(value: Any) -> np.ndarray | None:
    if not isinstance(value, np.ndarray) or value.size == 0:
        return None
    array = value.astype(np.float32)
    if array.ndim == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
    if array.ndim != 2:
        return None
    maximum = float(np.max(array))
    if maximum > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0)


def _extract_roi_pair(
    reference: np.ndarray | None,
    test: np.ndarray | None,
    focus_box: tuple[int, int, int, int] | list[int] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if (
        reference is None
        or test is None
        or not isinstance(reference, np.ndarray)
        or not isinstance(test, np.ndarray)
        or reference.size == 0
        or test.size == 0
    ):
        return None, None

    ref = reference.copy()
    tst = test.copy()
    if ref.shape != tst.shape:
        tst = cv2.resize(
            tst,
            (ref.shape[1], ref.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    if focus_box and len(focus_box) >= 4:
        x, y, width, height = (
            int(round(float(value))) for value in focus_box[:4]
        )
        x1, y1 = max(0, x), max(0, y)
        x2 = min(ref.shape[1], x1 + max(1, width))
        y2 = min(ref.shape[0], y1 + max(1, height))
        if x2 > x1 and y2 > y1:
            return (
                ref[y1:y2, x1:x2].copy(),
                tst[y1:y2, x1:x2].copy(),
            )
    return ref, tst


def _pixel_difference_map(
    reference: np.ndarray | None,
    test: np.ndarray | None,
    focus_box: tuple[int, int, int, int] | list[int] | None,
) -> np.ndarray:
    ref, tst = _extract_roi_pair(reference, test, focus_box)
    if ref is None or tst is None or min(ref.shape[:2]) < 2:
        return np.zeros((MAP_SIDE, MAP_SIDE), dtype=np.float32)

    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(tst, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(test_lab - ref_lab, axis=2) / 180.0
    delta = np.clip((delta - 0.025) / 0.60, 0.0, 1.0)
    delta = cv2.GaussianBlur(delta, (3, 3), 0)
    return cv2.resize(
        delta,
        (MAP_SIDE, MAP_SIDE),
        interpolation=cv2.INTER_AREA,
    )


def _combined_anomaly_map(
    detail: dict,
    reference: np.ndarray | None,
    test: np.ndarray | None,
    focus_box: tuple[int, int, int, int] | list[int] | None,
) -> np.ndarray:
    layers: list[tuple[np.ndarray, float]] = []

    for key, weight in (
        ("adhesive_excess_mask", 1.00),
        ("adhesive_padding_overlap_mask", 1.00),
        ("excess_mask", 0.95),
        ("padding_overlap_mask", 1.00),
        ("extra_mask", 0.85),
        ("missing_mask", 0.75),
        ("diff_mask", 0.80),
        ("heat_map_raw", 0.70),
        ("semantic_reconstruction_map", 0.65),
    ):
        normalized = _normalize_map(detail.get(key))
        if normalized is not None:
            layers.append(
                (
                    cv2.resize(
                        normalized,
                        (MAP_SIDE, MAP_SIDE),
                        interpolation=cv2.INTER_AREA,
                    ),
                    weight,
                )
            )

    layers.append((_pixel_difference_map(reference, test, focus_box), 0.60))

    combined = np.zeros((MAP_SIDE, MAP_SIDE), dtype=np.float32)
    for layer, weight in layers:
        combined = np.maximum(
            combined,
            np.clip(layer * weight, 0.0, 1.0),
        )

    peak = float(np.max(combined))
    if peak > 1e-8:
        combined = np.clip(combined / max(peak, 0.35), 0.0, 1.0)
    return combined


def _physics_vector(detail: dict) -> tuple[np.ndarray, dict]:
    roi_width = max(
        1.0,
        _safe_float(
            detail.get("adhesive_roi_width", detail.get("roi_width", 1))
        ),
    )
    roi_height = max(
        1.0,
        _safe_float(
            detail.get("adhesive_roi_height", detail.get("roi_height", 1))
        ),
    )
    dx = _safe_float(detail.get("adhesive_dx", detail.get("dx", 0.0)))
    dy = _safe_float(detail.get("adhesive_dy", detail.get("dy", 0.0)))

    values = np.asarray(
        [
            np.clip(_safe_float(detail.get("adhesive_score", 0.0)), 0.0, 1.0),
            np.clip(
                _safe_float(detail.get("excess_coverage", 0.0)) / 0.10,
                0.0,
                1.0,
            ),
            np.clip(
                _safe_float(detail.get("padding_overlap", 0.0)) / 0.045,
                0.0,
                1.0,
            ),
            np.clip(
                _safe_float(detail.get("area_growth_ratio", 0.0)) / 3.0,
                0.0,
                1.0,
            ),
            np.clip(
                _safe_float(detail.get("spread_growth_ratio", 0.0)) / 2.0,
                0.0,
                1.0,
            ),
            np.clip(
                _safe_float(detail.get("lower_leakage_ratio", 0.0)),
                0.0,
                1.0,
            ),
            np.clip(dx / roi_width * 0.5 + 0.5, 0.0, 1.0),
            np.clip(dy / roi_height * 0.5 + 0.5, 0.0, 1.0),
            np.clip(
                _safe_float(
                    detail.get(
                        "silk_error_pct",
                        detail.get("pct_changed", 0.0),
                    )
                ),
                0.0,
                1.0,
            ),
            np.clip(_safe_float(detail.get("extra_pct", 0.0)), 0.0, 1.0),
            np.clip(_safe_float(detail.get("missing_pct", 0.0)), 0.0, 1.0),
            np.clip(_safe_float(detail.get("local_score", 0.0)), 0.0, 1.0),
            np.clip(_safe_float(detail.get("ctx_score", 0.0)), 0.0, 1.0),
            np.clip(_safe_float(detail.get("semantic_loss", 0.0)), 0.0, 1.0),
            np.clip(
                _safe_float(detail.get("semantic_local_evidence", 0.0)),
                0.0,
                1.0,
            ),
            np.clip(
                _safe_float(detail.get("hist_corr", 1.0)) * -0.5 + 0.5,
                0.0,
                1.0,
            ),
        ],
        dtype=np.float32,
    )
    names = (
        "adhesive_score",
        "excess_coverage",
        "padding_overlap",
        "area_growth",
        "spread_growth",
        "lower_leakage",
        "dx_normalized",
        "dy_normalized",
        "structural_error",
        "structure_extra",
        "structure_missing",
        "texture_local",
        "texture_context",
        "semantic_score",
        "semantic_local",
        "histogram_inversion",
    )
    return values, {
        name: float(value)
        for name, value in zip(names, values)
    }


def build_anomaly_signature(
    reference: np.ndarray | None,
    test: np.ndarray | None,
    detail: dict | None,
    aoi_info: dict | None = None,
    focus_box: tuple[int, int, int, int] | list[int] | None = None,
) -> dict:
    """Cria a memória compacta da anomalia, não da peça completa."""
    payload = detail if isinstance(detail, dict) else {}
    context = aoi_info if isinstance(aoi_info, dict) else {}

    semantic_delta = _fixed_vector(
        payload.get("semantic_delta")
        or (payload.get("semantic_debug") or {}).get("delta_vector", []),
        SEMANTIC_SIZE,
    )
    spatial = _fixed_vector(
        ((payload.get("semantic_debug") or {}).get("spatial") or {}).get(
            "combined_delta_grid",
            [],
        ),
        SPATIAL_SIZE,
    )
    anomaly_map = _combined_anomaly_map(
        payload,
        reference,
        test,
        focus_box,
    )
    physics, physics_named = _physics_vector(payload)

    vector = np.concatenate(
        [
            semantic_delta,
            spatial,
            anomaly_map.reshape(-1),
            physics,
        ]
    ).astype(np.float32)

    weights = anomaly_map.astype(np.float64)
    total = float(weights.sum())
    centroid = None
    if total > 1e-9:
        yy, xx = np.indices(weights.shape)
        centroid = [
            float((xx * weights).sum() / total / max(MAP_SIDE - 1, 1)),
            float((yy * weights).sum() / total / max(MAP_SIDE - 1, 1)),
        ]

    return {
        "schema": SCHEMA_VERSION,
        "vector_size": VECTOR_SIZE,
        "vector": vector.tolist(),
        "ranges": {
            name: [start, end]
            for name, (start, end) in RANGES.items()
        },
        "group_weights": dict(GROUP_WEIGHTS),
        "semantic_delta": semantic_delta.tolist(),
        "spatial_grid_4x4": spatial.reshape(4, 4).tolist(),
        "anomaly_map_8x8": anomaly_map.tolist(),
        "physics": physics_named,
        "summary": {
            "magnitude": float(np.mean(vector)),
            "map_peak": float(np.max(anomaly_map)),
            "map_coverage": float(np.mean(anomaly_map >= 0.25)),
            "centroid_normalized": centroid,
        },
        "context": {
            "board": str(context.get("board", "")),
            "part": str(context.get("parts", "")),
            "category": str(context.get("category", "")),
            "value": str(context.get("value", "")),
        },
    }


def _group_similarity(query: np.ndarray, stored: np.ndarray) -> float:
    if query.size != stored.size or query.size == 0:
        return 0.0
    query_norm = float(np.linalg.norm(query))
    stored_norm = float(np.linalg.norm(stored))
    if query_norm <= 1e-9 and stored_norm <= 1e-9:
        return 1.0
    if query_norm <= 1e-9 or stored_norm <= 1e-9:
        return 0.0
    cosine = float(np.dot(query, stored) / (query_norm * stored_norm))
    cosine = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
    distance = float(np.mean(np.abs(query - stored)))
    magnitude = float(np.clip(1.0 - distance, 0.0, 1.0))
    return float(
        np.clip(cosine * 0.60 + magnitude * 0.40, 0.0, 1.0)
    )


def compare_anomaly_signatures(
    query_signature: dict,
    stored_signature: dict,
) -> tuple[float, dict]:
    """Compara forma, posição e física da anomalia por grupos."""
    query = _fixed_vector(query_signature.get("vector", []), VECTOR_SIZE)
    stored = _fixed_vector(stored_signature.get("vector", []), VECTOR_SIZE)

    scores = {}
    total = 0.0
    for name, (start, end) in RANGES.items():
        score = _group_similarity(query[start:end], stored[start:end])
        scores[name] = score
        total += score * GROUP_WEIGHTS[name]

    return float(np.clip(total, 0.0, 1.0)), {
        "schema": SCHEMA_VERSION,
        "groups": scores,
        "weights": dict(GROUP_WEIGHTS),
    }


def valid_anomaly_signature(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        array = np.asarray(value.get("vector", []), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    return array.size == VECTOR_SIZE and bool(np.all(np.isfinite(array)))
