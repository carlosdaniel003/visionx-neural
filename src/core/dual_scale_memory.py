"""Memória visual em duas escalas: epicentro local + contexto do componente.

A assinatura local de 224 dimensões continua intacta. Esta extensão acrescenta
uma assinatura contextual construída a partir da maior caixa verde da AOI e
combina as duas similaridades sem quebrar JSONs antigos.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.config.settings import settings


DUAL_SCALE_SCHEMA = "visionx.dual_scale.v1"
CONTEXT_SCHEMA = "visionx.context.v1"
EPICENTER_WEIGHT = 0.70
CONTEXT_WEIGHT = 0.30
CONTEXT_EMBEDDING_SIZE = 128
CONTEXT_MAP_SIDE = 8
CONTEXT_MIN_SIDE = 12

CONTEXT_FEATURE_WEIGHTS = {
    "reference_appearance": 0.25,
    "test_appearance": 0.25,
    "semantic_delta": 0.15,
    "spatial_delta": 0.10,
    "difference_map": 0.15,
    "geometry": 0.10,
}


def _safe_pair(reference: np.ndarray | None, test: np.ndarray | None):
    if (
        not isinstance(reference, np.ndarray)
        or not isinstance(test, np.ndarray)
        or reference.size == 0
        or test.size == 0
    ):
        return None, None
    ref = reference.copy()
    tst = test.copy()
    if ref.shape != tst.shape:
        tst = cv2.resize(tst, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return ref, tst


def find_component_context_box(image: np.ndarray | None):
    """Retorna a maior caixa verde, que representa o contexto do componente."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, settings.COLOR_GREEN_LOWER, settings.COLOR_GREEN_UPPER)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return None

    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [
        tuple(int(value) for value in box)
        for box in boxes
        if box[2] > CONTEXT_MIN_SIDE and box[3] > CONTEXT_MIN_SIDE
    ]
    if not boxes:
        return None

    unique = []
    for candidate in boxes:
        x, y, width, height = candidate
        duplicate = any(
            abs(x - ux) < 10
            and abs(y - uy) < 10
            and abs(width - uw) < 10
            and abs(height - uh) < 10
            for ux, uy, uw, uh in unique
        )
        if not duplicate:
            unique.append(candidate)

    if not unique:
        return None
    unique.sort(key=lambda box: box[2] * box[3], reverse=True)
    return unique[0]


def _crop_box(image: np.ndarray, box):
    x, y, width, height = (int(value) for value in box)
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + max(1, width))
    y2 = min(h, y + max(1, height))
    if x2 - x1 < CONTEXT_MIN_SIDE or y2 - y1 < CONTEXT_MIN_SIDE:
        return None

    margin = max(1, min(4, int(round(min(x2 - x1, y2 - y1) * 0.015))))
    if x2 - x1 > margin * 2 + CONTEXT_MIN_SIDE:
        x1 += margin
        x2 -= margin
    if y2 - y1 > margin * 2 + CONTEXT_MIN_SIDE:
        y1 += margin
        y2 -= margin
    return image[y1:y2, x1:x2].copy()


def _appearance_embedding(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_grid = []
    brightness_grid = []
    for y in range(0, 64, 16):
        for x in range(0, 64, 16):
            edge_grid.append(float(np.mean(edges[y:y + 16, x:x + 16]) / 255.0))
            brightness_grid.append(float(np.mean(gray[y:y + 16, x:x + 16]) / 255.0))

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histograms = []
    for channel, limit in ((0, 180), (1, 256), (2, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [32], [0, limit])
        cv2.normalize(histogram, histogram)
        histograms.append(histogram.flatten())

    embedding = np.concatenate(
        [np.asarray(edge_grid, dtype=np.float32), np.asarray(brightness_grid, dtype=np.float32), *histograms]
    ).astype(np.float32)
    return np.clip(embedding, 0.0, 1.0)


def _relative_delta(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    absolute = np.abs(test - reference)
    scale = np.abs(test) + np.abs(reference) + 0.04
    return np.clip(absolute / scale, 0.0, 1.0).astype(np.float32)


def _spatial_delta(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    edge_delta = _relative_delta(reference[:16], test[:16]).reshape(4, 4)
    brightness_delta = _relative_delta(reference[16:32], test[16:32]).reshape(4, 4)
    return np.clip(edge_delta * 0.62 + brightness_delta * 0.38, 0.0, 1.0).astype(np.float32)


def _difference_map(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    test_resized = test
    if reference.shape != test.shape:
        test_resized = cv2.resize(test, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab = cv2.cvtColor(test_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(test_lab - ref_lab, axis=2) / 180.0
    delta = np.clip((delta - 0.025) / 0.60, 0.0, 1.0)
    delta = cv2.GaussianBlur(delta.astype(np.float32), (3, 3), 0)
    return cv2.resize(delta, (CONTEXT_MAP_SIDE, CONTEXT_MAP_SIDE), interpolation=cv2.INTER_AREA).astype(np.float32)


def build_component_context_signature(reference, test, context_box=None) -> dict:
    """Descreve a aparência e a mudança dentro da caixa maior da AOI."""
    ref, tst = _safe_pair(reference, test)
    if ref is None or tst is None:
        return {}

    box = None
    if context_box and len(context_box) >= 4:
        try:
            box = tuple(int(round(float(value))) for value in context_box[:4])
        except (TypeError, ValueError):
            box = None
    if box is None:
        box = find_component_context_box(tst)
    if box is None:
        return {}

    ref_context = _crop_box(ref, box)
    test_context = _crop_box(tst, box)
    if ref_context is None or test_context is None:
        return {}
    if ref_context.shape != test_context.shape:
        test_context = cv2.resize(test_context, (ref_context.shape[1], ref_context.shape[0]), interpolation=cv2.INTER_AREA)

    reference_embedding = _appearance_embedding(ref_context)
    test_embedding = _appearance_embedding(test_context)
    semantic_delta = _relative_delta(reference_embedding, test_embedding)
    spatial_delta = _spatial_delta(reference_embedding, test_embedding)
    difference_map = _difference_map(ref_context, test_context)

    x, y, width, height = box
    full_h, full_w = ref.shape[:2]
    geometry = {
        "aspect_ratio": float(width / max(height, 1)),
        "width_normalized": float(width / max(full_w, 1)),
        "height_normalized": float(height / max(full_h, 1)),
        "area_normalized": float((width * height) / max(full_w * full_h, 1)),
    }

    return {
        "schema": CONTEXT_SCHEMA,
        "valid": True,
        "context_box": [int(x), int(y), int(width), int(height)],
        "context_size": [int(ref_context.shape[1]), int(ref_context.shape[0])],
        "reference_embedding_128": reference_embedding.tolist(),
        "test_embedding_128": test_embedding.tolist(),
        "semantic_delta_128": semantic_delta.tolist(),
        "spatial_delta_4x4": spatial_delta.tolist(),
        "difference_map_8x8": difference_map.tolist(),
        "geometry": geometry,
        "summary": {
            "delta_mean": float(np.mean(semantic_delta)),
            "delta_peak": float(np.max(semantic_delta)),
            "map_mean": float(np.mean(difference_map)),
            "map_peak": float(np.max(difference_map)),
        },
    }


def valid_context_signature(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema") != CONTEXT_SCHEMA:
        return False
    try:
        reference = np.asarray(value.get("reference_embedding_128", []), dtype=np.float32).reshape(-1)
        test = np.asarray(value.get("test_embedding_128", []), dtype=np.float32).reshape(-1)
        delta = np.asarray(value.get("semantic_delta_128", []), dtype=np.float32).reshape(-1)
        spatial = np.asarray(value.get("spatial_delta_4x4", []), dtype=np.float32).reshape(-1)
        difference = np.asarray(value.get("difference_map_8x8", []), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    arrays = (reference, test, delta, spatial, difference)
    sizes = (128, 128, 128, 16, 64)
    return bool(
        all(array.size == size for array, size in zip(arrays, sizes))
        and all(np.all(np.isfinite(array)) for array in arrays)
    )


def _vector_similarity(first: Any, second: Any) -> float:
    try:
        left = np.asarray(first, dtype=np.float32).reshape(-1)
        right = np.asarray(second, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if left.size == 0 or left.size != right.size:
        return 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-9 and right_norm <= 1e-9:
        return 1.0
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0

    # Os descritores contextuais são não-negativos. Usar o cosseno diretamente
    # evita inflar artificialmente a semelhança entre componentes diferentes.
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    cosine = float(np.clip(cosine, 0.0, 1.0))
    magnitude = float(np.clip(1.0 - np.mean(np.abs(left - right)), 0.0, 1.0))
    return float(np.clip(cosine * 0.60 + magnitude * 0.40, 0.0, 1.0))


def _geometry_similarity(query: dict, stored: dict) -> float:
    keys = ("aspect_ratio", "width_normalized", "height_normalized", "area_normalized")
    differences = []
    for key in keys:
        try:
            left = float((query.get("geometry") or {}).get(key, 0.0))
            right = float((stored.get("geometry") or {}).get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        scale = max(abs(left), abs(right), 0.05)
        differences.append(min(1.0, abs(left - right) / scale))
    return float(np.clip(1.0 - np.mean(differences), 0.0, 1.0))


def compare_component_context_signatures(query: dict, stored: dict):
    if not valid_context_signature(query) or not valid_context_signature(stored):
        return 0.0, {"schema": CONTEXT_SCHEMA, "valid": False, "reason": "context_signature_missing_or_invalid"}

    scores = {
        "reference_appearance": _vector_similarity(query.get("reference_embedding_128", []), stored.get("reference_embedding_128", [])),
        "test_appearance": _vector_similarity(query.get("test_embedding_128", []), stored.get("test_embedding_128", [])),
        "semantic_delta": _vector_similarity(query.get("semantic_delta_128", []), stored.get("semantic_delta_128", [])),
        "spatial_delta": _vector_similarity(query.get("spatial_delta_4x4", []), stored.get("spatial_delta_4x4", [])),
        "difference_map": _vector_similarity(query.get("difference_map_8x8", []), stored.get("difference_map_8x8", [])),
        "geometry": _geometry_similarity(query, stored),
    }
    similarity = float(sum(scores[name] * CONTEXT_FEATURE_WEIGHTS[name] for name in CONTEXT_FEATURE_WEIGHTS))
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return similarity, {
        "schema": CONTEXT_SCHEMA,
        "valid": True,
        "similarity": similarity,
        "features": scores,
        "weights": dict(CONTEXT_FEATURE_WEIGHTS),
        "query_box": list(query.get("context_box", [])),
        "stored_box": list(stored.get("context_box", [])),
    }


def attach_component_context(local_signature: dict, reference, test) -> dict:
    """Anexa a segunda escala sem alterar o vetor local já existente."""
    if not isinstance(local_signature, dict):
        return local_signature
    output = dict(local_signature)
    context = build_component_context_signature(reference, test)
    if valid_context_signature(context):
        output["dual_scale_schema"] = DUAL_SCALE_SCHEMA
        output["memory_scales"] = ["epicenter", "component_context"]
        output["scale_weights"] = {"epicenter": EPICENTER_WEIGHT, "component_context": CONTEXT_WEIGHT}
        output["context_signature"] = context
    else:
        output["memory_scales"] = ["epicenter"]
    return output


def install_dual_scale_memory(anomaly_memory_module, best_match_module, dataset_manager_module) -> None:
    """Expande construção e comparação mantendo contratos antigos intactos."""
    if getattr(anomaly_memory_module, "_dual_scale_memory_installed", False):
        return

    original_build = anomaly_memory_module.build_anomaly_signature
    original_compare = best_match_module.compare_anomaly_signatures

    def build_dual_scale_signature(reference, test, detail, aoi_info=None, focus_box=None):
        local = original_build(reference, test, detail, aoi_info, focus_box)
        return attach_component_context(local, reference, test)

    def compare_dual_scale_signatures(query_signature, stored_signature):
        epicenter_similarity, epicenter_breakdown = original_compare(query_signature, stored_signature)
        query_context = query_signature.get("context_signature", {}) if isinstance(query_signature, dict) else {}
        stored_context = stored_signature.get("context_signature", {}) if isinstance(stored_signature, dict) else {}

        if valid_context_signature(query_context) and valid_context_signature(stored_context):
            context_similarity, context_breakdown = compare_component_context_signatures(query_context, stored_context)
            combined = float(np.clip(epicenter_similarity * EPICENTER_WEIGHT + context_similarity * CONTEXT_WEIGHT, 0.0, 1.0))
            return combined, {
                "schema": DUAL_SCALE_SCHEMA,
                "policy": "epicenter_plus_component_context",
                "dual_scale": True,
                "similarity": combined,
                "epicenter_similarity": float(epicenter_similarity),
                "context_similarity": float(context_similarity),
                "scale_weights": {"epicenter": EPICENTER_WEIGHT, "component_context": CONTEXT_WEIGHT},
                "epicenter": epicenter_breakdown,
                "component_context": context_breakdown,
            }

        # JSON antigo: preserva exatamente a similaridade local anterior.
        return float(epicenter_similarity), {
            "schema": DUAL_SCALE_SCHEMA,
            "policy": "legacy_epicenter_only",
            "dual_scale": False,
            "similarity": float(epicenter_similarity),
            "epicenter_similarity": float(epicenter_similarity),
            "context_similarity": None,
            "scale_weights": {"epicenter": 1.0, "component_context": 0.0},
            "epicenter": epicenter_breakdown,
        }

    anomaly_memory_module.build_anomaly_signature = build_dual_scale_signature
    best_match_module.compare_anomaly_signatures = compare_dual_scale_signatures
    if hasattr(dataset_manager_module, "build_anomaly_signature"):
        dataset_manager_module.build_anomaly_signature = build_dual_scale_signature

    anomaly_memory_module._dual_scale_memory_installed = True


__all__ = [
    "CONTEXT_SCHEMA",
    "DUAL_SCALE_SCHEMA",
    "CONTEXT_WEIGHT",
    "EPICENTER_WEIGHT",
    "attach_component_context",
    "build_component_context_signature",
    "compare_component_context_signatures",
    "find_component_context_box",
    "install_dual_scale_memory",
    "valid_context_signature",
]
