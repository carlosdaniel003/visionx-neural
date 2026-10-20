"""Hipótese geométrica OFFLINE: contornos de perda/ganho no par AOI.

Em cada escala compara bordas do gabarito que desapareceram e bordas que
apareceram na captura. É um segundo ranking experimental; o ranking visual
v3 é mantido intacto no relatório A/B, sem decisão física ou treino.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.services.startup_regression.knn_archive_focus_signature import (
    _green_mask, _focus_boxes, _similarity, SIDE, compare_focus_signatures,
)

SCHEMA = "visionx.knn_archive_geometry.v1"
GEOMETRY_WEIGHT = .35
SCALES = (("global", .15), ("outer", .35), ("inner", .50))


def _edge_mismatch(reference: np.ndarray, test: np.ndarray,
                   excluded: np.ndarray, rect=None) -> dict:
    height, width = reference.shape[:2]
    if rect is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x, y, w, h = rect
        x1, y1, x2, y2 = max(0, x + 3), max(0, y + 3), min(width, x + w - 3), min(height, y + h - 3)
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            raise ValueError("Epicentro sem área útil para contornos")
    ref = cv2.cvtColor(reference[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    obs = cv2.cvtColor(test[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    ref_edge = cv2.Canny(cv2.GaussianBlur(ref, (3, 3), 0), 45, 120)
    obs_edge = cv2.Canny(cv2.GaussianBlur(obs, (3, 3), 0), 45, 120)
    invalid = excluded[y1:y2, x1:x2] > 0
    ref_edge[invalid] = 0
    obs_edge[invalid] = 0
    kernel = np.ones((3, 3), np.uint8)
    # Um deslocamento de um pixel não vira defeito por desregistro trivial.
    missing = cv2.bitwise_and(ref_edge, cv2.bitwise_not(cv2.dilate(obs_edge, kernel)))
    added = cv2.bitwise_and(obs_edge, cv2.bitwise_not(cv2.dilate(ref_edge, kernel)))
    size = float(max(1, ref.shape[0] * ref.shape[1]))
    def stats(edge):
        yy, xx = np.nonzero(edge)
        if not xx.size:
            return [0.0, 0.5, 0.5]
        return [
            float(xx.size / size),
            float((xx.mean() + .5) / edge.shape[1]),
            float((yy.mean() + .5) / edge.shape[0]),
        ]
    return {
        "missing_map": (cv2.resize(missing, (SIDE, SIDE),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255).tolist(),
        "added_map": (cv2.resize(added, (SIDE, SIDE),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255).tolist(),
        "missing_stats": stats(missing),
        "added_stats": stats(added),
    }


def build_geometry_signature(reference: np.ndarray, test: np.ndarray,
                             focus: dict) -> dict:
    if not (isinstance(reference, np.ndarray) and isinstance(test, np.ndarray)
            and reference.ndim == test.ndim == 3
            and reference.shape[2] == test.shape[2] == 3
            and min(reference.shape[:2]) >= 12 and min(test.shape[:2]) >= 12):
        raise ValueError("Par AOI inválido para geometria")
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]),
                          interpolation=cv2.INTER_AREA)
    if not isinstance(focus, dict):
        raise ValueError("É preciso usar epicentros detectados no descritor v3")
    # NUNCA estimar uma marca ausente a partir da classe ou da data.
    box = focus.get("outer_box")
    inner = focus.get("inner_box")
    ignored = cv2.dilate(cv2.bitwise_or(
        _green_mask(reference), _green_mask(test)
    ), np.ones((5, 5), np.uint8))
    scales = {
        "global": _edge_mismatch(reference, test, ignored),
        "outer": _edge_mismatch(reference, test, ignored, box) if box else None,
        "inner": _edge_mismatch(reference, test, ignored, inner) if inner else None,
    }
    return {"schema": SCHEMA, "scales": scales}


def _score_region(a: dict, b: dict) -> float:
    missing = _similarity(a["missing_map"], b["missing_map"])
    added = _similarity(a["added_map"], b["added_map"])
    # Centróides e quantidades de bordas novas/ausentes trazem informações
    # físicas de deslocamento mesmo quando o mapa de 16x16 é parecido.
    af = np.asarray(a["missing_stats"] + a["added_stats"], dtype=np.float32)
    bf = np.asarray(b["missing_stats"] + b["added_stats"], dtype=np.float32)
    feature = float(np.clip(1.0 - np.mean(np.abs(af - bf)), 0, 1))
    return float(np.clip(.40 * missing + .40 * added + .20 * feature, 0, 1))


def compare_geometry_signatures(query: dict, stored: dict) -> tuple[float, dict]:
    if query.get("schema") != SCHEMA or stored.get("schema") != SCHEMA:
        raise ValueError("Assinaturas geométricas incompatíveis")
    total, weighted, breakdown = 0., 0., {}
    for name, weight in SCALES:
        left = query["scales"].get(name)
        right = stored["scales"].get(name)
        if left is None and right is None:
            continue
        score = 0.0 if left is None or right is None else _score_region(left, right)
        breakdown[name] = round(score, 8)
        weighted += weight * score
        total += weight
    return float(np.clip(weighted / total if total else 0.0, 0, 1)), breakdown


def compare_geometry_hybrid(focus_query: dict, focus_stored: dict,
                            geo_query: dict, geo_stored: dict) -> tuple[float, dict]:
    color_score, _ = compare_focus_signatures(focus_query, focus_stored)
    geometry_score, parts = compare_geometry_signatures(geo_query, geo_stored)
    return float(np.clip(
        color_score * (1 - GEOMETRY_WEIGHT) + geometry_score * GEOMETRY_WEIGHT,
        0, 1,
    )), {
        "color_score": round(color_score, 8),
        "geometry_score": round(geometry_score, 8),
        "geometry_scales": parts,
        "geometry_weight": GEOMETRY_WEIGHT,
    }
