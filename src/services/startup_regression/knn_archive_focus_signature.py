"""Assinatura KNN OFFLINE do par AOI: gabarito vs. captura e duas ROIs.

Reconhece marcas verdes existentes no recorte real; nunca inventa um epicentro
menor a partir do próprio rótulo nem utiliza data ou julgamento do arquivo.
Não é utilizado pelo roteador de inspeção/produção.
"""
from __future__ import annotations

import cv2
import numpy as np

SCHEMA = "visionx.knn_archive_aoi_focus.v1"
SIDE = 16
GREEN_LOW = np.array((40, 100, 50), dtype=np.uint8)
GREEN_HIGH = np.array((80, 255, 255), dtype=np.uint8)


def _green_mask(image: np.ndarray) -> np.ndarray:
    return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), GREEN_LOW, GREEN_HIGH)


def _boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Contornos de molduras AOI, desduplicados sem confundir espessura."""
    mask = cv2.morphologyEx(
        _green_mask(image), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12 or (w * h) < 180:
            continue
        # Uma linha verde alongada ou ruído não é um retângulo AOI.
        if cv2.contourArea(contour) < w * h * .65:
            continue
        if w >= width * .98 and h >= height * .98:
            continue
        candidates.append((x, y, w, h))
    candidates.sort(key=lambda b: (-(b[2] * b[3]), b[0], b[1]))
    unique = []
    for box in candidates:
        if not any(
            abs(box[0] - old[0]) <= 7 and abs(box[1] - old[1]) <= 7
            and abs(box[2] - old[2]) <= 7 and abs(box[3] - old[3]) <= 7
            for old in unique
        ):
            unique.append(box)
    return unique


def _contains(outer, inner) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (
        iw * ih < ow * oh * .80 and
        ix >= ox - 3 and iy >= oy - 3 and
        ix + iw <= ox + ow + 3 and iy + ih <= oy + oh + 3
    )


def _focus_boxes(image: np.ndarray) -> tuple[tuple | None, tuple | None]:
    boxes = _boxes(image)
    if not boxes:
        return None, None
    # Quando há três molduras, seleciona o pai IMEDIATO do menor epicentro,
    # em vez da borda da tela como ROI de componente.
    pairs = [(outer, inner) for outer in boxes for inner in boxes
             if outer != inner and _contains(outer, inner)]
    if pairs:
        outer, inner = min(pairs, key=lambda pair: pair[0][2] * pair[0][3])
        return outer, inner
    # Uma única moldura válida permite a ROI maior, jamais fabrica a menor.
    return boxes[0], None


def _map(ref: np.ndarray, test: np.ndarray, mask: np.ndarray, box=None) -> list:
    h, w = ref.shape[:2]
    if box is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x, y, bw, bh = box
        x1, y1 = max(0, x + 3), max(0, y + 3)
        x2, y2 = min(w, x + bw - 3), min(h, y + bh - 3)
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return [[0.0] * SIDE for _ in range(SIDE)]
    ref_roi = ref[y1:y2, x1:x2]
    test_roi = test[y1:y2, x1:x2]
    excluded = mask[y1:y2, x1:x2] > 0
    a = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(test_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(a - b, axis=2) / 180.0
    delta = np.clip((delta - .025) / .60, 0, 1)
    delta[excluded] = 0.0  # Marcas verdes NÃO representam anomalia física.
    delta = cv2.GaussianBlur(delta, (3, 3), 0)
    return cv2.resize(delta, (SIDE, SIDE), interpolation=cv2.INTER_AREA).tolist()


def build_focus_signature(reference: np.ndarray, test: np.ndarray) -> dict:
    if not (isinstance(reference, np.ndarray) and isinstance(test, np.ndarray)
            and reference.ndim == test.ndim == 3
            and reference.shape[2] == test.shape[2] == 3
            and min(reference.shape[:2]) >= 12 and min(test.shape[:2]) >= 12):
        raise ValueError("Par AOI gabarito/teste inválido")
    if reference.shape != test.shape:
        test = cv2.resize(test, (reference.shape[1], reference.shape[0]),
                          interpolation=cv2.INTER_AREA)
    outer, inner = _focus_boxes(test)
    # A moldura pode estar somente no gabarito, dependendo do software AOI.
    if outer is None:
        outer, inner = _focus_boxes(reference)
    mark = cv2.bitwise_or(_green_mask(reference), _green_mask(test))
    mark = cv2.dilate(mark, np.ones((3, 3), np.uint8))
    return {
        "schema": SCHEMA,
        "global_map": _map(reference, test, mark),
        "outer_map": _map(reference, test, mark, outer) if outer else None,
        "inner_map": _map(reference, test, mark, inner) if inner else None,
        "focus": {
            "outer_detected": outer is not None,
            "inner_detected": inner is not None,
            "outer_box": list(outer) if outer else None,
            "inner_box": list(inner) if inner else None,
        },
    }


def _similarity(a, b) -> float:
    left, right = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    norm_l, norm_r = np.linalg.norm(left), np.linalg.norm(right)
    if norm_l < 1e-9 and norm_r < 1e-9:
        return 1.0
    if min(norm_l, norm_r) < 1e-9:
        return 0.0
    cosine = float(np.dot(left, right) / (norm_l * norm_r))
    mae = float(np.mean(np.abs(left - right)))
    return float(np.clip(.70 * cosine + .30 * (1.0 - mae), 0, 1))


def compare_focus_signatures(query: dict, stored: dict) -> tuple[float, dict]:
    if query.get("schema") != SCHEMA or stored.get("schema") != SCHEMA:
        raise ValueError("Assinaturas de epicentro incompatíveis")
    weights = {"global_map": .15, "outer_map": .35, "inner_map": .50}
    components, total, summed = {}, 0.0, 0.0
    for part, weight in weights.items():
        left, right = query.get(part), stored.get(part)
        if left is None and right is None:
            continue
        if left is None or right is None:
            # Ausência de ROI na consulta ou na memória: não declarar
            # equivalência quase perfeita apenas pelo fundo.
            score = .0
        else:
            score = _similarity(left, right)
        components[part] = round(float(score), 8)
        summed += weight * score
        total += weight
    similarity = summed / total if total else 0.0
    return float(np.clip(similarity, 0, 1)), {
        "groups": components,
        "method": "gabarito_vs_teste_global_outer_inner",
        "query_focus": query.get("focus"),
        "neighbor_focus": stored.get("focus"),
    }


def valid_focus_signature(value: dict) -> bool:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return False
    for name in ("global_map", "outer_map", "inner_map"):
        if name != "global_map" and value.get(name) is None:
            continue
        try:
            arr = np.asarray(value.get(name), dtype=np.float32)
        except (ValueError, TypeError):
            return False
        if arr.shape != (SIDE, SIDE) or not np.all(np.isfinite(arr)):
            return False
    return True
