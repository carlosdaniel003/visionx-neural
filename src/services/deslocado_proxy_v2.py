"""Proxies DESLOCADO v2: componente segmentado, reconstrução de fundo simétrica.

Uso somente EXPERIMENTAL. Sem máscara real do componente não é possível
comprovar que a região movida seja o componente. Falha de segmentação =>
proxy rejeitado, nunca rotular a alteração como NG real.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PROXY_VERSION = "component_mask_inpaint_symmetric_v2"


@dataclass(frozen=True)
class ProxyResult:
    normal: np.ndarray
    shifted: np.ndarray
    component_box: tuple[int, int, int, int]
    shift_xy: tuple[int, int]
    mask_area: int
    confidence_hint: str = "HEURISTIC_UNVERIFIED"


def component_mask(test: np.ndarray) -> tuple[np.ndarray, tuple[int,int,int,int]] | None:
    """Hipótese de objeto aproximadamente central, baseada no contraste local.

    Seleção geométrica/visual NÃO equivale a segmentação de componente
    validada em placas reais. Devolve None se a identificação for ambígua.
    """
    if (not isinstance(test, np.ndarray) or test.dtype != np.uint8
            or test.ndim != 3 or test.shape[2] != 3
            or min(test.shape[:2]) < 48):
        return None
    h, w = test.shape[:2]
    margin_x = max(3, round(w * .10))
    margin_y = max(3, round(h * .10))
    border = np.concatenate((
        test[:margin_y].reshape(-1, 3),
        test[h-margin_y:].reshape(-1, 3),
        test[margin_y:h-margin_y, :margin_x].reshape(-1, 3),
        test[margin_y:h-margin_y, w-margin_x:].reshape(-1, 3),
    )).astype(np.float32)
    background = np.median(border, axis=0)
    difference = np.linalg.norm(
        test.astype(np.float32) - background, axis=2
    )
    # Contraste robusto à variação de iluminação. Não aceitar quadro uniforme.
    noise = np.median(np.linalg.norm(border-background, axis=1))
    threshold = max(35., float(noise * 4.0))
    binary = (difference > threshold).astype(np.uint8)
    # Limitar segmento a região de inspeção; evitar OCR e bordas AOI.
    binary[:margin_y] = 0
    binary[h-margin_y:] = 0
    binary[:, :margin_x] = 0
    binary[:, w-margin_x:] = 0
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    n, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for idx in range(1, n):
        x, y, bw, bh, area = map(int, stats[idx])
        fraction = area / (h*w)
        cx, cy = centers[idx]
        offset = np.hypot((cx-w/2)/w, (cy-h/2)/h)
        if (
            .008 <= fraction <= .38 and bw >= 9 and bh >= 8
            and offset <= .27 and bw/w <= .78 and bh/h <= .76
        ):
            score = float(fraction / (1 + 6*offset))
            candidates.append((score, idx, (x,y,bw,bh), area))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    # Dois objetos com força parecida => não sabemos qual é o componente.
    if len(candidates) > 1 and candidates[1][0] >= .80*candidates[0][0]:
        return None
    _, idx, box, area = candidates[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[labels == idx] = 255
    return mask, box


def simulate_component_shift(
    test: np.ndarray, seed: int, *, shift_fraction: float | None = None,
) -> ProxyResult | None:
    """Produz OK reconstruído e NG sintético com MESMOS artefatos de inpaint.

    Mover somente máscara de objeto, não o patch inteiro. O fundo do local
    original é restaurado em AMBAS as classes. A mesma operação de
    composição (com deslocamento zero ou diferente) evita que o motor
    aprenda somente a presença de uma borda artificial no NG.
    """
    found = component_mask(test)
    if found is None:
        return None
    mask, (x,y,bw,bh) = found
    h, w = mask.shape
    rng = np.random.default_rng(seed)
    magnitude = (float(shift_fraction) if shift_fraction is not None
                 else float(rng.uniform(.10, .22)))
    if not .05 <= magnitude <= .35:
        raise ValueError("Deslocamento sintético inválido")
    angle = float(rng.uniform(0, 2*np.pi))
    dx = int(round(bw*magnitude*np.cos(angle)))
    dy = int(round(bh*magnitude*np.sin(angle)))
    # Evitar proxy idêntico ao original / deslocamento subpixel.
    if abs(dx)+abs(dy) < 2:
        dx = 2 if np.cos(angle) >= 0 else -2
    if (
        x+dx < 2 or y+dy < 2 or x+bw+dx >= w-2
        or y+bh+dy >= h-2
    ):
        return None

    # Dilatação pequena inclui bordas e sombras da peça.
    expanded = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    background = cv2.inpaint(test, expanded, 5, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask, (3, 3), .7).astype(np.float32)/255.
    alpha = np.clip(alpha, 0, 1)

    def compose(delta_x: int, delta_y: int) -> np.ndarray:
        transformation = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
        moved_color = cv2.warpAffine(
            test, transformation, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        moved_alpha = cv2.warpAffine(
            alpha, transformation, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[..., None]
        return np.clip(
            background.astype(np.float32)*(1-moved_alpha)
            + moved_color.astype(np.float32)*moved_alpha,
            0, 255,
        ).astype(np.uint8)

    normal = compose(0, 0)
    shifted = compose(dx, dy)
    if np.mean(np.abs(
        shifted.astype(np.float32)-normal.astype(np.float32)
    )) < .18:
        return None
    return ProxyResult(
        normal=normal, shifted=shifted,
        component_box=(x,y,bw,bh), shift_xy=(dx,dy),
        mask_area=int(np.count_nonzero(mask)),
    )


def paired_photometric(
    reference: np.ndarray, test: np.ndarray, seed: int,
    *, independent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Augmentação leve de iluminação tanto para OK quanto para proxy."""
    rng = np.random.default_rng(seed)
    gain = float(rng.uniform(.89, 1.11))
    bias = float(rng.uniform(-9., 9.))
    ref = np.clip(reference.astype(np.float32)*gain+bias, 0, 255)
    if independent:
        gain *= float(rng.uniform(.98, 1.02))
        bias += float(rng.uniform(-2., 2.))
    tst = np.clip(test.astype(np.float32)*gain+bias, 0, 255)
    return ref.astype(np.uint8), tst.astype(np.uint8)


__all__ = [
    "PROXY_VERSION", "ProxyResult", "component_mask",
    "simulate_component_shift", "paired_photometric",
]
