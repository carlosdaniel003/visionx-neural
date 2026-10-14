"""Preprocessamento compartilhado da CNN FALTANDO v2.

A rotina online usa exatamente os mesmos pixels que os datasets de treino,
sem precisar importar o runner de treinamento, geração de manifesto ou
treinamento incremental para preparar um único par AOI.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def _letterbox_rgb(image: np.ndarray, size: int) -> torch.Tensor:
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    target = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    interpolation = cv2.INTER_AREA if ratio <= 1 else cv2.INTER_LINEAR
    frame = cv2.resize(image, target, interpolation=interpolation)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    y = (size - target[1]) // 2
    x = (size - target[0]) // 2
    canvas[y:y+target[1], x:x+target[0]] = frame
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.


def _focus_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Zoom central: hipótese geométrica explícita, sem usar rótulo/ROI manual."""
    height, width = image.shape[:2]
    crop_h, crop_w = max(12, round(height * fraction)), max(12, round(width * fraction))
    cy, cx = height // 2, width // 2
    top = max(0, min(height - crop_h, cy - crop_h // 2))
    left = max(0, min(width - crop_w, cx - crop_w // 2))
    return image[top:top+crop_h, left:left+crop_w]


__all__ = ["_letterbox_rgb", "_focus_crop"]
