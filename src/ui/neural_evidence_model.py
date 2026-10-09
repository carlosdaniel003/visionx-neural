"""Diagnostic pixel views of AOI epicenters (NOT Grad-CAM or CNN attention).

This module does not access the classifier, train, classify, or influence 0/1.
Grayscale, differences and block means are transforms of the *observed* pixels.
The FALTANDO v2 model actually receives RGB letterboxed full-frame and central
focus crops; AOI rectangles are separate diagnostic regions, not its weights.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_DIMENSION = 720
VIEWS = ("Cinza", "Diferenças", "Blocos")
EPICENTERS = ("major", "minor")


def _valid(image) -> bool:
    return isinstance(image, np.ndarray) and image.dtype == np.uint8 and (
        image.ndim == 3 and image.shape[-1] == 3
        and image.size > 0 and min(image.shape[:2]) >= 2
    )


def _limit(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = min(1.0, MAX_DIMENSION / max(h, w))
    if ratio == 1.0:
        return image.copy()
    return cv2.resize(image, (max(2, round(w*ratio)), max(2, round(h*ratio))),
                      interpolation=cv2.INTER_AREA)


def evidence_views(reference, observed) -> dict | None:
    """Three fixed, reproducible visualizations and descriptive measurements."""
    if not _valid(reference) or not _valid(observed):
        return None
    ref = _limit(reference)
    test = _limit(observed)
    if ref.shape[:2] != test.shape[:2]:
        ref = cv2.resize(ref, (test.shape[1], test.shape[0]),
                         interpolation=cv2.INTER_AREA)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_ref, gray_test)
    gray_bgr = cv2.cvtColor(gray_test, cv2.COLOR_GRAY2BGR)
    # Fixed absolute 0-255 scale: low difference stays low, rather than
    # exaggerating a tiny noisy difference to full heat-map intensity.
    palette = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    # At zero difference the original remains unchanged (NO false hot spot).
    alpha = (diff.astype(np.float32) / 255.0 * 0.85)[..., None]
    heat = np.clip(
        test.astype(np.float32)*(1.0-alpha)
        + palette.astype(np.float32)*alpha, 0, 255,
    ).astype(np.uint8)
    cell = max(6, min(20, int(round(min(test.shape[:2]) / 9))))
    small_w = max(1, (test.shape[1]+cell-1)//cell)
    small_h = max(1, (test.shape[0]+cell-1)//cell)
    blocks = cv2.resize(
        cv2.resize(gray_test, (small_w, small_h), interpolation=cv2.INTER_AREA),
        (test.shape[1], test.shape[0]), interpolation=cv2.INTER_NEAREST,
    )
    blocks = cv2.cvtColor(blocks, cv2.COLOR_GRAY2BGR)
    sobel_x = cv2.Sobel(gray_test, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_test, cv2.CV_32F, 0, 1, ksize=3)
    contrast = float(np.std(gray_test))
    mean_diff = float(np.mean(diff))
    edge = float(np.mean(cv2.magnitude(sobel_x, sobel_y)))
    return {
        "images": (gray_bgr, heat, blocks),
        "difference_mean": round(mean_diff, 2),
        "contrast_std": round(contrast, 2),
        "edge_mean": round(edge, 2),
        "dimensions": (int(observed.shape[1]), int(observed.shape[0])),
        "transform": "PIXEL_DIAGNOSTIC_ONLY",
        "attention": False,
    }


def epicenter_evidence(payload: dict | None) -> dict:
    """Uses AOI's already-cropped large and small pairs; never guesses a box."""
    p = payload if isinstance(payload, dict) else {}
    out = {}
    for key, reference_key, observed_key in (
        ("major", "large_reference", "large"),
        ("minor", "small_reference", "small"),
    ):
        out[key] = evidence_views(p.get(reference_key), p.get(observed_key))
    return out
