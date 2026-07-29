"""Alinhamento visual robusto da mesma ROI entre gabarito e teste.

A captura do epicentro não é alterada. Este módulo corrige somente a
translação aplicada depois do recorte:

* o Comparador Estrutural avalia os dois sentidos do deslocamento e conserva
  aquele que realmente aproxima o teste do gabarito;
* o Motor FALTANDO mantém o julgamento sobre o recorte original, mas mostra no
  debugger uma versão alinhada para facilitar a comparação visual.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _safe_pair(reference: np.ndarray, test: np.ndarray):
    if (
        not isinstance(reference, np.ndarray)
        or not isinstance(test, np.ndarray)
        or reference.size == 0
        or test.size == 0
    ):
        return None, None
    reference = reference.copy()
    test = test.copy()
    if reference.shape != test.shape:
        test = cv2.resize(
            test,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return reference, test


def _gray_float(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _warp(
    image: np.ndarray,
    dx: float,
    dy: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT101,
) -> np.ndarray:
    matrix = np.asarray(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=0,
    )


def _valid_region(shape, dx: float, dy: float):
    height, width = shape[:2]
    left = max(0, int(math.ceil(dx)))
    right = min(width, width + int(math.floor(dx)))
    top = max(0, int(math.ceil(dy)))
    bottom = min(height, height + int(math.floor(dy)))
    margin = 3
    left += margin
    right -= margin
    top += margin
    bottom -= margin
    if right - left < 12 or bottom - top < 12:
        return 0, 0, width, height
    return left, top, right, bottom


def _alignment_score(reference_gray, candidate_gray, dx: float, dy: float) -> float:
    x1, y1, x2, y2 = _valid_region(reference_gray.shape, dx, dy)
    reference = reference_gray[y1:y2, x1:x2]
    candidate = candidate_gray[y1:y2, x1:x2]
    if reference.size < 64 or candidate.size != reference.size:
        return 0.0

    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    )
    correlation = (
        float(np.sum(reference_centered * candidate_centered) / denominator)
        if denominator > 1e-8
        else 0.0
    )
    correlation_score = float(np.clip((correlation + 1.0) * 0.5, 0.0, 1.0))

    reference_u8 = np.clip(reference * 255.0, 0, 255).astype(np.uint8)
    candidate_u8 = np.clip(candidate * 255.0, 0, 255).astype(np.uint8)
    reference_edges = cv2.Canny(reference_u8, 45, 145) > 0
    candidate_edges = cv2.Canny(candidate_u8, 45, 145) > 0
    intersection = float(np.count_nonzero(reference_edges & candidate_edges))
    edge_total = float(np.count_nonzero(reference_edges) + np.count_nonzero(candidate_edges))
    edge_dice = (2.0 * intersection / edge_total) if edge_total > 0 else 1.0

    mean_difference = float(np.mean(np.abs(reference - candidate)))
    photometric = float(np.clip(1.0 - mean_difference / 0.42, 0.0, 1.0))
    return float(
        np.clip(
            0.58 * correlation_score + 0.27 * edge_dice + 0.15 * photometric,
            0.0,
            1.0,
        )
    )


def _candidate_shifts(reference_gray, test_gray, maximum_shift: int):
    candidates = {(0.0, 0.0)}

    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            80,
            1e-6,
        )
        _, warp = cv2.findTransformECC(
            reference_gray,
            test_gray,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            3,
        )
        ecc_dx = float(warp[0, 2])
        ecc_dy = float(warp[1, 2])
        candidates.add((ecc_dx, ecc_dy))
        candidates.add((-ecc_dx, -ecc_dy))
    except cv2.error:
        pass

    try:
        phase_shift, _ = cv2.phaseCorrelate(reference_gray, test_gray)
        phase_dx, phase_dy = float(phase_shift[0]), float(phase_shift[1])
        candidates.add((phase_dx, phase_dy))
        candidates.add((-phase_dx, -phase_dy))
    except cv2.error:
        pass

    bounded = set()
    for dx, dy in candidates:
        dx = float(np.clip(dx, -maximum_shift, maximum_shift))
        dy = float(np.clip(dy, -maximum_shift, maximum_shift))
        rounded_x = int(round(dx))
        rounded_y = int(round(dy))
        for offset_y in range(-2, 3):
            for offset_x in range(-2, 3):
                candidate_x = int(np.clip(rounded_x + offset_x, -maximum_shift, maximum_shift))
                candidate_y = int(np.clip(rounded_y + offset_y, -maximum_shift, maximum_shift))
                bounded.add((float(candidate_x), float(candidate_y)))
    return bounded


def best_translation(reference: np.ndarray, test: np.ndarray):
    """Retorna teste alinhado, deslocamento aplicado, score e ganho."""
    reference, test = _safe_pair(reference, test)
    if reference is None:
        return test, (0.0, 0.0), 0.0, 0.0

    height, width = reference.shape[:2]
    if min(height, width) < 12:
        return test.copy(), (0.0, 0.0), 0.0, 0.0

    reference_gray = _gray_float(reference)
    test_gray = _gray_float(test)
    maximum_shift = max(3, min(32, int(round(min(height, width) * 0.20))))

    base_score = _alignment_score(reference_gray, test_gray, 0.0, 0.0)
    best_score = base_score
    best_shift = (0.0, 0.0)
    best_image = test.copy()

    for dx, dy in _candidate_shifts(reference_gray, test_gray, maximum_shift):
        if dx == 0.0 and dy == 0.0:
            continue
        candidate = _warp(test, dx, dy)
        candidate_gray = _gray_float(candidate)
        score = _alignment_score(reference_gray, candidate_gray, dx, dy)
        if score > best_score:
            best_score = score
            best_shift = (dx, dy)
            best_image = candidate

    gain = float(best_score - base_score)
    minimum_gain = 0.018
    if gain < minimum_gain:
        return test.copy(), (0.0, 0.0), float(base_score), 0.0
    return best_image, best_shift, float(best_score), gain


def _crop_pair(full_reference, full_test, box):
    reference, test = _safe_pair(full_reference, full_test)
    if reference is None or not box or len(box) < 4:
        return None, None, None
    try:
        x, y, width, height = (int(round(float(value))) for value in box[:4])
    except (TypeError, ValueError):
        return None, None, None
    image_height, image_width = reference.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + max(1, width))
    y2 = min(image_height, y + max(1, height))
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    return (
        reference[y1:y2, x1:x2].copy(),
        test[y1:y2, x1:x2].copy(),
        (x1, y1, x2 - x1, y2 - y1),
    )


def install_roi_visual_alignment(silk_expert_cls, missing_expert_cls) -> None:
    if getattr(silk_expert_cls, "_robust_roi_alignment_installed", False):
        return

    def structural_alignment(reference, test):
        aligned, shift, score, _ = best_translation(reference, test)
        return aligned, shift, score

    silk_expert_cls._align_test_to_reference = staticmethod(structural_alignment)
    silk_expert_cls._robust_roi_alignment_installed = True

    original_missing_analyze = missing_expert_cls.analyze

    def missing_analyze(
        self,
        full_reference,
        full_test,
        global_box_info=None,
        aoi_info=None,
        aoi_epicenters=None,
    ):
        result = original_missing_analyze(
            self,
            full_reference,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )
        if not isinstance(result, dict) or not result.get("missing_active", False):
            return result

        box = result.get("missing_roi_box")
        reference, test, resolved_box = _crop_pair(full_reference, full_test, box)
        if resolved_box is None:
            return result

        aligned_test, shift, score, gain = best_translation(reference, test)
        dx, dy = shift
        result["missing_test_input_raw"] = test
        result["missing_test_aligned_raw"] = aligned_test
        result["missing_visual_alignment_shift"] = (float(dx), float(dy))
        result["missing_visual_alignment_dx"] = float(dx)
        result["missing_visual_alignment_dy"] = float(dy)
        result["missing_visual_alignment_score"] = float(score)
        result["missing_visual_alignment_gain"] = float(gain)
        result["missing_visual_alignment_applied"] = bool(dx != 0.0 or dy != 0.0)

        if dx == 0.0 and dy == 0.0:
            return result

        anomaly_mask = result.get("roi_anomaly_mask")
        residual = result.get("missing_residual_map")
        if not isinstance(anomaly_mask, np.ndarray) or anomaly_mask.size == 0:
            return result
        if not isinstance(residual, np.ndarray) or residual.size == 0:
            return result

        aligned_mask = _warp(
            anomaly_mask,
            dx,
            dy,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
        )
        aligned_residual = _warp(
            residual.astype(np.float32),
            dx,
            dy,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
        )

        result["missing_test_view_raw"] = result.get("missing_test_view")
        result["missing_reconstruction_view_raw"] = result.get(
            "missing_reconstruction_view"
        )
        reference_view, test_view, reconstruction, heatmap = self._views(
            reference,
            aligned_test,
            aligned_residual,
            aligned_mask,
        )
        result["missing_reference_view"] = reference_view
        result["missing_test_view"] = test_view
        result["missing_reconstruction_view"] = reconstruction
        result["missing_heatmap_aligned"] = heatmap
        result["missing_display_mode"] = "aligned_debug"
        return result

    missing_expert_cls.analyze = missing_analyze
    missing_expert_cls._robust_roi_alignment_installed = True


__all__ = ["best_translation", "install_roi_visual_alignment"]
