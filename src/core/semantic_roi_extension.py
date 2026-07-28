"""Restringe o especialista semântico à ROI exata enviada pela AOI.

O especialista não pode mais recorrer à imagem completa. Se o epicentro estiver
ausente ou inválido, ele se abstém da análise semântica.
"""

from __future__ import annotations

import cv2
import numpy as np


def _empty_roi_result(reason: str) -> dict:
    return {
        "is_defect": False,
        "score": 0.0,
        "reason": reason,
        "bounding_box": None,
        "semantic_active": False,
        "semantic_scope": "epicenter_roi",
        "semantic_loss": 0.0,
        "semantic_global_loss": 0.0,
        "semantic_local_evidence": 0.0,
        "semantic_distance_cosine": 0.0,
        "query_emb": None,
        "ref_emb": None,
        "semantic_delta": [],
        "semantic_debug": None,
        "semantic_reconstruction_map": None,
        "semantic_reconstruction_view": None,
        "semantic_focus_reference": None,
        "semantic_focus_test": None,
        "semantic_focus_box": None,
    }


def _extract_exact_roi(reference, test, epicenters):
    if (
        not isinstance(reference, np.ndarray)
        or not isinstance(test, np.ndarray)
        or reference.size == 0
        or test.size == 0
        or not epicenters
    ):
        return None, None, None

    if reference.shape != test.shape:
        test = cv2.resize(
            test,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    candidate = epicenters[0]
    if not candidate or len(candidate) < 4:
        return None, None, None

    try:
        x, y, width, height = (int(round(float(value))) for value in candidate[:4])
    except (TypeError, ValueError):
        return None, None, None

    image_height, image_width = reference.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + max(1, width))
    y2 = min(image_height, y + max(1, height))
    if x2 <= x1 or y2 <= y1:
        return None, None, None

    reference_roi = reference[y1:y2, x1:x2].copy()
    test_roi = test[y1:y2, x1:x2].copy()
    if (
        reference_roi.size == 0
        or test_roi.size == 0
        or min(reference_roi.shape[:2]) < 3
    ):
        return None, None, None

    if reference_roi.shape != test_roi.shape:
        test_roi = cv2.resize(
            test_roi,
            (reference_roi.shape[1], reference_roi.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return reference_roi, test_roi, (x1, y1, x2 - x1, y2 - y1)


def install_semantic_roi_extension(semantic_expert_cls) -> None:
    """Faz o embedding, delta e reconstrução usarem exclusivamente o epicentro."""
    if getattr(semantic_expert_cls, "_epicenter_roi_only_installed", False):
        return

    original_analyze = semantic_expert_cls.analyze

    def analyze(
        self,
        crop_gab,
        crop_test,
        global_box_info=None,
        aoi_info=None,
        aoi_epicenters=None,
    ):
        reference_roi, test_roi, absolute_box = _extract_exact_roi(
            crop_gab,
            crop_test,
            aoi_epicenters,
        )
        if absolute_box is None:
            return _empty_roi_result(
                "Debug semântico abstido: ROI do epicentro ausente ou inválida"
            )

        # A implementação original recebe agora somente o recorte canônico. Sem
        # epicenters locais, ela trata todo o argumento como seu universo visual.
        result = original_analyze(
            self,
            reference_roi,
            test_roi,
            global_box_info,
            aoi_info,
            None,
        )
        if not isinstance(result, dict):
            return _empty_roi_result("Debug semântico abstido: resultado inválido")

        result["semantic_active"] = True
        result["semantic_scope"] = "epicenter_roi"
        result["semantic_focus_box"] = absolute_box
        result["semantic_focus_reference"] = reference_roi
        result["semantic_focus_test"] = test_roi

        debug = result.get("semantic_debug")
        if isinstance(debug, dict):
            debug["analysis_scope"] = "epicenter_roi"
            debug["source_scope"] = "aoi_epicenter"
            debug["focus_box"] = [int(value) for value in absolute_box]
            debug["focus_size"] = [
                int(reference_roi.shape[1]),
                int(reference_roi.shape[0]),
            ]
            debug["full_image_fallback"] = False
        return result

    semantic_expert_cls.analyze = analyze
    semantic_expert_cls._epicenter_roi_only_installed = True


def install_semantic_roi_widget(widget_cls) -> None:
    """Exibe no primeiro campo da telemetria que a análise usa só a ROI."""
    if getattr(widget_cls, "_epicenter_roi_telemetry_installed", False):
        return

    original_telemetry_lines = widget_cls._telemetry_lines

    def telemetry_lines(self):
        lines = list(original_telemetry_lines(self))
        scope = (
            self.debug.get("analysis_scope", "")
            if isinstance(self.debug, dict)
            else ""
        )
        if lines and scope == "epicenter_roi":
            lines[0] = "escopo=ROI DO EPICENTRO • " + lines[0]
        return lines

    widget_cls._telemetry_lines = telemetry_lines
    widget_cls._epicenter_roi_telemetry_installed = True


__all__ = [
    "install_semantic_roi_extension",
    "install_semantic_roi_widget",
]
