"""Análise independente das iluminações auxiliares SIDE/TOP/MID.

O nome do módulo é mantido por compatibilidade histórica. TOP e MID percorrem
o mesmo pipeline técnico usado pela imagem SIDE em qualquer categoria AOI:
detect_anomalies -> EpicenterExtractor -> MoEOrchestrator.inspect.

O resultado retornado é diagnóstico por iluminação. Ele não substitui
current_analysis; somente a camada posterior de fusão multilight pode promover
um julgamento final da peça.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor
from src.core.inspection import detect_anomalies


def valid_image(value) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim >= 2
    )


def build_lighting_context(
    sample_crop: np.ndarray,
    ng_crop: np.ndarray,
) -> dict:
    """Executa uma única extração geométrica reutilizável por visual e MoE."""
    if not valid_image(sample_crop) or not valid_image(ng_crop):
        return {
            "valid": False,
            "raw_anomalies": [],
            "old_epicenters": [],
            "global_box_info": {},
            "real_epicenters": [],
            "focus_gab": np.array([]),
            "focus_ng": np.array([]),
        }

    (
        raw_anomalies,
        old_epicenters,
        global_box_info,
        _gab_focus,
        _test_focus,
    ) = detect_anomalies(sample_crop, ng_crop)

    real_epicenters, focus_gab, focus_ng = EpicenterExtractor.extract_focus(
        sample_crop,
        ng_crop,
        old_epicenters,
        global_box_info,
    )

    return {
        "valid": True,
        "raw_anomalies": list(raw_anomalies or []),
        "old_epicenters": list(old_epicenters or []),
        "global_box_info": dict(global_box_info or {}),
        "real_epicenters": list(real_epicenters or []),
        "focus_gab": (
            focus_gab.copy()
            if valid_image(focus_gab)
            else np.array([])
        ),
        "focus_ng": (
            focus_ng.copy()
            if valid_image(focus_ng)
            else np.array([])
        ),
    }


def analyze_lighting(
    orchestrator,
    sample_crop: np.ndarray,
    ng_crop: np.ndarray,
    aoi_info: dict | None,
    lighting_mode: str,
    context: dict | None = None,
) -> dict | None:
    """Executa os especialistas sem promover o resultado a decisão da peça."""
    if orchestrator is None:
        return None
    if not valid_image(sample_crop) or not valid_image(ng_crop):
        return None

    frame_context = (
        context
        if isinstance(context, dict)
        else build_lighting_context(sample_crop, ng_crop)
    )
    if not frame_context.get("valid", False):
        return None

    info = deepcopy(aoi_info if isinstance(aoi_info, dict) else {})
    normalized_mode = str(lighting_mode or "").strip().upper()
    info["lighting_mode"] = normalized_mode

    analysis = orchestrator.inspect(
        sample_crop,
        ng_crop,
        frame_context.get("raw_anomalies", []),
        info,
        frame_context.get("global_box_info", {}),
        frame_context.get("real_epicenters", []),
    )
    if not isinstance(analysis, dict):
        return None

    analysis["lighting_mode"] = normalized_mode
    analysis["multilight_visual_analysis"] = True
    analysis["eligible_for_final_decision"] = False

    detail = analysis.setdefault("detail", {})
    detail["lighting_mode"] = normalized_mode
    detail["multilight_visual_analysis"] = True
    detail["eligible_for_final_decision"] = False
    return analysis


__all__ = [
    "analyze_lighting",
    "build_lighting_context",
    "valid_image",
]
