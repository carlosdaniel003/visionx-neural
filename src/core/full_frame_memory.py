"""Terceira escala da memória visual: quadro completo da área de inspeção.

A memória existente continua preservada:
- epicentro local;
- contexto maior do componente.

Esta extensão acrescenta uma assinatura do quadro completo de gabarito/teste,
independente das caixas detectadas. O objetivo é impedir que um epicentro
incorreto ou uma caixa maior deslocada apaguem da memória um defeito evidente
em outra região da imagem.

JSONs antigos continuam comparáveis: quando a terceira escala não existe em um
dos lados, a política anterior é usada sem alteração.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dual_scale_memory import (
    build_component_context_signature,
    compare_component_context_signatures,
    valid_context_signature,
)


FULL_FRAME_SCHEMA = "visionx.full_frame_memory.v1"
PREVIOUS_SCALE_WEIGHT = 0.80
FULL_FRAME_WEIGHT = 0.20


def build_full_frame_signature(reference, test) -> dict:
    """Reutiliza o descritor contextual, mas força a ROI para o quadro inteiro."""
    if (
        not isinstance(reference, np.ndarray)
        or not isinstance(test, np.ndarray)
        or reference.size == 0
        or test.size == 0
    ):
        return {}

    height, width = reference.shape[:2]
    if height <= 0 or width <= 0:
        return {}

    signature = build_component_context_signature(
        reference,
        test,
        context_box=(0, 0, width, height),
    )
    if not valid_context_signature(signature):
        return {}

    output = dict(signature)
    output["memory_role"] = "full_frame"
    return output


def attach_full_frame_signature(
    anomaly_signature: dict,
    reference,
    test,
) -> dict:
    """Anexa o quadro completo sem alterar as escalas já existentes."""
    if not isinstance(anomaly_signature, dict):
        return anomaly_signature

    output = dict(anomaly_signature)
    full_frame = build_full_frame_signature(reference, test)
    if valid_context_signature(full_frame):
        output["full_frame_memory_schema"] = FULL_FRAME_SCHEMA
        output["full_frame_signature"] = full_frame
        scales = list(output.get("memory_scales", []) or [])
        if "full_frame" not in scales:
            scales.append("full_frame")
        output["memory_scales"] = scales
    return output


def compare_with_full_frame(
    previous_compare,
    query_signature: dict,
    stored_signature: dict,
):
    """Combina a política anterior com 20% de contexto global quando disponível."""
    previous_similarity, previous_breakdown = previous_compare(
        query_signature,
        stored_signature,
    )

    query_full = (
        query_signature.get("full_frame_signature", {})
        if isinstance(query_signature, dict)
        else {}
    )
    stored_full = (
        stored_signature.get("full_frame_signature", {})
        if isinstance(stored_signature, dict)
        else {}
    )

    if not (
        valid_context_signature(query_full)
        and valid_context_signature(stored_full)
    ):
        return float(previous_similarity), previous_breakdown

    full_similarity, full_breakdown = compare_component_context_signatures(
        query_full,
        stored_full,
    )
    combined = float(
        np.clip(
            float(previous_similarity) * PREVIOUS_SCALE_WEIGHT
            + float(full_similarity) * FULL_FRAME_WEIGHT,
            0.0,
            1.0,
        )
    )
    return combined, {
        "schema": FULL_FRAME_SCHEMA,
        "policy": "previous_memory_plus_full_frame",
        "similarity": combined,
        "previous_similarity": float(previous_similarity),
        "full_frame_similarity": float(full_similarity),
        "scale_weights": {
            "previous_memory": PREVIOUS_SCALE_WEIGHT,
            "full_frame": FULL_FRAME_WEIGHT,
        },
        "previous": previous_breakdown,
        "full_frame": full_breakdown,
    }


def install_full_frame_memory(
    anomaly_memory_module,
    best_match_module,
    dataset_manager_module,
) -> None:
    """Expande build/compare depois da dual-scale, mantendo fallback legado."""
    if getattr(anomaly_memory_module, "_full_frame_memory_installed", False):
        return

    previous_build = anomaly_memory_module.build_anomaly_signature
    previous_compare = best_match_module.compare_anomaly_signatures

    def build_three_scale_signature(
        reference,
        test,
        detail,
        aoi_info=None,
        focus_box=None,
    ):
        base = previous_build(
            reference,
            test,
            detail,
            aoi_info,
            focus_box,
        )
        return attach_full_frame_signature(base, reference, test)

    def compare_three_scale_signatures(query_signature, stored_signature):
        return compare_with_full_frame(
            previous_compare,
            query_signature,
            stored_signature,
        )

    anomaly_memory_module.build_anomaly_signature = build_three_scale_signature
    best_match_module.compare_anomaly_signatures = compare_three_scale_signatures
    if hasattr(dataset_manager_module, "build_anomaly_signature"):
        dataset_manager_module.build_anomaly_signature = (
            build_three_scale_signature
        )

    anomaly_memory_module._full_frame_memory_installed = True


__all__ = [
    "FULL_FRAME_SCHEMA",
    "FULL_FRAME_WEIGHT",
    "PREVIOUS_SCALE_WEIGHT",
    "attach_full_frame_signature",
    "build_full_frame_signature",
    "compare_with_full_frame",
    "install_full_frame_memory",
]
