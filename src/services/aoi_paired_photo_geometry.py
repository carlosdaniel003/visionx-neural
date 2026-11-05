"""Consistência geométrica dos dois painéis da mesma captura AOI.

Quando um painel tem contornos válidos e o outro não tem limite inferior
detectável, usa-se o retângulo de foto do painel irmão, relativo à barra
correspondente. Não classifica OK/NG, não acessa arquivo nem usa categoria.
"""
from __future__ import annotations

import numpy as np


def reconcile_paired_photos(
    frame: np.ndarray, images: dict, geometries: dict,
) -> tuple[dict, dict]:
    """Obtém dois recortes co-registrados sem usar altura fixa ou nome de PNG.

    Em ausência de prova geométrica no painel irmão, preserva os dois
    recortes originais e relata explicitamente que não houve correção.
    """
    result = dict(images)
    audit = {
        "schema": "visionx.aoi_paired_photo_geometry.v1",
        "status": "UNCHANGED",
        "corrected_panels": [],
        "before_shapes": {
            key: list(value.shape) if isinstance(value, np.ndarray) else []
            for key, value in result.items()
        },
    }
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3
            or not isinstance(geometries, dict)
            or set(result) != {"SAMPLE", "TEST"}):
        audit["status"] = "INVALID_INPUT"
        return result, audit
    reliable = [
        key for key in result
        if isinstance(geometries.get(key), dict)
        and geometries[key].get("boundary_source") == "DETECTED"
    ]
    ambiguous = [
        key for key in result
        if isinstance(geometries.get(key), dict)
        and geometries[key].get("boundary_source") == "UNBOUNDED"
    ]
    if len(reliable) != 1 or len(ambiguous) != 1:
        audit["status"] = (
            "BOTH_BOUNDED" if len(reliable) == 2
            else "BOTH_UNBOUNDED" if len(ambiguous) == 2
            else "NO_SHARED_BOUNDARY"
        )
        audit["after_shapes"] = audit["before_shapes"]
        return result, audit

    known, target = reliable[0], ambiguous[0]
    source_geom, target_geom = geometries[known], geometries[target]
    expected_top = source_geom.get("relative_top")
    expected_bottom = source_geom.get("relative_bottom")
    bar_bottom = target_geom.get("bar_bottom")
    source_bar_bottom = source_geom.get("bar_bottom")
    x1, x2 = target_geom.get("x1"), target_geom.get("x2")
    try:
        coords = [
            expected_top, expected_bottom, bar_bottom,
            source_bar_bottom, x1, x2,
        ]
        if not all(isinstance(a, int) for a in coords):
            raise ValueError("Geometria incompleta")
        # A AOI apresenta dois quadros na mesma linha; rejeitar quando
        # as supostas barras pertencem a regiões verticalmente distantes.
        # Usa a mesma tolerância do detector de barras irmãs existente.
        if abs(bar_bottom - source_bar_bottom) > 30:
            raise ValueError("Barras não co-registradas")
        top, bottom = bar_bottom + expected_top, bar_bottom + expected_bottom
        bounded_height = int(result[known].shape[0])
        if (bottom - top != bounded_height or bottom <= top
                or top < 0 or bottom > frame.shape[0]
                or x1 < 0 or x2 > frame.shape[1] or x2 <= x1):
            raise ValueError("Retângulo irmão fora da captura AOI")
        # O alvo é uma foto abaixo da respectiva barra, no mesmo plano
        # visual do painel confiável (não o strip artificial de 600 px).
        proposed = frame[top:bottom, x1:x2].copy()
        if (proposed.size == 0
                or proposed.shape[0] != bounded_height
                or proposed.shape[1] != x2 - x1):
            raise ValueError("Foto alinhada incompleta")
    except (TypeError, ValueError, IndexError):
        audit["status"] = "ALIGNMENT_UNAVAILABLE"
        audit["after_shapes"] = audit["before_shapes"]
        return result, audit

    result[target] = proposed
    audit.update(
        status="ALIGNED_FROM_SIBLING_BOUNDARY",
        source_panel=known,
        corrected_panels=[target],
        after_shapes={key: list(value.shape) for key, value in result.items()},
        source_offsets={"top": expected_top, "bottom": expected_bottom},
    )
    return result, audit


__all__ = ["reconcile_paired_photos"]
