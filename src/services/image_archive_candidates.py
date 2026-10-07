"""Resolve quais imagens completas devem ir ao arquivo visual após OK/NG."""

from __future__ import annotations

import numpy as np

LIGHTING_ORDER = ("SIDE", "TOP", "MID")


def _valid_image(value) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim >= 2
    )


def archive_image_candidates(
    panel,
    *,
    event_id: str,
    category: str,
    primary_image: np.ndarray | None,
) -> list[tuple[np.ndarray, str]]:
    """Retorna 1 imagem normal ou SIDE/TOP/MID da mesma peça multilight.

    O segundo item da tupla é a categoria usada no nome do arquivo. Em uma
    sessão multilight completa, inclui a iluminação para distinguir os 3 PNGs.
    """
    normalized_event_id = str(event_id or "").strip()
    normalized_category = str(category or "").strip()

    multilight_event_id = str(
        getattr(panel, "adhesive_multilight_last_event_id", "") or ""
    ).strip()
    frames = getattr(
        panel,
        "adhesive_multilight_last_source_frames",
        {},
    )

    if (
        normalized_event_id
        and normalized_event_id == multilight_event_id
        and isinstance(frames, dict)
    ):
        resolved = []
        for mode in LIGHTING_ORDER:
            image = frames.get(mode)
            if not _valid_image(image):
                resolved = []
                break
            resolved.append(
                (
                    image.copy(),
                    f"{normalized_category}_{mode}",
                )
            )
        if len(resolved) == len(LIGHTING_ORDER):
            return resolved

    if _valid_image(primary_image):
        return [(primary_image.copy(), normalized_category)]

    return []


__all__ = [
    "LIGHTING_ORDER",
    "archive_image_candidates",
]
