"""Resolve quais imagens completas devem ir ao arquivo visual após OK/NG."""

from __future__ import annotations

import numpy as np

from src.utils.text_normalizer import normalize_aoi_text


LIGHTING_ORDER = ("SIDE", "TOP", "MID")
ADHESIVE_CATEGORY = "MUITO ADESIVO"


def _valid_image(value) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim >= 2
    )


def _is_adhesive_category(category: str) -> bool:
    normalized, _value = normalize_aoi_text(str(category or ""))
    return normalized == ADHESIVE_CATEGORY


def archive_image_candidates(
    panel,
    *,
    event_id: str,
    category: str,
    primary_image: np.ndarray | None,
) -> list[tuple[np.ndarray, str]]:
    """Retorna 1 imagem normal ou SIDE/TOP/MID da mesma peça de adesivo.

    O segundo item da tupla é a categoria usada no nome do arquivo. Para
    adesivo, inclui a iluminação para que os três PNGs sejam distinguíveis.
    """
    normalized_event_id = str(event_id or "").strip()
    normalized_category = str(category or "").strip()

    if _is_adhesive_category(normalized_category):
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
                    continue
                resolved.append(
                    (
                        image.copy(),
                        f"{normalized_category}_{mode}",
                    )
                )
            if resolved:
                return resolved

    if _valid_image(primary_image):
        return [(primary_image.copy(), normalized_category)]

    return []


__all__ = [
    "ADHESIVE_CATEGORY",
    "LIGHTING_ORDER",
    "archive_image_candidates",
]
