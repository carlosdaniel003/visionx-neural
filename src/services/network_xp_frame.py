"""Fonte única do frame completo recebido da AOI Windows XP.

Tanto o botão "Copiar imagem XP" quanto o arquivo visual NG devem usar
exatamente este contrato para nunca divergirem entre si.
"""

from __future__ import annotations

import numpy as np


def network_xp_record_event_id(panel) -> str:
    record = getattr(panel, "network_intake_last_validation", None)
    if not isinstance(record, dict):
        return ""
    return str(record.get("event_id", "") or "")


def network_xp_frame_available(panel) -> bool:
    """Confirma que o frame preservado pertence ao mesmo evento do diagnóstico."""
    image = getattr(panel, "network_intake_last_image", None)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return False

    record_event = network_xp_record_event_id(panel)
    image_event = str(
        getattr(panel, "network_intake_last_image_event_id", "") or ""
    )
    return bool(record_event and image_event and record_event == image_event)


def network_xp_frame_snapshot(panel) -> np.ndarray | None:
    """Retorna cópia exata do frame que o botão Copiar imagem XP deve usar."""
    if not network_xp_frame_available(panel):
        return None

    image = getattr(panel, "network_intake_last_image", None)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    return image.copy()


__all__ = [
    "network_xp_frame_available",
    "network_xp_frame_snapshot",
    "network_xp_record_event_id",
]
