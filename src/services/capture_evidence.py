"""Fonte genérica da evidência visual e do debug da última captura analisada.

A origem pode ser Windows XP (rede) ou captura local MSS. Esta camada existe
para que "Copiar debug" e "Copiar imagem" nunca reutilizem evidência de outro
ciclo/origem.
"""

from __future__ import annotations

import numpy as np


def capture_debug_record(panel) -> dict:
    record = getattr(panel, "capture_debug_last_record", None)
    return dict(record) if isinstance(record, dict) else {}


def capture_debug_event_id(panel) -> str:
    record = capture_debug_record(panel)
    return str(record.get("event_id", "") or "")


def capture_debug_source(panel) -> str:
    record = capture_debug_record(panel)
    return str(record.get("source", "") or "")


def capture_image_available(panel) -> bool:
    image = getattr(panel, "capture_debug_last_image", None)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return False

    record_event = capture_debug_event_id(panel)
    image_event = str(
        getattr(panel, "capture_debug_last_image_event_id", "") or ""
    )
    return bool(record_event and image_event and record_event == image_event)


def capture_image_snapshot(panel) -> np.ndarray | None:
    if not capture_image_available(panel):
        return None

    image = getattr(panel, "capture_debug_last_image", None)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    return image.copy()


def store_capture_evidence(panel, image, record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    event_id = str(record.get("event_id", "") or "")
    if not event_id:
        return False
    if not isinstance(image, np.ndarray) or image.size == 0:
        return False

    panel.capture_debug_last_image = image.copy()
    panel.capture_debug_last_image_event_id = event_id
    panel.capture_debug_last_record = dict(record)
    return True


def update_capture_debug_record(panel, record: dict) -> bool:
    if not isinstance(record, dict):
        return False

    event_id = str(record.get("event_id", "") or "")
    image_event = str(
        getattr(panel, "capture_debug_last_image_event_id", "") or ""
    )
    if not event_id or event_id != image_event:
        return False

    panel.capture_debug_last_record = dict(record)
    return True


__all__ = [
    "capture_debug_event_id",
    "capture_debug_record",
    "capture_debug_source",
    "capture_image_available",
    "capture_image_snapshot",
    "store_capture_evidence",
    "update_capture_debug_record",
]
