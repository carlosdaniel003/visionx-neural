"""Utilitários compartilhados de deduplicação dos arquivos visuais OK/NG."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def image_fingerprint(image: np.ndarray) -> str:
    """SHA-256 determinístico do conteúdo visual exato."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return ""

    array = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii", errors="ignore"))
    digest.update(str(tuple(array.shape)).encode("ascii", errors="ignore"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_archive_fingerprints(output_dir: Path) -> set[str]:
    """Indexa PNGs já existentes para impedir repetição após reiniciar o ODIN."""
    fingerprints: set[str] = set()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for image_path in path.glob("*.png"):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            fingerprint = image_fingerprint(image)
            if fingerprint:
                fingerprints.add(fingerprint)
        except Exception:
            continue

    return fingerprints


def unique_archive_target(base_target: Path) -> Path:
    """Evita sobrescrever imagens diferentes que caiam no mesmo minuto/nome."""
    target = Path(base_target)
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    index = 2

    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


__all__ = [
    "image_fingerprint",
    "load_archive_fingerprints",
    "unique_archive_target",
]
