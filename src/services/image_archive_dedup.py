"""Deduplicação exata e visual conservadora dos arquivos OK/NG.

A comparação visual é usada SOMENTE na fila de arquivamento. Nunca modifica
julgamento, memória verificada, modelos CNN ou comandos industriais.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import cv2
import numpy as np

from src.services.image_archive_naming import safe_archive_category


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


# A imagem é a tela AOI completa (não apenas o componente). Perfis pequenos e
# PNGs sintéticos antigos continuam com a regra SHA exato.
_PARTS = (
    (0.0, 0.0, 1.0, 1.0),
    (.02, .32, .49, .98),
    (.52, .32, .99, .98),
)
_VISUAL_THRESHOLDS = (.985, .975, .965, .82)
_MAX_MAE = (.04, .045, .045, .060)
_ARCHIVE_STEM = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_(.+)$")


def archive_category_from_filename(path: Path) -> str:
    """Recupera categoria+luz de nomes atuais e _2/_3; legado sem luz=SIDE."""
    match = _ARCHIVE_STEM.match(Path(path).stem.upper())
    if not match:
        return ""
    stem = re.sub(r"_([2-9]|[1-9]\d+)$", "", match.group(1))
    if stem.endswith(("_SIDE", "_TOP", "_MID")):
        return stem
    return stem + "_SIDE"


def archive_visual_scope(category: str) -> str:
    """Mesmo contexto de categoria e iluminação; nunca cruza OK e NG."""
    value = safe_archive_category(category)
    if value.endswith(("_SIDE", "_TOP", "_MID")):
        return value
    return value + "_SIDE"


def archive_visual_descriptor(image: np.ndarray):
    """Quatro mapas compactos: tela, gabarito, teste e diferença entre ambos."""
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or image.shape[0] < 240 or image.shape[1] < 600):
        return None
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parts = []
    for x0, y0, x1, y1 in _PARTS:
        crop = gray[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        if crop.size == 0:
            return None
        parts.append(
            cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
            .astype(np.float32)
        )
    difference = cv2.absdiff(parts[1], parts[2])
    return (round(w / h, 3), (*parts, difference))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.reshape(-1), right.reshape(-1)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    # Quadros uniformes não são prova de equivalência visual.
    return float(np.dot(a, b) / denom) if denom > 1e-5 else -1.0


def near_duplicate_visual(a, b) -> bool:
    """Exige concordância simultânea da tela, das duas imagens e da anomalia."""
    if a is None or b is None or abs(a[0] - b[0]) > .015:
        return False
    for aa, bb, cutoff, mae_limit in zip(
        a[1], b[1], _VISUAL_THRESHOLDS, _MAX_MAE
    ):
        if _correlation(aa, bb) < cutoff:
            return False
        if float(np.mean(np.abs(aa - bb))) / 255.0 > mae_limit:
            return False
    return True


class ArchiveVisualIndex:
    """Índice de pequenas assinaturas, propriedade exclusiva do worker."""
    def __init__(self):
        self._by_scope: dict[str, list] = {}

    def register(self, image: np.ndarray, category: str) -> None:
        descriptor = archive_visual_descriptor(image)
        if descriptor is not None:
            self._by_scope.setdefault(archive_visual_scope(category), []).append(
                descriptor
            )

    def register_existing(self, image: np.ndarray, path: Path) -> None:
        scope = archive_category_from_filename(path)
        descriptor = archive_visual_descriptor(image)
        if scope and descriptor is not None:
            self._by_scope.setdefault(scope, []).append(descriptor)

    def contains_similar(self, image: np.ndarray, category: str) -> bool:
        descriptor = archive_visual_descriptor(image)
        if descriptor is None:
            return False
        return any(near_duplicate_visual(descriptor, old)
                   for old in self._by_scope.get(archive_visual_scope(category), ()))


def load_archive_fingerprints(
    output_dir: Path, *, visual_index: ArchiveVisualIndex | None = None
) -> set[str]:
    """Indexa imagens uma só vez ao iniciar fila, inclusive nomes legados."""
    fingerprints: set[str] = set()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for image_path in path.glob("*.png"):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            fingerprint = image_fingerprint(image)
            if fingerprint:
                fingerprints.add(fingerprint)
                if visual_index is not None:
                    visual_index.register_existing(image, image_path)
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
    "ArchiveVisualIndex",
    "archive_category_from_filename",
    "archive_visual_descriptor",
    "archive_visual_scope",
    "image_fingerprint",
    "load_archive_fingerprints",
    "near_duplicate_visual",
    "unique_archive_target",
]
