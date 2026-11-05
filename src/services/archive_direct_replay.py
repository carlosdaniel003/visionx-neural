"""Replay da fonte PNG real, sem fotografar o visualizador com MSS.

O arquivo é lido no worker QThread pelo MESMO extrator AOI usado por
captura local e XP. Não há rótulo forçado, memória fictícia nem comando XP.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

import cv2
import numpy as np

from src.config.settings import BASE_DIR
from src.services.image_archive_dedup import (
    archive_category_from_filename, image_fingerprint,
)
from src.services.screen_monitor import ScreenMonitor

SOURCE_SCHEMA = "visionx.archive_direct_replay.v1"


def validate_archive_source(path: Path, *, root: Path = BASE_DIR) -> dict:
    """Permite somente PNG regular nas duas pastas do acervo, sem traversals."""
    root = Path(root).resolve()
    target = Path(path).expanduser().resolve()
    folders = {
        (root / "public" / "ok_archive").resolve(): "OK",
        (root / "public" / "ng_archive").resolve(): "NG",
    }
    if target.parent not in folders or target.suffix.lower() != ".png":
        raise ValueError("Selecione um PNG dentro de ok_archive ou ng_archive")
    if not target.is_file():
        raise ValueError("PNG original não encontrado")
    scope = archive_category_from_filename(target)
    if not scope or not scope.endswith(("_SIDE", "_TOP", "_MID")):
        raise ValueError("Nome do PNG não identifica categoria/luz AOI")
    size = target.stat().st_size
    if size <= 0 or size > 75 * 1024 * 1024:
        raise ValueError("Tamanho de PNG inválido")
    return {
        "schema": SOURCE_SCHEMA,
        "source": str(target.relative_to(root)).replace("\\", "/"),
        "label": folders[target.parent],
        "light": scope.rsplit("_", 1)[-1],
        "category_hint": scope.rsplit("_", 1)[0],
        "file_size": size,
        "source_verified": True,
        "label_physically_verified": False,
        "aoi_exact_pair_verified": False,
        "diagnostic_only": True,
        "decision_authorized": False,
    }


def decode_archive_source(path: Path, *, root: Path = BASE_DIR) -> tuple[np.ndarray, dict]:
    identity = validate_archive_source(path, root=root)
    target = (Path(root).resolve() / identity["source"]).resolve()
    data = target.read_bytes()
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8
            or raw.ndim != 3 or raw.shape[-1] != 3
            or min(raw.shape[:2]) < 240):
        raise ValueError("PNG original não contém tela AOI válida")
    identity.update(
        file_sha256=sha256(data).hexdigest(),
        pixel_sha256=image_fingerprint(raw),
        shape=list(raw.shape),
    )
    return raw, identity


class ArchivePNGMonitor(ScreenMonitor):
    """Substitui somente a origem do frame; os sinais e o extrator são iguais."""

    def __init__(self, source: Path, *, root: Path = BASE_DIR):
        super().__init__()
        self.source = Path(source)
        self.root = Path(root)
        self.archive_direct_metadata = {}
        self._replay_no_debug = True

    def run(self):
        try:
            frame, meta = decode_archive_source(self.source, root=self.root)
            self.archive_direct_metadata = meta
            self.last_capture_received_at = time.perf_counter()
            self.process_external_image(frame)
        except (ValueError, OSError, cv2.error) as exc:
            self.log_updated.emit(
                "Replay PNG rejeitado: " + type(exc).__name__ + ": " + str(exc)
            )
        finally:
            self.running = False


__all__ = [
    "ArchivePNGMonitor", "decode_archive_source", "validate_archive_source",
    "SOURCE_SCHEMA",
]
