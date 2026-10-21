"""Leitura VISUAL, não decisória, das telas XP já presentes nos arquivos.

O scanner roda apenas em worker de UI. Usa a mesma assinatura conservadora da
deduplicação existente, sem acessar disco durante a decisão SIDE e sem usar
rótulos de arquivo como confirmação humana ou alterar o KNN exato.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.settings import settings
from src.services.image_archive_dedup import (
    ArchiveVisualIndex, archive_visual_descriptor,
)

SCHEMA = "visionx.side_archive_visual_presence.v1"
LABELS = ("OK", "NG")


class ArchiveVisualPresenceIndex:
    """Índice por sessão; carrega PNG apenas uma vez e atualiza novos arquivos."""

    def __init__(self, *, ok_dir: Path | None = None, ng_dir: Path | None = None):
        self._folders = {
            "OK": Path(ok_dir if ok_dir is not None else settings.OK_ARCHIVE_DIR),
            "NG": Path(ng_dir if ng_dir is not None else settings.NG_ARCHIVE_DIR),
        }
        self._indices = {label: ArchiveVisualIndex() for label in LABELS}
        self._seen_paths = {label: set() for label in LABELS}
        self._valid_counts = {label: 0 for label in LABELS}

    def _refresh(self) -> bool:
        """Somente no worker. Sem modificar, remover ou reenfileirar PNGs."""
        available = False
        for label, folder in self._folders.items():
            try:
                paths = folder.glob("*.png")
                for path in paths:
                    key = str(path.resolve())
                    if key in self._seen_paths[label]:
                        continue
                    # PNG truncado não vira prova e é reavaliado no futuro.
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if archive_visual_descriptor(image) is None:
                        continue
                    self._indices[label].register_existing(image, path)
                    self._seen_paths[label].add(key)
                    self._valid_counts[label] += 1
                available = available or folder.is_dir()
            except (OSError, PermissionError) as exc:
                # Falha parcial jamais vira "semelhante", "novo" ou voto OK/NG.
                return False
        return available

    def inspect(self, frame: np.ndarray, category: str, light: str) -> dict:
        """Apenas observação visual de mesma categoria/luz, independentemente de KNN."""
        result = {
            "schema": SCHEMA, "status": "UNAVAILABLE", "label": "",
            "lighting": str(light or "").upper(),
            "diagnostic_only": True, "decision_authorized": False,
            "source": "PUBLIC_ARCHIVE_PNG", "matched_labels": [],
            "indexed_ok": 0, "indexed_ng": 0,
        }
        if (result["lighting"] not in {"SIDE", "TOP", "MID"}
                or not str(category or "").strip()
                or archive_visual_descriptor(frame) is None):
            return result

        if not self._refresh():
            return result
        query = f"{category}_{result['lighting']}"
        matching = [
            label for label in LABELS
            if self._indices[label].contains_similar(frame, query)
        ]
        result["indexed_ok"] = self._valid_counts["OK"]
        result["indexed_ng"] = self._valid_counts["NG"]
        result["matched_labels"] = matching
        if len(matching) == 2:
            result["status"] = "AMBIGUOUS"
        elif len(matching) == 1:
            result["status"] = "SIMILAR"
            result["label"] = matching[0]
        else:
            result["status"] = "NO_SIMILAR"
        return result


__all__ = ["ArchiveVisualPresenceIndex", "SCHEMA"]
