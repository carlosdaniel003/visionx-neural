"""Trava thread-safe para manter uma única captura ativa por ciclo.

Uma imagem reserva o ciclo. Novas imagens só podem ser aceitas depois que a
captura anterior for julgada como OK/NG ou explicitamente descartada.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class ImageCycleSnapshot:
    accepting_images: bool
    generation: int
    ignored_images: int


class ImageCycleGate:
    """Controla atomicamente a admissão de imagens da rede."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._accepting_images = True
        self._generation = 0
        self._ignored_images = 0

    def try_reserve(self) -> bool:
        """Reserva o ciclo para uma imagem, se não houver captura pendente."""
        with self._lock:
            if not self._accepting_images:
                self._ignored_images += 1
                return False
            self._accepting_images = False
            self._generation += 1
            return True

    def lock(self) -> None:
        """Fecha a entrada sem criar uma nova geração de imagem."""
        with self._lock:
            self._accepting_images = False

    def release(self) -> None:
        """Libera a próxima imagem após julgamento ou descarte."""
        with self._lock:
            self._accepting_images = True
            self._ignored_images = 0

    def note_ignored(self) -> int:
        with self._lock:
            self._ignored_images += 1
            return self._ignored_images

    def is_open(self) -> bool:
        with self._lock:
            return bool(self._accepting_images)

    def snapshot(self) -> ImageCycleSnapshot:
        with self._lock:
            return ImageCycleSnapshot(
                accepting_images=bool(self._accepting_images),
                generation=int(self._generation),
                ignored_images=int(self._ignored_images),
            )


__all__ = ["ImageCycleGate", "ImageCycleSnapshot"]
