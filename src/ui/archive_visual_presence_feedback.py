"""Consulta assíncrona do acervo XP; atualização visual ligada ao event_id."""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Thread

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from src.services.archive_visual_presence import (
    ArchiveVisualPresenceIndex, SCHEMA,
)


class SideArchivePresenceFeedback(QObject):
    """O julgamento e o comando XP nunca aguardam leitura de PNGs."""

    finished = pyqtSignal(object, object, object)

    def __init__(self, panel, *, index=None):
        super().__init__(panel)
        self.panel = panel
        self._index = index if index is not None else ArchiveVisualPresenceIndex()
        self._requests = Queue(maxsize=1)
        self.finished.connect(self._deliver)
        self._thread = Thread(
            target=self._run, name="ODINArchiveVisualPresence", daemon=True
        )
        self._thread.start()

    def inspect(self, analysis: dict, image, category: str, event_id: str) -> bool:
        if (not isinstance(analysis, dict)
                or not isinstance(image, np.ndarray) or image.size == 0
                or not str(event_id or "").strip()):
            return False
        detail = analysis.get("detail")
        if not isinstance(detail, dict):
            return False
        # O usuário verá o estado transitório em vez de SEM PAR EXATO
        # enquanto o arquivo realmente está sendo consultado.
        detail["side_archive_visual_presence"] = {
            "schema": SCHEMA, "status": "PENDING",
            "diagnostic_only": True, "decision_authorized": False,
        }
        task = (analysis, image.copy(), str(category or ""),
                str(event_id))
        try:
            self._requests.put_nowait(task)
        except Full:
            try:
                self._requests.get_nowait()
                self._requests.task_done()
            except Empty:
                pass
            self._requests.put_nowait(task)
        return True

    def _run(self):
        # Reutiliza a política Win32 já empregada no observador KNN:
        # reduz prioridade SOMENTE desta thread, nunca do processo XP/UI.
        try:
            from src.services.knn_archive_auto_memory import (
                _set_background_thread_mode,
            )
            _set_background_thread_mode(True)
        except (ImportError, AttributeError):
            pass
        while True:
            analysis, frame, category, event_id = self._requests.get()
            try:
                result = self._index.inspect(frame, category, "SIDE")
            except Exception as exc:
                result = {
                    "schema": SCHEMA, "status": "UNAVAILABLE", "label": "",
                    "diagnostic_only": True, "decision_authorized": False,
                    "error": type(exc).__name__,
                }
            finally:
                self._requests.task_done()
            self.finished.emit(event_id, analysis, result)

    def _deliver(self, event_id, analysis, result):
        """Executado na thread Qt; nunca atualiza UI de outra peça ou pós-0/1."""
        panel = self.panel
        if (getattr(panel, "current_analysis", None) is not analysis
                or str(getattr(panel, "adhesive_multilight_primary_event_id", "") or "")
                != event_id):
            return
        detail = analysis.get("detail")
        if not isinstance(detail, dict):
            return
        if detail.get("side_archive_visual_presence", {}).get("status") != "PENDING":
            return
        detail["side_archive_visual_presence"] = result
        # Atualiza apenas o painel de referências e os avisos vinculados;
        # não reexecuta especialistas, motor CNN, gate ou comandos ao XP.
        refresh = getattr(panel, "_update_reference_panel", None)
        if callable(refresh):
            try:
                refresh(analysis)
            except Exception as exc:
                print(f"Falha não fatal ao exibir memória visual SIDE: {exc}")


def submit_side_archive_presence(panel, analysis: dict, category: str) -> bool:
    """Dispara apenas para SIDE final; o frame deve ser da própria peça."""
    event_id = str(getattr(panel, "adhesive_multilight_primary_event_id", "") or "")
    source_event_id = str(getattr(panel, "adhesive_multilight_last_event_id", "") or "")
    frames = getattr(panel, "adhesive_multilight_last_source_frames", None)
    if (not event_id or event_id != source_event_id
            or not isinstance(frames, dict)):
        return False
    image = frames.get("SIDE")
    if not isinstance(image, np.ndarray) or image.size == 0:
        return False
    controller = getattr(panel, "_side_archive_presence_feedback", None)
    if controller is None:
        controller = SideArchivePresenceFeedback(panel)
        panel._side_archive_presence_feedback = controller
    return controller.inspect(analysis, image, category, event_id)


__all__ = ["SideArchivePresenceFeedback", "submit_side_archive_presence"]
