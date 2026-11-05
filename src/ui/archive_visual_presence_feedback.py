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
        # A consulta visual não depende de a fonte ser um QWidget.
        # Regressões da captura MSS também usam controladores sem Qt.
        super().__init__(panel if isinstance(panel, QObject) else None)
        self.panel = panel
        self._index = index if index is not None else ArchiveVisualPresenceIndex()
        self._requests = Queue(maxsize=1)
        self.finished.connect(self._deliver)
        self._thread = Thread(
            target=self._run, name="ODINArchiveVisualPresence", daemon=True
        )
        self._thread.start()

    def inspect(self, analysis: dict, image, category: str, event_id: str, *, frames=None, candidate_paths=None) -> bool:
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
        images = {"SIDE": image.copy()}
        if isinstance(frames, dict):
            for light in ("SIDE", "TOP", "MID"):
                frame = frames.get(light)
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    return False
                images[light] = frame.copy()
        task = (
            analysis, images, str(category or ""), str(event_id),
            tuple(candidate_paths or ()),
        )
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
            analysis, frames, category, event_id, candidates = self._requests.get()
            try:
                by_light = {
                    light: self._index.inspect(frame, category, light)
                    for light, frame in frames.items()
                }
                result = dict(by_light["SIDE"])
                result["by_light"] = by_light
                result["exact_frame_lights"] = [
                    light for light, entry in by_light.items()
                    if entry.get("status") == "EXACT_FRAME"
                ]
                # A câmera MSS pode fotografar um PNG de arquivo que
                # foi reescalado em um visualizador. A consulta do hash
                # continua primária; o alinhamento é diagnóstico distinto.
                if candidates and not result["exact_frame_lights"]:
                    from src.services.archive_rendered_source_match import (
                        match_rendered_archive,
                    )
                    result["rendered_archive_match"] = match_rendered_archive(
                        frames["SIDE"], category, list(candidates),
                    )
            except Exception as exc:
                result = {
                    "schema": SCHEMA, "status": "UNAVAILABLE", "label": "",
                    "diagnostic_only": True, "decision_authorized": False,
                    "error": type(exc).__name__,
                }
            finally:
                self._requests.task_done()
            try:
                self.finished.emit(event_id, analysis, result)
            except RuntimeError:
                # Janela Qt destruída ao encerrar o ODIN; o worker é daemon.
                return

    def _deliver(self, event_id, analysis, result):
        """Executado na thread Qt; nunca atualiza UI de outra peça ou pós-0/1."""
        panel = self.panel
        if getattr(panel, "current_analysis", None) is not analysis:
            return
        xp_event = str(getattr(panel, "adhesive_multilight_primary_event_id", "") or "")
        mss_event = str(getattr(panel, "local_capture_debug_event_id", "") or "")
        if event_id not in {xp_event, mss_event}:
            return
        detail = analysis.get("detail")
        if not isinstance(detail, dict):
            return
        if detail.get("side_archive_visual_presence", {}).get("status") != "PENDING":
            return
        detail["side_archive_visual_presence"] = result
        # A captura MSS usa o mesmo resultado do scanner do XP. O relatório
        # local também registra a consulta concluída, sem alterar o veredito.
        if event_id == mss_event:
            try:
                from src.services.capture_evidence import (
                    capture_debug_record, update_capture_debug_record,
                )
                record = capture_debug_record(panel)
                if (isinstance(record, dict)
                        and record.get("event_id") == mss_event
                        and record.get("source") == "local_mss"):
                    updated = dict(record)
                    decision = dict(updated.get("decision") or {})
                    decision["archive_visual_presence"] = result
                    updated["decision"] = decision
                    if update_capture_debug_record(panel, updated):
                        from src.ui.network_xp_debug import (
                            sync_network_debug_controls,
                        )
                        sync_network_debug_controls(panel)
            except (OSError, ValueError, TypeError) as exc:
                print(f"Falha não fatal ao registrar comparação MSS: {exc}")
        # Atualiza apenas o painel de referências e os avisos vinculados;
        # não reexecuta especialistas, motor CNN, gate ou comandos ao XP.
        refresh = getattr(panel, "_update_reference_panel", None)
        if callable(refresh):
            try:
                refresh(analysis)
            except Exception as exc:
                print(f"Falha não fatal ao exibir memória visual SIDE: {exc}")


def submit_side_archive_presence(
    panel, analysis: dict, category: str, *, all_lights: bool = False,
) -> bool:
    """Consulta a tela original da peça, em SIDE ou na trinca completa."""
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
    return controller.inspect(
        analysis, image, category, event_id,
        frames=frames if all_lights else None,
    )


def submit_local_mss_archive_presence(panel, analysis: dict, category: str) -> bool:
    """Usa o mesmo scanner XP para o frame MSS real, sem inferir rótulos.

    Consulta SIDE/TOP/MID porque a janela MSS não garante a iluminação
    correta do PNG aberto. Uma correspondência RGB exata continua exata
    somente do FRAME, jamais do par gabarito/teste.
    """
    event_id = str(getattr(panel, "local_capture_debug_event_id", "") or "")
    # O evento MSS preserva uma cópia imutável do frame ANTES de
    # desligar o monitor. Não depender do último frame mutável da thread.
    from src.services.capture_evidence import (
        capture_debug_record, capture_image_snapshot,
    )
    record = capture_debug_record(panel)
    if record.get("event_id") != event_id or record.get("source") != "local_mss":
        return False
    frame = capture_image_snapshot(panel)
    if (not event_id or not isinstance(analysis, dict)
            or not isinstance(frame, np.ndarray) or frame.size == 0):
        return False
    detail = analysis.get("detail")
    knn = detail.get("knn_c") if isinstance(detail, dict) else None
    candidates = []
    if isinstance(knn, dict):
        for key in ("neighbors_A", "neighbors_B"):
            for entry in knn.get(key, ())[:5]:
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    candidates.append(entry["path"])
    controller = getattr(panel, "_side_archive_presence_feedback", None)
    if controller is None:
        controller = SideArchivePresenceFeedback(panel)
        panel._side_archive_presence_feedback = controller
    return controller.inspect(
        analysis, frame, category, event_id,
        frames={light: frame for light in ("SIDE", "TOP", "MID")},
        candidate_paths=list(dict.fromkeys(candidates)),
    )


__all__ = [
    "SideArchivePresenceFeedback", "submit_side_archive_presence",
    "submit_local_mss_archive_presence",
]
