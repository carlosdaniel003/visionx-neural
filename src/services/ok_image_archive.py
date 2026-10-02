"""Arquivo opcional de evidências visuais OK confirmadas pelo operador.

Quando habilitado, salva em background exatamente a mesma evidência completa
resolvida pelo botão "Copiar imagem" para o evento atual. Aceita capturas XP e
MSS, mas somente decisões humanas OK.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from src.config.settings import settings
from src.services.capture_evidence import (
    current_copy_image_event_id,
    current_copy_image_snapshot,
)
from src.services.image_archive_naming import (
    build_archive_filename,
    safe_archive_category,
)


OPERATOR_OK_SOURCES = {"button", "xp_keyboard"}


def build_ok_archive_filename(
    category: str,
    timestamp: datetime | None = None,
) -> str:
    """Usa exatamente o mesmo formato de nome do arquivo visual NG."""
    return build_archive_filename(category, timestamp)


class OKImageArchiveQueue:
    """Fila serial daemon para não bloquear julgamento nem próxima captura."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = Path(output_dir or settings.OK_ARCHIVE_DIR)
        self._queue: Queue[tuple[np.ndarray, str, datetime] | None] = Queue()
        self._worker = Thread(
            target=self._run,
            name="VisionXOKImageArchive",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        image: np.ndarray,
        category: str,
        timestamp: datetime | None = None,
    ) -> bool:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return False
        self._queue.put(
            (
                image.copy(),
                str(category or ""),
                timestamp or datetime.now(),
            )
        )
        return True

    def pending_count(self) -> int:
        return int(self._queue.unfinished_tasks)

    def wait_until_idle(self) -> None:
        self._queue.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                image, category, timestamp = item
                self.output_dir.mkdir(parents=True, exist_ok=True)
                target = self.output_dir / build_ok_archive_filename(
                    category,
                    timestamp,
                )
                if not cv2.imwrite(str(target), image):
                    print(f"Falha ao salvar arquivo visual OK: {target}")
            except Exception as exc:
                print(f"Falha não fatal ao arquivar imagem OK: {exc}")
            finally:
                self._queue.task_done()


def _sync_archive_control(panel) -> None:
    enabled = bool(getattr(panel, "ok_archive_enabled", False))
    button = getattr(panel, "btn_toggle_ok_archive", None)
    status = getattr(panel, "lbl_ok_archive_status", None)

    if button is not None:
        button.blockSignals(True)
        button.setChecked(enabled)
        button.setText(
            "Salvar imagens OK • ATIVADO"
            if enabled
            else "Salvar imagens OK • DESATIVADO"
        )
        button.setProperty("archiveEnabled", enabled)
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()
        button.blockSignals(False)

    if status is not None:
        status.setText(
            f"Ativado • {settings.OK_ARCHIVE_DIR}"
            if enabled
            else "Desativado • fluxo atual mantido"
        )
        status.setProperty("archiveEnabled", enabled)
        style = status.style()
        style.unpolish(status)
        style.polish(status)
        status.update()


def install_ok_image_archive(control_panel_cls) -> None:
    """Adiciona toggle e arquivo OK sem alterar a regra de decisão."""
    if getattr(control_panel_cls, "_ok_image_archive_installed", False):
        return

    original_init = control_panel_cls.__init__
    original_save_label = control_panel_cls.save_label

    def wrapped_init(self, *args, **kwargs):
        self.ok_archive_enabled = True
        self._ok_archive_queue = None
        self._ok_archive_last_event_id = ""
        original_init(self, *args, **kwargs)
        _sync_archive_control(self)

    def set_ok_archive_enabled(self, enabled: bool):
        self.ok_archive_enabled = bool(enabled)
        _sync_archive_control(self)
        state = "ativado" if self.ok_archive_enabled else "desativado"
        updater = getattr(self, "update_network_status", None)
        if callable(updater):
            updater(f"Arquivo visual OK {state}.")
        return self.ok_archive_enabled

    def wrapped_save_label(self, user_decision: str, source="button"):
        normalized = str(user_decision or "").strip().upper()
        normalized_source = str(source or "").strip().lower()
        current_cycle_source = str(
            getattr(self, "capture_cycle_source", "") or ""
        ).strip().lower()
        current_event_id = current_copy_image_event_id(self)
        current_category = str(
            (getattr(self, "current_aoi_info", {}) or {}).get(
                "category",
                "",
            )
            or ""
        ).strip()
        has_live_analysis = bool(
            getattr(self, "current_ng", None) is not None
            and getattr(self, "current_analysis", None) is not None
        )
        duplicate_event = bool(
            current_event_id
            and current_event_id
            == str(getattr(self, "_ok_archive_last_event_id", "") or "")
        )
        operator_decision = normalized_source in OPERATOR_OK_SOURCES
        supported_capture = current_cycle_source in {"network", "local"}

        should_archive = bool(
            normalized == "OK"
            and operator_decision
            and getattr(self, "ok_archive_enabled", False)
            and supported_capture
            and has_live_analysis
            and current_event_id
            and current_category
            and not duplicate_event
        )

        archive_image = None
        archive_category = ""
        if should_archive:
            archive_image = current_copy_image_snapshot(self)
            archive_category = current_category
            if archive_image is not None:
                # Reserva antes de concluir a decisão para impedir duplicação
                # caso o mesmo julgamento retorne pelo hook do XP.
                self._ok_archive_last_event_id = current_event_id

        if (
            normalized == "OK"
            and operator_decision
            and getattr(self, "ok_archive_enabled", False)
            and supported_capture
            and has_live_analysis
            and not duplicate_event
            and archive_image is None
        ):
            updater = getattr(self, "update_network_status", None)
            if callable(updater):
                if not current_category:
                    updater(
                        "OK não arquivado: a captura ativa não possui categoria "
                        "AOI. Nenhum arquivo SEM_CATEGORIA foi criado."
                    )
                else:
                    updater(
                        "OK não arquivado: a evidência de Copiar imagem do "
                        "evento atual não está disponível. Nenhum recorte "
                        "alternativo foi usado."
                    )

        result = original_save_label(
            self,
            user_decision,
            source=source,
        )

        if archive_image is not None:
            submitter = getattr(self, "_ok_archive_submitter", None)
            if callable(submitter):
                submitter(archive_image, archive_category)
            else:
                queue = getattr(self, "_ok_archive_queue", None)
                if queue is None:
                    queue = OKImageArchiveQueue()
                    self._ok_archive_queue = queue
                queue.submit(archive_image, archive_category)

        return result

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.set_ok_archive_enabled = set_ok_archive_enabled
    control_panel_cls._ok_image_archive_installed = True


__all__ = [
    "OKImageArchiveQueue",
    "OPERATOR_OK_SOURCES",
    "build_ok_archive_filename",
    "install_ok_image_archive",
    "safe_archive_category",
]
