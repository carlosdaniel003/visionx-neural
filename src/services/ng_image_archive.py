"""Arquivo opcional de evidências visuais NG.

Quando habilitado pelo operador, salva em background exatamente o mesmo frame
completo do Windows XP que o botão "Copiar imagem XP" disponibiliza para o
evento atual. Este arquivo é independente do dataset/KNN.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from queue import Queue
import re
from threading import Thread
import unicodedata

import cv2
import numpy as np

from src.config.settings import settings
from src.services.network_xp_frame import (
    network_xp_frame_snapshot,
    network_xp_record_event_id,
)


def safe_archive_category(value: str) -> str:
    """Converte a categoria para um trecho seguro de nome de arquivo."""
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_text.upper()).strip("_")
    return compact or "SEM_CATEGORIA"


def build_ng_archive_filename(
    category: str,
    timestamp: datetime | None = None,
) -> str:
    moment = timestamp or datetime.now()
    return (
        f"{moment:%d}d{moment:%m}m{moment:%Y}_"
        f"{moment:%H}h{moment:%M}min{moment:%S}s"
        f"{moment.microsecond // 1000:03d}ms_"
        f"{safe_archive_category(category)}.png"
    )


class NGImageArchiveQueue:
    """Fila serial daemon para não bloquear julgamento nem recepção da AOI."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = Path(output_dir or settings.NG_ARCHIVE_DIR)
        self._queue: Queue[tuple[np.ndarray, str, datetime] | None] = Queue()
        self._worker = Thread(
            target=self._run,
            name="VisionXNGImageArchive",
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
        """Apoio determinístico para testes/manutenção."""
        self._queue.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                image, category, timestamp = item
                self.output_dir.mkdir(parents=True, exist_ok=True)
                target = self.output_dir / build_ng_archive_filename(
                    category,
                    timestamp,
                )
                if not cv2.imwrite(str(target), image):
                    print(f"Falha ao salvar arquivo visual NG: {target}")
            except Exception as exc:
                print(f"Falha não fatal ao arquivar imagem NG: {exc}")
            finally:
                self._queue.task_done()


def _sync_archive_control(panel) -> None:
    enabled = bool(getattr(panel, "ng_archive_enabled", False))
    button = getattr(panel, "btn_toggle_ng_archive", None)
    status = getattr(panel, "lbl_ng_archive_status", None)

    if button is not None:
        button.blockSignals(True)
        button.setChecked(enabled)
        button.setText(
            "Salvar imagens NG • ATIVADO"
            if enabled
            else "Salvar imagens NG • DESATIVADO"
        )
        button.setProperty("archiveEnabled", enabled)
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()
        button.blockSignals(False)

    if status is not None:
        status.setText(
            f"Ativado • {settings.NG_ARCHIVE_DIR}"
            if enabled
            else "Desativado • fluxo atual mantido"
        )
        status.setProperty("archiveEnabled", enabled)
        style = status.style()
        style.unpolish(status)
        style.polish(status)
        status.update()


def install_ng_image_archive(control_panel_cls) -> None:
    """Adiciona toggle e arquivamento sem alterar a regra de decisão."""
    if getattr(control_panel_cls, "_ng_image_archive_installed", False):
        return

    original_init = control_panel_cls.__init__
    original_save_label = control_panel_cls.save_label

    def wrapped_init(self, *args, **kwargs):
        self.ng_archive_enabled = False
        self._ng_archive_queue = None
        self._ng_archive_last_event_id = ""
        original_init(self, *args, **kwargs)
        _sync_archive_control(self)

    def set_ng_archive_enabled(self, enabled: bool):
        self.ng_archive_enabled = bool(enabled)
        _sync_archive_control(self)
        state = "ativado" if self.ng_archive_enabled else "desativado"
        updater = getattr(self, "update_network_status", None)
        if callable(updater):
            updater(f"Arquivo visual NG {state}.")
        return self.ng_archive_enabled

    def wrapped_save_label(self, user_decision: str, source="button"):
        normalized = str(user_decision or "").strip().upper()

        current_cycle_source = str(
            getattr(self, "capture_cycle_source", "") or ""
        ).strip().lower()
        current_event_id = network_xp_record_event_id(self)
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
            == str(getattr(self, "_ng_archive_last_event_id", "") or "")
        )

        should_archive = bool(
            normalized == "NG"
            and getattr(self, "ng_archive_enabled", False)
            and current_cycle_source == "network"
            and has_live_analysis
            and current_event_id
            and current_category
            and not duplicate_event
        )

        archive_image = None
        archive_category = ""
        if should_archive:
            # Mesma fonte e mesma validação de event_id do botão
            # "Copiar imagem XP". Não usar current_ng como imagem: ele serve
            # somente como prova de que ainda existe uma captura ativa.
            archive_image = network_xp_frame_snapshot(self)
            archive_category = current_category

            # Reserva o evento ANTES de concluir o julgamento. O comando
            # PRESS_1 pode reaparecer pelo hook do XP como CMD_NG depois que a
            # interface já limpou current_aoi_info. Sem esta trava, o mesmo
            # frame podia ser salvo uma segunda vez como SEM_CATEGORIA.
            if archive_image is not None:
                self._ng_archive_last_event_id = current_event_id

        if (
            normalized == "NG"
            and getattr(self, "ng_archive_enabled", False)
            and current_cycle_source == "network"
            and has_live_analysis
            and not duplicate_event
            and archive_image is None
        ):
            updater = getattr(self, "update_network_status", None)
            if callable(updater):
                if not current_category:
                    updater(
                        "NG não arquivado: a captura ativa não possui categoria "
                        "AOI. Nenhum arquivo SEM_CATEGORIA foi criado."
                    )
                else:
                    updater(
                        "NG não arquivado: o frame XP do evento atual não está "
                        "disponível. Nenhum recorte alternativo foi usado."
                    )

        result = original_save_label(
            self,
            user_decision,
            source=source,
        )

        if archive_image is not None:
            submitter = getattr(self, "_ng_archive_submitter", None)
            if callable(submitter):
                submitter(archive_image, archive_category)
            else:
                queue = getattr(self, "_ng_archive_queue", None)
                if queue is None:
                    queue = NGImageArchiveQueue()
                    self._ng_archive_queue = queue
                queue.submit(archive_image, archive_category)

        return result

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.set_ng_archive_enabled = set_ng_archive_enabled
    control_panel_cls._ng_image_archive_installed = True


__all__ = [
    "NGImageArchiveQueue",
    "build_ng_archive_filename",
    "install_ng_image_archive",
    "safe_archive_category",
]
