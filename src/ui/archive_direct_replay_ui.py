"""Seletor de replay para PNGs reais; nunca fotografa o visualizador MSS."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog

from src.config.settings import BASE_DIR
from src.services.archive_direct_replay import (
    ArchivePNGMonitor, validate_archive_source,
)


def start_archive_replay(panel, path: Path, *, root: Path = BASE_DIR) -> bool:
    """Reusa o mesmo ciclo isolado de Teste e o monitor/extrator da AOI.

    A fábrica dura apenas uma captura; não contamina o próximo MSS.
    Nenhum botão, regra, resultado ou pacote XP é chamado diretamente.
    """
    if panel.combo_mode.currentText() != "Modo Teste":
        return False
    source = Path(path)
    validate_archive_source(source, root=root)
    if bool(getattr(panel, "local_capture_pending", False)):
        return False
    if bool(getattr(panel, "capture_cycle_local_transition", False)):
        return False
    if bool(getattr(panel, "capture_cycle_discarding", False)):
        return False

    def one_shot():
        # O monitor normal volta a ser o padrão antes de iniciar o
        # replay, inclusive se QThread falhar posteriormente.
        if getattr(panel, "_screen_monitor_factory", None) is one_shot:
            delattr(panel, "_screen_monitor_factory")
        return ArchivePNGMonitor(source, root=root)

    panel._screen_monitor_factory = one_shot
    try:
        accepted = bool(panel.start_monitoring())
    except Exception:
        if getattr(panel, "_screen_monitor_factory", None) is one_shot:
            delattr(panel, "_screen_monitor_factory")
        raise
    if not accepted and getattr(panel, "_screen_monitor_factory", None) is one_shot:
        delattr(panel, "_screen_monitor_factory")
    return accepted


def install_archive_direct_replay(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_archive_direct_replay_installed", False):
        return

    def open_archive_png_direct(self) -> bool:
        if self.combo_mode.currentText() != "Modo Teste":
            return False
        choice, _ = QFileDialog.getOpenFileName(
            self, "Analisar PNG original do acervo",
            str(BASE_DIR / "public" / "ok_archive"),
            "Imagens AOI (*.png)",
        )
        if not choice:
            return False
        try:
            return start_archive_replay(self, Path(choice))
        except (ValueError, OSError) as exc:
            self.update_network_status("Replay de acervo rejeitado: " + str(exc))
            return False

    control_panel_cls.open_archive_png_direct = open_archive_png_direct
    control_panel_cls._archive_direct_replay_installed = True


__all__ = ["start_archive_replay", "install_archive_direct_replay"]
