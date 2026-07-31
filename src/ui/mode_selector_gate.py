"""Bloqueia a troca de modo enquanto existe uma captura ativa."""

from __future__ import annotations

from PyQt6.QtCore import Qt


def install_mode_selector_gate(presenter_cls) -> None:
    if getattr(presenter_cls, "_mode_selector_gate_installed", False):
        return

    original_sync = presenter_cls.sync

    def sync(self, force: bool = False):
        result = original_sync(self, force=force)
        panel = self.panel
        combo = getattr(panel, "combo_mode", None)
        if combo is None:
            return result

        cycle_active = bool(getattr(panel, "capture_cycle_active", False))
        locked = bool(getattr(panel, "is_locked", False))
        local_pending = bool(getattr(panel, "local_capture_pending", False))
        enabled = not (cycle_active or locked or local_pending)

        try:
            combo.setEnabled(enabled)
            combo.setCursor(
                Qt.CursorShape.PointingHandCursor
                if enabled
                else Qt.CursorShape.ArrowCursor
            )
            combo.setToolTip(
                "Selecione o modo de operação."
                if enabled
                else "Finalize ou descarte a captura atual antes de trocar o modo."
            )
        except Exception as exc:
            print(f"Falha não fatal ao sincronizar seletor de modo: {exc}")
        return result

    presenter_cls.sync = sync
    presenter_cls._mode_selector_gate_installed = True


__all__ = ["install_mode_selector_gate"]
