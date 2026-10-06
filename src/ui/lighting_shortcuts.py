"""Atalhos de janela para controle manual das iluminações da AOI.

Usa QShortcut em vez de depender de keyPressEvent do painel. Assim as setas
continuam funcionando mesmo quando o foco está em botões, scroll areas ou
outros widgets filhos do ODIN.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut


LIGHTING_SHORTCUTS = (
    ("Left", "TOP", "btn_light_top"),
    ("Down", "SIDE", "btn_light_side"),
    ("Right", "MID", "btn_light_mid"),
)


def _activate_lighting(panel, light_mode: str, button_name: str) -> bool:
    """Aciona a iluminação somente quando o controle equivalente está habilitado."""
    button = getattr(panel, button_name, None)
    if button is not None and not button.isEnabled():
        return False

    change_lighting = getattr(panel, "change_lighting", None)
    if not callable(change_lighting):
        return False

    result = change_lighting(light_mode, "odin_keyboard")
    return result is not False


def install_lighting_shortcuts(panel) -> None:
    """Instala ←/↓/→ como atalhos válidos em toda a janela do ODIN."""
    if getattr(panel, "_lighting_shortcuts_installed", False):
        return

    shortcuts = []
    for sequence, light_mode, button_name in LIGHTING_SHORTCUTS:
        shortcut = QShortcut(QKeySequence(sequence), panel)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(
            lambda selected=light_mode, control=button_name: _activate_lighting(
                panel,
                selected,
                control,
            )
        )
        shortcuts.append(shortcut)

    panel._lighting_shortcuts = shortcuts
    panel._lighting_shortcuts_installed = True


__all__ = [
    "LIGHTING_SHORTCUTS",
    "_activate_lighting",
    "install_lighting_shortcuts",
]
