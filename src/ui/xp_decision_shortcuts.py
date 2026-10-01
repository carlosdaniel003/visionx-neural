"""Atalhos 0/1 do VisionX para decisões enviadas à AOI do Windows XP.

Os atalhos reutilizam os próprios QPushButtons de decisão. Assim, todas as
travas, wrappers de segurança, persistência e envio PRESS_0/PRESS_1 já
existentes continuam sendo a única regra de negócio.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut


def _activate_decision(panel, decision: str) -> bool:
    """Aciona a mesma decisão do botão somente quando ela está disponível."""
    normalized = str(decision or "").strip().upper()
    attribute = "btn_save_ok" if normalized == "OK" else "btn_save_ng"
    button = getattr(panel, attribute, None)

    if button is None or not button.isEnabled():
        return False

    button.click()

    show_feedback = getattr(panel, "show_decision_key_feedback", None)
    if callable(show_feedback):
        show_feedback(normalized, source="odin_keyboard")
    return True


def install_xp_decision_shortcuts(panel) -> None:
    """Instala 0/1 e Num0/Num1 como atalhos de decisão na janela VisionX."""
    if getattr(panel, "_xp_decision_shortcuts_installed", False):
        return

    shortcuts = []
    bindings = (
        ("0", "OK"),
        ("Num+0", "OK"),
        ("1", "NG"),
        ("Num+1", "NG"),
    )

    for sequence, decision in bindings:
        shortcut = QShortcut(QKeySequence(sequence), panel)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(
            lambda selected=decision: _activate_decision(panel, selected)
        )
        shortcuts.append(shortcut)

    # Manter referências explícitas evita coleta dos QShortcut durante a sessão.
    panel._xp_decision_shortcuts = shortcuts
    panel._xp_decision_shortcuts_installed = True


__all__ = [
    "_activate_decision",
    "install_xp_decision_shortcuts",
]
