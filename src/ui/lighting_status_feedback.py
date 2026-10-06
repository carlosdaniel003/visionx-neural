"""Overlay persistente da iluminação atual e ponte visual dos controles de luz."""

from __future__ import annotations

from functools import wraps

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


LIGHTING_STATUS_WIDTH = 300
LIGHTING_STATUS_HEIGHT = 88
LIGHTING_STATUS_MARGIN = 24
LIGHTING_STATUS_TOP_OFFSET = 184

LIGHTING_COMMAND_BY_MODE = {
    "TOP": "LEFT",
    "SIDE": "DOWN",
    "MID": "RIGHT",
}

LIGHTING_KEY_SYMBOL = {
    "TOP": "←",
    "SIDE": "↓",
    "MID": "→",
}

LIGHTING_STATUS_STYLESHEET = """
QFrame#lightingStatusFeedback {
    background-color: rgba(13, 13, 13, 248);
    border: 1px solid #f5c518;
    border-radius: 8px;
}
QLabel#lightingStatusHeader {
    color: #f5c518;
    background: transparent;
    border: none;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#lightingStatusValue {
    color: #f5f5f5;
    background: transparent;
    border: none;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#lightingStatusKey {
    color: #f5c518;
    background: transparent;
    border: none;
    font-size: 28px;
    font-weight: 900;
}
"""


class LightingStatusOverlay(QFrame):
    """Card fixo que mantém a iluminação atual visível para o operador."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.current_mode = ""

        self.setObjectName("lightingStatusFeedback")
        self.setFixedSize(LIGHTING_STATUS_WIDTH, LIGHTING_STATUS_HEIGHT)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(LIGHTING_STATUS_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 9, 14, 9)
        root.setSpacing(0)

        header = QLabel("ILUMINAÇÃO ATUAL")
        header.setObjectName("lightingStatusHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        self.value_label = QLabel("-")
        self.value_label.setObjectName("lightingStatusValue")
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.key_label = QLabel("")
        self.key_label.setObjectName("lightingStatusKey")
        self.key_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        body.addStretch(1)
        body.addWidget(self.value_label)
        body.addWidget(self.key_label)
        body.addStretch(1)

        root.addWidget(header)
        root.addLayout(body, 1)

        self.panel.installEventFilter(self)

    @staticmethod
    def _normalize(mode: str) -> str:
        normalized = str(mode or "").strip().upper()
        return normalized if normalized in LIGHTING_COMMAND_BY_MODE else ""

    def _top_right_position(self) -> QPoint:
        max_x = max(0, self.panel.width() - self.width())
        max_y = max(0, self.panel.height() - self.height())
        x = max(
            0,
            self.panel.width() - self.width() - LIGHTING_STATUS_MARGIN,
        )
        y = min(LIGHTING_STATUS_TOP_OFFSET, max_y)
        return QPoint(min(x, max_x), y)

    def reposition(self) -> None:
        self.move(self._top_right_position())
        self.raise_()

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
        }:
            QTimer.singleShot(0, self.reposition)
        return super().eventFilter(watched, event)

    def set_lighting(self, mode: str) -> bool:
        normalized = self._normalize(mode)
        if not normalized:
            return False

        self.current_mode = normalized
        self.value_label.setText(normalized)
        self.key_label.setText(LIGHTING_KEY_SYMBOL[normalized])
        self.reposition()
        self.show()
        return True


def install_lighting_status_feedback_hooks(control_panel_cls) -> None:
    """Atualiza os overlays somente quando a troca de iluminação foi aceita."""
    if getattr(control_panel_cls, "_lighting_status_feedback_hooks", False):
        return

    original_change_lighting = control_panel_cls.change_lighting

    @wraps(original_change_lighting)
    def change_lighting(self, light_mode: str, source: str):
        normalized_mode = str(light_mode or "").strip().upper()
        result = original_change_lighting(self, normalized_mode, source)
        if result is False:
            return result

        update_status = getattr(self, "update_lighting_status_feedback", None)
        if callable(update_status):
            update_status(normalized_mode)

        normalized_source = str(source or "").strip().lower()
        if normalized_source in {
            "local",
            "button",
            "odin_control",
            "odin_keyboard",
        }:
            show_key = getattr(self, "show_operational_key_feedback", None)
            command = LIGHTING_COMMAND_BY_MODE.get(normalized_mode)
            if callable(show_key) and command:
                visual_source = (
                    "odin_keyboard"
                    if normalized_source == "odin_keyboard"
                    else "odin_control"
                )
                show_key(command, source=visual_source)

        return result

    control_panel_cls.change_lighting = change_lighting
    control_panel_cls._lighting_status_feedback_hooks = True


def install_lighting_status_feedback(panel) -> None:
    """Cria o card persistente e o inicia no estado conhecido da interface."""
    if getattr(panel, "_lighting_status_feedback_installed", False):
        return

    overlay = LightingStatusOverlay(panel)
    panel.lighting_status_feedback = overlay
    panel.update_lighting_status_feedback = overlay.set_lighting
    panel._lighting_status_feedback_installed = True

    current = "SIDE"
    if hasattr(panel, "lbl_light_value"):
        current = panel.lbl_light_value.text().strip().upper() or "SIDE"
    overlay.set_lighting(current)


__all__ = [
    "LIGHTING_COMMAND_BY_MODE",
    "LIGHTING_KEY_SYMBOL",
    "LIGHTING_STATUS_HEIGHT",
    "LIGHTING_STATUS_MARGIN",
    "LIGHTING_STATUS_TOP_OFFSET",
    "LIGHTING_STATUS_WIDTH",
    "LightingStatusOverlay",
    "install_lighting_status_feedback",
    "install_lighting_status_feedback_hooks",
]
